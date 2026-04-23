"""
app/rl/training_env.py
=======================
Gymnasium Environment for PPO Training
----------------------------------------
Wraps the fairness pipeline as a reinforcement learning environment.
The agent learns to minimise bias by deciding PASS / MITIGATE / BLOCK.

Usage (offline training from audit log replay buffer):
    python scripts/train_rl_agent.py

The environment samples experience from the AuditLog database,
reconstructing (state, reward) pairs to train the PPO policy offline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import structlog

from app.rl.ppo_agent import STATE_DIM, ACTION_DIM, FairnessPolicy
from app.core.config import settings

log = structlog.get_logger(__name__)


class FairnessEnv(gym.Env):
    """
    Custom Gymnasium environment for fairness-aware RL training.

    Observation space: Box(7,) ∈ [0, 1]
    Action space:      Discrete(3) = {PASS=0, MITIGATE=1, BLOCK=2}
    Reward:            R = -(α * L_CF + β * Σ|w_p|)

    Episodes are sampled from a replay buffer loaded from the audit log.
    """

    metadata = {"render_modes": []}

    def __init__(self, replay_buffer: list[Dict[str, Any]]) -> None:
        super().__init__()
        self.replay_buffer = replay_buffer
        self._episode_idx = 0

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)

        # Penalty for blocking a FAIR request (false positive cost)
        self.false_block_penalty = -0.5

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._episode_idx = (self._episode_idx + 1) % len(self.replay_buffer)
        episode = self.replay_buffer[self._episode_idx]
        self._current_episode = episode
        obs = np.array(episode["state"], dtype=np.float32)
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        episode = self._current_episode
        ground_truth_reward = episode["reward"]
        L_cf = episode["cf_variance"]
        shap_p = episode["protected_shap_sum"]

        # Compute action-specific reward
        # Action 0 = PASS
        if action == 0:
            if L_cf < settings.CF_VARIANCE_THRESHOLD and shap_p < settings.SHAP_MITIGATE_LOW:
                reward = 1.0  # Correct: fair decision, correctly passed
            else:
                reward = ground_truth_reward  # Punished for passing biased decision
        # Action 1 = MITIGATE
        elif action == 1:
            if shap_p >= settings.SHAP_MITIGATE_LOW:
                reward = 0.5 + ground_truth_reward * 0.5  # Partial credit
            else:
                reward = -0.2  # Unnecessary mitigation
        # Action 2 = BLOCK
        else:
            if shap_p >= settings.SHAP_PROTECTED_THRESHOLD:
                reward = 1.0  # Correct: severe bias, correctly blocked
            else:
                reward = self.false_block_penalty  # False positive

        obs = np.array(episode["state"], dtype=np.float32)
        return obs, float(reward), True, False, {"original_reward": ground_truth_reward}


# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer (lightweight single-machine implementation)
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    Minimal PPO training loop.
    For distributed training, swap this for Ray RLlib's PPOConfig.
    """

    def __init__(
        self,
        env: FairnessEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        n_epochs: int = 4,
        batch_size: int = 64,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.policy = FairnessPolicy()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.mse_loss = nn.MSELoss()

    def collect_rollout(self, n_steps: int = 512) -> Dict[str, list]:
        """Collect experience from the environment."""
        states, actions, rewards, log_probs, values = [], [], [], [], []

        obs, _ = self.env.reset()
        for _ in range(n_steps):
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            logits, value = self.policy(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_obs, reward, done, _, _ = self.env.step(action.item())

            states.append(obs)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())

            obs = next_obs if not done else self.env.reset()[0]

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "log_probs": log_probs,
            "values": values,
        }

    def compute_returns(self, rewards: list, values: list) -> list:
        """Compute discounted returns with GAE-lambda = 1.0 (Monte Carlo)."""
        returns = []
        G = 0.0
        for r, v in zip(reversed(rewards), reversed(values)):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns

    def update(self, rollout: Dict[str, list]) -> Dict[str, float]:
        """PPO update step."""
        states_t = torch.FloatTensor(rollout["states"])
        actions_t = torch.LongTensor(rollout["actions"])
        old_log_probs_t = torch.FloatTensor(rollout["log_probs"])
        returns_t = torch.FloatTensor(
            self.compute_returns(rollout["rewards"], rollout["values"])
        )
        values_t = torch.FloatTensor(rollout["values"])
        advantages_t = returns_t - values_t
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(self.n_epochs):
            logits, values_new = self.policy(states_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)

            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs_t)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            policy_loss = -torch.min(ratio * advantages_t, clipped_ratio * advantages_t).mean()

            # Value function loss
            value_loss = self.mse_loss(values_new.squeeze(), returns_t)

            loss = policy_loss + 0.5 * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            "policy_loss": total_policy_loss / self.n_epochs,
            "value_loss": total_value_loss / self.n_epochs,
        }

    def train(self, n_iterations: int = 100, save_path: str = settings.RL_CHECKPOINT_PATH) -> None:
        """Full training loop."""
        import os
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

        for i in range(n_iterations):
            rollout = self.collect_rollout()
            losses = self.update(rollout)

            if i % 10 == 0:
                mean_reward = float(np.mean(rollout["rewards"]))
                log.info(
                    "ppo.training_step",
                    iteration=i,
                    mean_reward=round(mean_reward, 4),
                    policy_loss=round(losses["policy_loss"], 4),
                    value_loss=round(losses["value_loss"], 4),
                )

        torch.save(self.policy.state_dict(), save_path)
        log.info("ppo.checkpoint_saved", path=save_path)
