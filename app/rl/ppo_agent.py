"""
app/rl/ppo_agent.py
====================
Proximal Policy Optimisation (PPO) Fairness Corrector
-------------------------------------------------------
The RL agent is the "brain" of the Fairness Proxy. It observes the
fairness state and decides one of three actions:

  Action 0 → PASS              : Decision is fair; forward unchanged.
  Action 1 → MITIGATE          : Apply corrective re-scoring.
  Action 2 → BLOCK             : Decision too biased to forward at all.

State vector (7-dimensional):
  s = [
    y_hat,                   # Original decision score
    L_cf,                    # Counterfactual variance
    max_delta,               # Max |y_hat - y_hat'_i|
    mean_delta,              # Mean |y_hat - y_hat'_i|
    protected_shap_sum,      # Σ|w_p| (SHAP on protected attrs)
    top_protected_shap,      # Max single protected SHAP value
    dpr,                     # Demographic parity ratio (1.0 = fair)
  ]

Reward:
  R = -(α * L_CF + β * Σ|w_p|)   (defined in fairness_evaluator.py)

Training:
  We use a custom gymnasium.Env wrapper and a PyTorch policy network.
  Ray RLlib can be used to scale training across multiple workers.
  The agent is initially pre-trained on synthetic biased datasets and
  then fine-tuned online from the audit log replay buffer.

For INFERENCE (production), we load a pre-trained checkpoint.
If no checkpoint exists, we fall back to a rule-based heuristic agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import structlog

from app.core.config import settings
from app.models.schemas import (
    FairnessMetrics,
    FairnessVerdict,
    MitigationAction,
    RLDecision,
)

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Policy Network Architecture
# ─────────────────────────────────────────────────────────────────────────────

STATE_DIM = 7
ACTION_DIM = 3   # PASS, MITIGATE, BLOCK


class FairnessPolicy(nn.Module):
    """
    Compact MLP policy network for the PPO agent.
    Takes a 7-dim state vector; outputs action logits and value estimate.
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, ACTION_DIM)   # Policy head
        self.critic = nn.Linear(hidden_dim, 1)            # Value head

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def select_action(self, state: np.ndarray) -> int:
        """Greedy action selection (for inference)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            logits, _ = self.forward(s)
            return int(torch.argmax(logits, dim=-1).item())


# ─────────────────────────────────────────────────────────────────────────────
# RL Agent (Inference Mode)
# ─────────────────────────────────────────────────────────────────────────────

class FairnessRLAgent:
    """
    Production inference wrapper.
    Loads a pre-trained PPO checkpoint if available;
    otherwise runs a rule-based heuristic fallback.
    """

    ACTION_MAP = {0: FairnessVerdict.PASS, 1: FairnessVerdict.MITIGATE, 2: FairnessVerdict.BLOCK}

    def __init__(self) -> None:
        self.policy: Optional[FairnessPolicy] = None
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        checkpoint_path = Path(settings.RL_CHECKPOINT_PATH)
        if checkpoint_path.exists():
            try:
                self.policy = FairnessPolicy()
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                self.policy.load_state_dict(state_dict)
                self.policy.eval()
                log.info("rl_agent.checkpoint_loaded", path=str(checkpoint_path))
            except Exception as exc:
                log.warning("rl_agent.checkpoint_load_failed", error=str(exc))
                self.policy = None
        else:
            log.warning(
                "rl_agent.no_checkpoint",
                path=str(checkpoint_path),
                fallback="rule_based_heuristic",
            )

    def build_state_vector(self, metrics: FairnessMetrics) -> np.ndarray:
        """
        Encode FairnessMetrics → 7-dim normalised state vector for the policy network.
        """
        dpr = metrics.demographic_parity_ratio or 1.0
        dpr_normalised = abs(1.0 - min(dpr, 2.0)) / 1.0  # 0=fair, 1=completely unfair

        state = np.array([
            metrics.original_score,                             # [0, 1]
            min(metrics.counterfactual_variance, 1.0),          # [0, 1] clipped
            min(metrics.max_score_delta, 1.0),                  # [0, 1] clipped
            min(metrics.mean_score_delta, 1.0),                 # [0, 1] clipped
            min(metrics.xai_report.total_protected_shap, 1.0),  # [0, 1] clipped
            min(metrics.xai_report.top_protected_shap, 1.0),    # [0, 1] clipped
            dpr_normalised,                                      # [0, 1]
        ], dtype=np.float32)

        return state

    def decide(
        self,
        metrics: FairnessMetrics,
        original_payload: Dict[str, Any],
        protected_features: Dict[str, Any],
    ) -> RLDecision:
        """
        Main inference call: state → action → RLDecision.
        """
        state = self.build_state_vector(metrics)

        # Use trained policy if available, otherwise rule-based
        if self.policy is not None:
            action_idx = self.policy.select_action(state)
            verdict = self.ACTION_MAP[action_idx]
            log.debug("rl_agent.policy_decision", action=verdict, state=state.tolist())
        else:
            verdict = self._heuristic_decision(metrics)
            log.debug("rl_agent.heuristic_decision", verdict=verdict)

        return self._build_rl_decision(verdict, metrics, original_payload, protected_features)

    def _heuristic_decision(self, metrics: FairnessMetrics) -> FairnessVerdict:
        """
        Rule-based fallback when no trained checkpoint is available.
        Mirrors the thresholds in config.py.
        """
        L_cf = metrics.counterfactual_variance
        shap_p = metrics.xai_report.total_protected_shap

        if (
            shap_p >= settings.SHAP_PROTECTED_THRESHOLD
            or L_cf >= settings.CF_VARIANCE_THRESHOLD * 2
        ):
            return FairnessVerdict.BLOCK

        if (
            shap_p >= settings.SHAP_MITIGATE_LOW
            or L_cf >= settings.CF_VARIANCE_THRESHOLD
        ):
            return FairnessVerdict.MITIGATE

        return FairnessVerdict.PASS

    def _build_rl_decision(
        self,
        verdict: FairnessVerdict,
        metrics: FairnessMetrics,
        original_payload: Dict[str, Any],
        protected_features: Dict[str, Any],
    ) -> RLDecision:
        """
        Translate a verdict into a concrete RLDecision with corrected scores.
        """
        if verdict == FairnessVerdict.PASS:
            return RLDecision(
                verdict=FairnessVerdict.PASS,
                action_taken=MitigationAction.NONE,
                corrected_score=metrics.original_score,
                corrected_binary_decision=metrics.original_score >= 0.5,
                scrubbed_payload=None,
                explanation=(
                    f"No significant bias detected. "
                    f"L_CF={metrics.counterfactual_variance:.4f} < threshold, "
                    f"protected SHAP={metrics.xai_report.total_protected_shap:.4f}."
                ),
            )

        elif verdict == FairnessVerdict.MITIGATE:
            # ── Confidence Adjustment ─────────────────────────────────────
            # Re-score the decision using only the mean of twin scores
            # (which represent merit-only decisions since P is randomised).
            # This effectively removes the protected-attribute contribution.
            twin_mean = float(np.mean(metrics.twin_scores)) if metrics.twin_scores else metrics.original_score
            corrected = float(np.clip(twin_mean, 0.0, 1.0))

            # Also produce a scrubbed payload with protected attrs removed
            scrubbed = {k: v for k, v in original_payload.items() if k not in protected_features}

            return RLDecision(
                verdict=FairnessVerdict.MITIGATE,
                action_taken=MitigationAction.CONFIDENCE_ADJUSTMENT,
                corrected_score=corrected,
                corrected_binary_decision=corrected >= 0.5,
                scrubbed_payload=scrubbed,
                explanation=(
                    f"Bias detected (SHAP={metrics.xai_report.total_protected_shap:.4f}, "
                    f"L_CF={metrics.counterfactual_variance:.4f}). "
                    f"Score adjusted from {metrics.original_score:.4f} → {corrected:.4f} "
                    f"using counterfactual mean (protected attrs neutralised). "
                    f"Protected features: {list(protected_features.keys())}."
                ),
            )

        else:  # BLOCK
            return RLDecision(
                verdict=FairnessVerdict.BLOCK,
                action_taken=MitigationAction.NONE,
                corrected_score=None,
                corrected_binary_decision=None,
                scrubbed_payload=None,
                explanation=(
                    f"Decision BLOCKED. Severe bias detected: "
                    f"protected SHAP={metrics.xai_report.total_protected_shap:.4f} "
                    f"≥ threshold={settings.SHAP_PROTECTED_THRESHOLD}, "
                    f"L_CF={metrics.counterfactual_variance:.4f}. "
                    f"Discriminatory feature(s): "
                    f"{metrics.xai_report.top_protected_feature}. "
                    f"Request not forwarded to end user."
                ),
            )


# Module-level singleton
_rl_agent: Optional[FairnessRLAgent] = None


def get_rl_agent() -> FairnessRLAgent:
    global _rl_agent
    if _rl_agent is None:
        _rl_agent = FairnessRLAgent()
    return _rl_agent
