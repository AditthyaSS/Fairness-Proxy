"""
scripts/train_rl_agent.py
==========================
Offline PPO training script.

Loads experience from the AuditLog database (or generates synthetic
data if the database is empty), trains the FairnessPolicy network,
and saves a checkpoint.

Usage:
    python scripts/train_rl_agent.py --iterations 200 --synthetic 5000

Arguments:
    --iterations   : Number of PPO update iterations (default: 100)
    --synthetic    : Number of synthetic episodes to pre-train on if DB is empty
    --checkpoint   : Output path for the trained policy weights
"""

import argparse
import asyncio
import json
import sys
import os
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import structlog

from app.core.config import settings
from app.core.logging import configure_logging
from app.rl.training_env import FairnessEnv, PPOTrainer

configure_logging()
log = structlog.get_logger(__name__)


def generate_synthetic_episodes(n: int) -> list[dict]:
    """
    Generate synthetic fairness training episodes for cold-start PPO training.
    
    Covers three regimes:
      - Fair decisions  (low L_CF, low SHAP)     → reward ≈ 0
      - Mildly biased   (medium L_CF/SHAP)        → reward ≈ -0.3
      - Severely biased (high L_CF, high SHAP)    → reward ≈ -0.8
    """
    episodes = []
    rng = random.Random(42)

    for _ in range(n):
        regime = rng.choices(["fair", "mild", "severe"], weights=[0.4, 0.35, 0.25])[0]

        if regime == "fair":
            y_hat        = rng.uniform(0.3, 0.9)
            L_cf         = rng.uniform(0.0, 0.02)
            max_delta    = rng.uniform(0.0, 0.05)
            mean_delta   = max_delta * 0.6
            shap_p       = rng.uniform(0.0, 0.04)
            top_shap     = shap_p * 0.7
            dpr_dev      = rng.uniform(0.0, 0.05)

        elif regime == "mild":
            y_hat        = rng.uniform(0.3, 0.8)
            L_cf         = rng.uniform(0.02, 0.08)
            max_delta    = rng.uniform(0.05, 0.20)
            mean_delta   = max_delta * 0.6
            shap_p       = rng.uniform(0.05, 0.14)
            top_shap     = shap_p * 0.75
            dpr_dev      = rng.uniform(0.05, 0.25)

        else:  # severe
            y_hat        = rng.uniform(0.1, 0.7)
            L_cf         = rng.uniform(0.08, 0.4)
            max_delta    = rng.uniform(0.20, 0.60)
            mean_delta   = max_delta * 0.6
            shap_p       = rng.uniform(0.15, 0.55)
            top_shap     = shap_p * 0.8
            dpr_dev      = rng.uniform(0.25, 0.80)

        state = [
            float(np.clip(y_hat, 0, 1)),
            float(np.clip(L_cf, 0, 1)),
            float(np.clip(max_delta, 0, 1)),
            float(np.clip(mean_delta, 0, 1)),
            float(np.clip(shap_p, 0, 1)),
            float(np.clip(top_shap, 0, 1)),
            float(np.clip(dpr_dev, 0, 1)),
        ]

        reward = -(settings.RL_ALPHA * L_cf + settings.RL_BETA * shap_p)

        episodes.append({
            "state": state,
            "reward": reward,
            "cf_variance": L_cf,
            "protected_shap_sum": shap_p,
        })

    log.info("synthetic_episodes.generated", count=n)
    return episodes


async def load_db_episodes() -> list[dict]:
    """Load RL episodes from the audit database replay buffer."""
    from app.db.session import AsyncSessionLocal, RLEpisode
    from sqlalchemy import select

    episodes = []
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(RLEpisode))
        rows = result.scalars().all()
        for row in rows:
            episodes.append({
                "state": json.loads(row.state_json),
                "reward": row.reward,
                "cf_variance": row.cf_variance,
                "protected_shap_sum": row.protected_shap_sum,
            })

    log.info("db_episodes.loaded", count=len(episodes))
    return episodes


async def main(args: argparse.Namespace) -> None:
    # Try to load real episodes from DB first
    try:
        db_episodes = await load_db_episodes()
    except Exception as exc:
        log.warning("db_episodes.load_failed", error=str(exc))
        db_episodes = []

    # Fall back to synthetic data if DB is empty
    if len(db_episodes) < 100:
        log.info("using_synthetic_pretraining", synthetic_count=args.synthetic)
        episodes = generate_synthetic_episodes(args.synthetic) + db_episodes
    else:
        log.info("using_db_episodes", count=len(db_episodes))
        episodes = db_episodes

    # Build environment and trainer
    env = FairnessEnv(replay_buffer=episodes)
    trainer = PPOTrainer(env=env, lr=3e-4, gamma=0.99, clip_eps=0.2, n_epochs=4)

    log.info(
        "ppo_training.start",
        iterations=args.iterations,
        episodes=len(episodes),
        checkpoint=args.checkpoint,
    )

    trainer.train(n_iterations=args.iterations, save_path=args.checkpoint)
    log.info("ppo_training.complete", checkpoint=args.checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Fairness Proxy PPO agent")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--synthetic", type=int, default=5000)
    parser.add_argument("--checkpoint", type=str, default=settings.RL_CHECKPOINT_PATH)
    args = parser.parse_args()

    asyncio.run(main(args))
