"""
app/services/fairness_evaluator.py
=====================================
Fairness Metrics Computation
-------------------------------
Implements the core mathematical pipeline:

  1. Counterfactual Variance (L_CF):
       L_CF = (1/N) * Σ (ŷ - ŷ'_i)²
       Measures decision sensitivity to protected attribute changes.
       Target: L_CF ≈ 0 (counterfactual consistency / individual fairness).

  2. RL Reward Function R:
       R = -(α * L_CF + β * Σ|w_p|) + γ * accuracy_gain
       Punishes: (a) high counterfactual variance, (b) heavy reliance on
       protected features (high SHAP weights on P).
       Rewards: (c) accuracy improvement from ground truth feedback.

  3. Demographic Parity Ratio:
       DPR = P(ŷ=1|group_a) / P(ŷ=1|group_b)
       Fair iff DPR ≈ 1.0. Computed across twin groups.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import structlog

from app.core.config import settings
from app.models.schemas import FairnessMetrics, UpstreamInference, XAIReport

log = structlog.get_logger(__name__)


class FairnessEvaluator:
    """
    Stateless evaluator. Takes inference results + XAI report,
    returns a fully populated FairnessMetrics object.
    """

    def __init__(
        self,
        alpha: float = settings.RL_ALPHA,
        beta: float = settings.RL_BETA,
        gamma: float = 1.5,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def evaluate(
        self,
        original_inference: UpstreamInference,
        twin_inferences: List[UpstreamInference],
        xai_report: XAIReport,
        true_label: Optional[float] = None,
        final_score: Optional[float] = None,
    ) -> FairnessMetrics:
        """
        Compute all fairness metrics and the RL reward signal.

        Parameters
        ----------
        original_inference : UpstreamInference
            The upstream AI's response to the original (un-permuted) input.
        twin_inferences : List[UpstreamInference]
            Responses for each counterfactual twin x'_i.
        xai_report : XAIReport
            KernelSHAP output with per-feature importance weights.
        true_label : Optional[float]
            Ground truth label (0.0 or 1.0) from UCI Adult dataset.
        final_score : Optional[float]
            The RL-corrected score (after mitigation). If None, uses original.

        Returns
        -------
        FairnessMetrics
            All fairness statistics + RL reward.
        """
        y_hat: float = original_inference.decision_score
        twin_scores: List[float] = [inf.decision_score for inf in twin_inferences]

        # ── 1. Counterfactual Variance L_CF ─────────────────────────────
        # L_CF = (1/N) * sum((y_hat - y_hat'_i)^2)
        N = len(twin_scores)
        if N == 0:
            L_cf = 0.0
        else:
            squared_diffs = np.array([(y_hat - y_prime) ** 2 for y_prime in twin_scores])
            L_cf = float(np.mean(squared_diffs))

        # ── 2. Delta statistics ──────────────────────────────────────────
        if twin_scores:
            abs_deltas = np.abs(np.array(twin_scores) - y_hat)
            max_delta = float(np.max(abs_deltas))
            mean_delta = float(np.mean(abs_deltas))
        else:
            max_delta = 0.0
            mean_delta = 0.0

        # ── 3. Demographic Parity Ratio ──────────────────────────────────
        dpr: Optional[float] = None
        if N >= 2:
            half = N // 2
            group_a = [twin_inferences[i].binary_decision for i in range(half)]
            group_b = [twin_inferences[i].binary_decision for i in range(half, N)]

            rate_a = float(np.mean(group_a)) if group_a else 0.0
            rate_b = float(np.mean(group_b)) if group_b else 0.0

            if rate_b > 0:
                dpr = rate_a / rate_b
            elif rate_a == 0 and rate_b == 0:
                dpr = 1.0
            else:
                dpr = float("inf")

        # ── 4. Base RL Reward R ──────────────────────────────────────────
        protected_shap_sum = xai_report.total_protected_shap
        base_reward = -(self.alpha * L_cf + self.beta * protected_shap_sum)

        # ── 5. Ground Truth Accuracy Bonus ───────────────────────────────
        original_error: Optional[float] = None
        corrected_error: Optional[float] = None
        accuracy_gain: Optional[float] = None
        bonus_reward: float = 0.0

        if true_label is not None:
            _final = final_score if final_score is not None else y_hat
            original_error = abs(y_hat - true_label)
            corrected_error = abs(_final - true_label)
            accuracy_gain = original_error - corrected_error
            bonus_reward = self.gamma * accuracy_gain

        total_reward = base_reward + bonus_reward

        log.info(
            "fairness.metrics_computed",
            L_cf=round(L_cf, 6),
            max_delta=round(max_delta, 4),
            protected_shap=round(protected_shap_sum, 4),
            rl_reward=round(total_reward, 4),
            dpr=round(dpr, 4) if dpr is not None else None,
            accuracy_gain=round(accuracy_gain, 4) if accuracy_gain is not None else None,
            bonus_reward=round(bonus_reward, 4),
        )

        return FairnessMetrics(
            original_score=y_hat,
            twin_scores=twin_scores,
            counterfactual_variance=L_cf,
            max_score_delta=max_delta,
            mean_score_delta=mean_delta,
            demographic_parity_ratio=dpr,
            xai_report=xai_report,
            rl_reward=total_reward,
            original_error=original_error,
            corrected_error=corrected_error,
            accuracy_gain=accuracy_gain,
            bonus_reward=bonus_reward,
        )

    def summarise(self, metrics: FairnessMetrics) -> str:
        """Human-readable one-liner summary for logging and responses."""
        return (
            f"L_CF={metrics.counterfactual_variance:.4f}, "
            f"Δ_max={metrics.max_score_delta:.4f}, "
            f"SHAP_protected={metrics.xai_report.total_protected_shap:.4f}, "
            f"R={metrics.rl_reward:.4f}"
        )


# Module-level singleton
_evaluator: Optional[FairnessEvaluator] = None


def get_fairness_evaluator() -> FairnessEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = FairnessEvaluator()
    return _evaluator
