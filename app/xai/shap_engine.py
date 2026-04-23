"""
app/xai/shap_engine.py
=======================
Model-Agnostic XAI via KernelSHAP
------------------------------------
Because we are treating the upstream AI as a BLACK BOX, we use
KernelSHAP (a model-agnostic variant of SHAP) via perturbation testing.

Algorithm:
  1. Build a local prediction function f(X) that calls the upstream API.
  2. Run KernelSHAP around the input point x.
  3. Extract the Shapley value w_p for each protected feature p ∈ P.
  4. Return a FeatureImportance list and aggregate protected SHAP score.

The RL reward function uses: β * Σ|w_p| to punish the proxy for
passing through inputs where protected attributes have high influence.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import shap
import structlog

from app.models.schemas import FeatureImportance, XAIReport

log = structlog.get_logger(__name__)


class KernelSHAPEngine:
    """
    Wraps shap.KernelExplainer with an async prediction function adapter.

    IMPORTANT: KernelSHAP is computationally expensive (O(2^M) worst-case).
    We use a background sample of size 20 and nsamples=100 for speed.
    In production, run this in a separate thread pool.
    """

    def __init__(
        self,
        n_background_samples: int = 20,
        n_shap_samples: int = 100,
    ) -> None:
        self.n_background = n_background_samples
        self.n_shap_samples = n_shap_samples

    async def compute_shap(
        self,
        original_payload: Dict[str, Any],
        merit_features: Dict[str, Any],
        protected_features: Dict[str, Any],
        prediction_fn: Callable[[Dict[str, Any]], float],
    ) -> XAIReport:
        """
        Async wrapper. Runs KernelSHAP in a thread to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._compute_shap_sync,
            original_payload,
            merit_features,
            protected_features,
            prediction_fn,
        )

    def _compute_shap_sync(
        self,
        original_payload: Dict[str, Any],
        merit_features: Dict[str, Any],
        protected_features: Dict[str, Any],
        prediction_fn: Callable[[Dict[str, Any]], float],
    ) -> XAIReport:
        """
        Synchronous KernelSHAP computation.

        We encode ALL features (merit + protected) into a numeric vector,
        then use KernelSHAP to attribute the prediction score to each feature.
        """
        all_features = {**merit_features, **protected_features}
        feature_names = list(all_features.keys())
        protected_set = set(protected_features.keys())

        # ── Encode: convert dict to numeric vector ────────────────────────
        def encode_row(row: Dict[str, Any]) -> np.ndarray:
            vec = []
            for k in feature_names:
                v = row.get(k, 0)
                if isinstance(v, bool):
                    vec.append(float(v))
                elif isinstance(v, (int, float)):
                    vec.append(float(v))
                else:
                    # Categorical → integer hash (deterministic)
                    vec.append(float(hash(str(v)) % 1000) / 1000.0)
            return np.array(vec, dtype=np.float64)

        # ── Build background dataset (perturb each feature independently) ─
        original_vec = encode_row(all_features)
        background = self._build_background(original_vec, self.n_background)

        # ── Prediction function for SHAP (takes 2D numpy array) ───────────
        def shap_predict(X: np.ndarray) -> np.ndarray:
            scores = []
            for row_vec in X:
                row_dict = {
                    feature_names[i]: self._decode_numeric(feature_names[i], row_vec[i], all_features)
                    for i in range(len(feature_names))
                }
                try:
                    score = prediction_fn(row_dict)
                except Exception:
                    score = 0.5
                scores.append(score)
            return np.array(scores, dtype=np.float64)

        # ── Run KernelSHAP ────────────────────────────────────────────────
        try:
            explainer = shap.KernelExplainer(shap_predict, background)
            shap_values = explainer.shap_values(
                original_vec.reshape(1, -1),
                nsamples=self.n_shap_samples,
                silent=True,
            )
            # shap_values shape: (1, num_features)
            shap_vals = shap_values[0] if len(shap_values.shape) == 2 else shap_values
        except Exception as exc:
            log.error("shap.computation_failed", error=str(exc))
            # Fall back to zero SHAP values
            shap_vals = np.zeros(len(feature_names))

        # ── Build feature importance list ─────────────────────────────────
        importances: List[FeatureImportance] = []
        for i, fname in enumerate(feature_names):
            sv = float(shap_vals[i])
            importances.append(FeatureImportance(
                feature_name=fname,
                shap_value=sv,
                abs_shap_value=abs(sv),
                is_protected=fname in protected_set,
            ))

        # Sort by absolute SHAP value descending
        importances.sort(key=lambda x: x.abs_shap_value, reverse=True)

        # ── Protected aggregate ───────────────────────────────────────────
        protected_importances = [fi for fi in importances if fi.is_protected]
        total_protected_shap = sum(fi.abs_shap_value for fi in protected_importances)

        top_protected: Optional[FeatureImportance] = (
            max(protected_importances, key=lambda x: x.abs_shap_value)
            if protected_importances
            else None
        )

        log.info(
            "shap.computed",
            total_protected_shap=round(total_protected_shap, 4),
            top_feature=importances[0].feature_name if importances else None,
        )

        return XAIReport(
            method="KernelSHAP",
            feature_importances=importances,
            total_protected_shap=total_protected_shap,
            top_protected_feature=top_protected.feature_name if top_protected else None,
            top_protected_shap=top_protected.abs_shap_value if top_protected else 0.0,
        )

    def _build_background(self, original_vec: np.ndarray, n: int) -> np.ndarray:
        """
        Build a small background dataset by adding Gaussian noise to the
        original vector. This approximates the marginal distribution.
        """
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 0.1, size=(n, len(original_vec)))
        background = np.clip(original_vec + noise, 0.0, 1.0)
        # Include the original itself
        background[0] = original_vec
        return background

    def _decode_numeric(
        self,
        key: str,
        numeric_val: float,
        original_features: Dict[str, Any],
    ) -> Any:
        """
        Convert a numeric SHAP value back to a usable type for the prediction function.
        We keep the original value's TYPE but scale the magnitude.
        """
        original = original_features.get(key, 0)
        if isinstance(original, bool):
            return numeric_val > 0.5
        if isinstance(original, int):
            return int(round(numeric_val * 1000))  # De-normalise
        if isinstance(original, float):
            return numeric_val * 1000.0
        return original  # Categorical: return original (SHAP perturbs encoding)


# Module-level singleton
_shap_engine: Optional[KernelSHAPEngine] = None


def get_shap_engine() -> KernelSHAPEngine:
    global _shap_engine
    if _shap_engine is None:
        _shap_engine = KernelSHAPEngine()
    return _shap_engine
