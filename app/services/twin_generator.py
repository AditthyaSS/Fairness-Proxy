"""
app/services/twin_generator.py
================================
Counterfactual Twin Generation
--------------------------------
Produces N synthetic permutations of the original payload where:
  - Merit features M are LOCKED (unchanged).
  - Protected features P are drawn from pre-defined marginal distributions
    or exhaustively permuted (for binary attributes like gender).

Mathematical guarantee: for any twin x'_i:
  x'_i[M] == x[M]   (strict equality)
  x'_i[P] != x[P]   (at least one protected attribute differs)
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List

import numpy as np
import structlog

from app.core.config import settings
from app.models.schemas import CounterfactualTwin, TwinGenerationResult

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Marginal distributions for protected attributes.
# These represent population-level distributions, NOT any individual's value.
# Extending this dict adds support for new protected attributes automatically.
# ─────────────────────────────────────────────────────────────────────────────

PROTECTED_DISTRIBUTIONS: Dict[str, List[Any]] = {
    "gender": ["male", "female", "non-binary"],
    "sex": ["male", "female"],
    "race": ["White", "Black", "Hispanic", "Asian", "Native American", "Other"],
    "ethnicity": ["White", "Black", "Hispanic", "Asian", "Native American", "Other"],
    "age": list(range(22, 68, 5)),           # [22, 27, 32, 37, 42, 47, 52, 57, 62, 67]
    "date_of_birth": None,                   # Handled via age
    "nationality": ["US", "UK", "CA", "IN", "MX", "CN", "NG", "PH"],
    "marital_status": ["single", "married", "divorced", "widowed"],
    "familial_status": ["no_children", "one_child", "multiple_children"],
    "disability": [True, False],
    "religion": ["Christianity", "Islam", "Judaism", "Hinduism", "Buddhism", "None"],
    "zip_code": ["10001", "90210", "60601", "77001", "85001", "30301"],
}

# Attributes to treat as continuous noise-perturbed rather than categorical
CONTINUOUS_PROTECTED: Dict[str, Dict[str, float]] = {
    "age": {"std": 10.0, "min": 18.0, "max": 80.0},
}


class TwinGenerator:
    """
    Generates counterfactual twins by permuting protected features
    while strictly preserving merit features.
    """

    def __init__(self, num_twins: int = settings.NUM_TWINS, seed: int = settings.TWIN_SEED):
        self.num_twins = num_twins
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def generate(
        self,
        original_payload: Dict[str, Any],
        merit_features: Dict[str, Any],
        protected_features: Dict[str, Any],
    ) -> TwinGenerationResult:
        """
        Main entry point. Returns original + N twins with swapped protected attrs.

        Strategy:
          Twin 0  → Single feature swap (isolates one protected attribute)
          Twins 1-N → Full random permutation from marginal distributions
        """
        if not protected_features:
            log.warning("twin_generator.no_protected_features")
            # Return identity twins (no swap possible)
            twins = [
                CounterfactualTwin(
                    twin_id=i,
                    payload=copy.deepcopy(original_payload),
                    swapped_features={},
                )
                for i in range(self.num_twins)
            ]
            return TwinGenerationResult(original=original_payload, twins=twins)

        twins: List[CounterfactualTwin] = []

        # Twin 0: single-attribute swap (most interpretable)
        first_key = next(iter(protected_features))
        swap_0 = self._swap_one(first_key, protected_features[first_key])
        twin_0_payload = {**merit_features, **protected_features, **swap_0}
        twins.append(CounterfactualTwin(
            twin_id=0,
            payload=twin_0_payload,
            swapped_features=swap_0,
        ))

        # Twins 1-N: full random permutations
        for i in range(1, self.num_twins):
            swapped = self._permute_all(protected_features)
            twin_payload = {**merit_features, **protected_features, **swapped}
            twins.append(CounterfactualTwin(
                twin_id=i,
                payload=twin_payload,
                swapped_features=swapped,
            ))

        log.info(
            "twins.generated",
            num_twins=len(twins),
            protected_keys=list(protected_features.keys()),
        )
        return TwinGenerationResult(original=original_payload, twins=twins)

    def _swap_one(self, key: str, original_value: Any) -> Dict[str, Any]:
        """Swap a single protected attribute to a different value."""
        new_val = self._sample_different(key, original_value)
        return {key: new_val}

    def _permute_all(self, protected_features: Dict[str, Any]) -> Dict[str, Any]:
        """Permute ALL protected attributes simultaneously."""
        permuted: Dict[str, Any] = {}
        for key, original_value in protected_features.items():
            permuted[key] = self._sample_different(key, original_value)
        return permuted

    def _sample_different(self, key: str, original_value: Any) -> Any:
        """
        Sample a value from the marginal distribution for `key`
        that is strictly different from `original_value`.
        """
        key_lower = key.lower()

        # Continuous numeric perturbation
        if key_lower in CONTINUOUS_PROTECTED:
            cfg = CONTINUOUS_PROTECTED[key_lower]
            try:
                orig_float = float(original_value)
                noise = self.np_rng.normal(0, cfg["std"])
                new_val = np.clip(orig_float + noise, cfg["min"], cfg["max"])
                # Ensure strictly different by at least 2 years for age
                if abs(new_val - orig_float) < 2.0:
                    new_val = orig_float + (cfg["std"] if noise >= 0 else -cfg["std"])
                    new_val = np.clip(new_val, cfg["min"], cfg["max"])
                return int(round(float(new_val)))
            except (ValueError, TypeError):
                pass

        # Categorical sampling
        candidates = PROTECTED_DISTRIBUTIONS.get(key_lower)

        if candidates is None:
            # Unknown protected attribute — try boolean flip or string mangle
            if isinstance(original_value, bool):
                return not original_value
            if isinstance(original_value, (int, float)):
                # Perturb by ±20%
                delta = max(1, abs(original_value) * 0.2)
                return type(original_value)(original_value + self.rng.choice([-delta, delta]))
            # String: just mark as [REDACTED] to strip the value
            return f"[PERMUTED_{key.upper()}]"

        # Filter out original value and sample
        alternatives = [c for c in candidates if str(c).lower() != str(original_value).lower()]
        if not alternatives:
            alternatives = candidates  # Edge case: only one value in distribution

        chosen = self.rng.choice(alternatives)
        # Match original type where possible
        if isinstance(original_value, int) and isinstance(chosen, (int, float)):
            return int(chosen)
        return chosen


# Module-level singleton
_twin_generator: TwinGenerator | None = None


def get_twin_generator() -> TwinGenerator:
    global _twin_generator
    if _twin_generator is None:
        _twin_generator = TwinGenerator()
    return _twin_generator
