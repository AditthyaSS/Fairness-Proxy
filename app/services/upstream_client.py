"""
app/services/upstream_client.py
=================================
Real upstream client using trained XGBoost model on UCI Adult dataset.
Mock client simulates a biased upstream for demo purposes.
Falls back to original HTTP client when TARGET_AI_API_KEY is set.
"""

from __future__ import annotations

import asyncio
import time
import json
from typing import Any, Dict, Optional

import httpx
import numpy as np
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.models.schemas import UpstreamInference

log = structlog.get_logger(__name__)

# UCI Adult feature schema
PROTECTED_COLS = ["sex", "race", "age", "native-country"]
MERIT_COLS = ["education", "hours-per-week", "workclass", "occupation"]
ALL_COLS = MERIT_COLS + PROTECTED_COLS


class XGBoostUpstreamClient:
    """
    Real upstream: runs trained XGBoost on the UCI Adult feature schema.
    Handles encoding, missing columns, and returns a calibrated probability.
    """

    def __init__(self) -> None:
        self._model = None
        self._encoders = None
        self._metadata = None
        self._load()

    def _load(self) -> None:
        from pathlib import Path
        import joblib

        model_path = Path("models/xgboost_upstream.joblib")
        if not model_path.exists():
            log.warning("upstream.model_not_found",
                        hint="run: python scripts/train_upstream_model.py")
            return

        self._model = joblib.load(model_path)
        self._encoders = joblib.load(Path("models/encoders.joblib"))
        with open(Path("models/metadata.json")) as f:
            self._metadata = json.load(f)

        log.info("upstream.xgboost_loaded",
                 accuracy=self._metadata.get("test_accuracy"),
                 auc=self._metadata.get("test_auc"))

    def _encode_payload(self, payload: Dict[str, Any]) -> np.ndarray:
        row = {}
        for col in ALL_COLS:
            val = payload.get(col, payload.get(col.replace("-", "_"), 0))
            if col in self._encoders:
                classes = self._encoders[col]
                val_str = str(val).strip()
                if val_str in classes:
                    row[col] = classes.index(val_str)
                else:
                    row[col] = 0
            else:
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = 0.0
        return np.array([[row[c] for c in ALL_COLS]], dtype=np.float32)

    async def close(self) -> None:
        pass

    async def infer(
        self, endpoint: str, payload: Dict[str, Any], method: str = "POST"
    ) -> UpstreamInference:
        t0 = time.monotonic()

        if self._model is None:
            score = float(np.random.beta(2, 2))
        else:
            X = self._encode_payload(payload)
            score = float(self._model.predict_proba(X)[0, 1])

        latency = (time.monotonic() - t0) * 1000

        return UpstreamInference(
            input_payload=payload,
            raw_response={"score": score, "model": "xgboost_adult"},
            decision_score=score,
            binary_decision=score >= 0.5,
            latency_ms=latency,
        )


class MockBiasedUpstreamClient:
    """
    Demo client: deliberately biased model that penalises
    race=Black, sex=Female by injecting a multiplier.
    Used when ?mock=true to demonstrate the proxy catching bias.
    """

    BIAS_PENALTIES = {
        "race": {"Black": -0.30, "Amer-Indian-Eskimo": -0.25,
                 "Other": -0.15, "Asian-Pac-Islander": -0.10},
        "sex": {"Female": -0.18},
        "native-country": {"Mexico": -0.15, "Guatemala": -0.10,
                           "El-Salvador": -0.10, "Puerto-Rico": -0.10},
    }

    async def close(self) -> None:
        pass

    async def infer(
        self, endpoint: str, payload: Dict[str, Any], method: str = "POST"
    ) -> UpstreamInference:
        t0 = time.monotonic()

        hours = float(payload.get("hours-per-week", payload.get("hours_per_week", 40)))
        age = float(payload.get("age", 35))
        base = 0.55 + (hours - 40) * 0.004 + (age - 35) * 0.002

        # Apply discriminatory penalties
        penalty = 0.0
        for attr, penalties in self.BIAS_PENALTIES.items():
            val = str(payload.get(attr, "")).strip()
            if val in penalties:
                penalty += penalties[val]

        score = float(np.clip(base + penalty + np.random.normal(0, 0.02), 0.05, 0.95))
        latency = (time.monotonic() - t0) * 1000

        return UpstreamInference(
            input_payload=payload,
            raw_response={"score": score, "model": "biased_mock",
                          "penalty_applied": penalty},
            decision_score=score,
            binary_decision=score >= 0.5,
            latency_ms=latency,
        )


class UpstreamClient:
    """
    Thin async wrapper around the target AI API (for production use
    when TARGET_AI_API_KEY is set).
    """

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.TARGET_AI_BASE_URL,
            headers={
                "Authorization": f"Bearer {settings.TARGET_AI_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=settings.TARGET_AI_TIMEOUT_S,
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def infer(
        self, endpoint: str, payload: Dict[str, Any], method: str = "POST"
    ) -> UpstreamInference:
        t0 = time.monotonic()
        if method.upper() == "POST":
            response = await self._client.post(endpoint, json=payload)
        else:
            response = await self._client.get(endpoint, params=payload)

        response.raise_for_status()
        raw = response.json()
        latency_ms = (time.monotonic() - t0) * 1000
        score, decision = self._extract_score(raw)

        return UpstreamInference(
            input_payload=payload,
            raw_response=raw,
            decision_score=score,
            binary_decision=decision,
            latency_ms=latency_ms,
        )

    def _extract_score(self, raw: Dict[str, Any]) -> tuple[float, bool]:
        for key in ("score", "probability", "confidence", "decision_score"):
            if key in raw:
                score = float(raw[key])
                return score, score >= 0.5
        for key in ("approved", "hired", "accepted"):
            if key in raw:
                score = 1.0 if raw[key] else 0.0
                return score, bool(raw[key])
        return 0.5, False


# Singletons
_real_client: Optional[XGBoostUpstreamClient] = None
_mock_client: Optional[MockBiasedUpstreamClient] = None
_http_client: Optional[UpstreamClient] = None


def get_upstream_client(mock: bool = False):
    global _real_client, _mock_client, _http_client

    if mock:
        if _mock_client is None:
            _mock_client = MockBiasedUpstreamClient()
        log.warning("upstream.using_mock_client")
        return _mock_client

    # Use XGBoost if no external API key configured
    if not settings.TARGET_AI_API_KEY:
        if _real_client is None:
            _real_client = XGBoostUpstreamClient()
        return _real_client

    # Use external HTTP client
    if _http_client is None:
        _http_client = UpstreamClient()
    return _http_client
