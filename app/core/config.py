"""
app/core/config.py
==================
Centralised settings loaded from environment variables / .env file.
Every tunable constant in the system is declared here.
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )

    # ── Application ────────────────────────────────────────────────────────
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    CORS_ORIGINS: List[str] = ["*"]

    # ── Database ───────────────────────────────────────────────────────────
    DATABASE_URL: str = "sqlite+aiosqlite:///./fairness_proxy.db"

    # ── Redis (rate-limiting + caching) ───────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── Third-party AI target ─────────────────────────────────────────────
    # The upstream AI endpoint that this proxy intercepts.
    TARGET_AI_BASE_URL: str = "https://api.openai.com/v1"
    TARGET_AI_API_KEY: str = ""
    TARGET_AI_TIMEOUT_S: float = 30.0

    # ── Counterfactual twin generation ────────────────────────────────────
    NUM_TWINS: int = 5          # N synthetic permutations per request
    TWIN_SEED: int = 42         # RNG seed for reproducibility

    # ── Fairness thresholds ───────────────────────────────────────────────
    # Counterfactual variance above this triggers MITIGATION or BLOCK.
    CF_VARIANCE_THRESHOLD: float = 0.05
    # Aggregate absolute SHAP weight on protected attrs above this → BLOCK.
    SHAP_PROTECTED_THRESHOLD: float = 0.15
    # If SHAP is between LOW and HIGH → MITIGATE; above HIGH → BLOCK.
    SHAP_MITIGATE_LOW: float = 0.05
    SHAP_MITIGATE_HIGH: float = 0.15

    # ── RL Agent (PPO) ────────────────────────────────────────────────────
    RL_CHECKPOINT_PATH: str = "checkpoints/ppo_fairness_agent"
    RL_ALPHA: float = 1.0       # Weight on L_CF in reward
    RL_BETA: float = 2.0        # Weight on SHAP protected penalty

    # ── Zero-shot domain classifier ───────────────────────────────────────
    CLASSIFIER_MODEL: str = "cross-encoder/nli-deberta-v3-small"
    CANDIDATE_DOMAINS: List[str] = [
        "loan_approval",
        "job_application",
        "healthcare_triage",
        "insurance_underwriting",
        "credit_scoring",
        "rental_application",
    ]

    # ── Ngrok Tunnel ──────────────────────────────────────────────────────
    NGROK_AUTHTOKEN: str = ""
    NGROK_DOMAIN: str = ""
    NGROK_ENABLED: bool = False

    # ── Logging ───────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"   # "json" | "console"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
