"""
app/db/session.py + ORM models
================================
Async SQLAlchemy setup with audit log table.
Run `alembic upgrade head` for production migrations.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, Text, Integer
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings

# ─── Engine ──────────────────────────────────────────────────────────────────
_is_sqlite = settings.DATABASE_URL.startswith("sqlite")
_engine_kwargs = dict(
    echo=settings.DEBUG,
)
if not _is_sqlite:
    _engine_kwargs.update(pool_pre_ping=True, pool_size=10, max_overflow=20)

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


# ─── ORM Models ──────────────────────────────────────────────────────────────

class AuditLog(Base):
    """
    Persistent record of every request processed by the Fairness Proxy.
    Used for RL training replay buffers and regulatory reporting.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String(64), unique=True, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    domain = Column(String(64), nullable=False)

    # Verdict & action
    verdict = Column(String(16), nullable=False)          # PASS | MITIGATE | BLOCK
    action_taken = Column(String(32), nullable=False)

    # Scores
    original_score = Column(Float, nullable=False)
    final_score = Column(Float, nullable=False)
    original_decision = Column(Boolean, nullable=False)
    final_decision = Column(Boolean, nullable=False)

    # Fairness metrics
    counterfactual_variance = Column(Float, nullable=False)
    total_protected_shap = Column(Float, nullable=False)
    rl_reward = Column(Float, nullable=False)
    max_score_delta = Column(Float, nullable=False)

    # Full JSON blobs (for audit / replay)
    xai_report_json = Column(Text, nullable=False)
    fairness_metrics_json = Column(Text, nullable=False)
    rl_decision_json = Column(Text, nullable=False)

    # PII-safe payload fingerprint
    payload_hash = Column(String(64), nullable=False)

    total_latency_ms = Column(Float)


class RLEpisode(Base):
    """
    One RL training episode: state → action → reward.
    Used to build the experience replay buffer for PPO training.
    """
    __tablename__ = "rl_episodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    request_id = Column(String(64), index=True)

    # State vector (serialised)
    state_json = Column(Text, nullable=False)

    # Action taken by RL agent
    action = Column(String(32), nullable=False)

    # Reward computed
    reward = Column(Float, nullable=False)
    cf_variance = Column(Float, nullable=False)
    protected_shap_sum = Column(Float, nullable=False)

    # Ground truth accuracy tracking
    true_label = Column(Float, nullable=True)
    original_error = Column(Float, nullable=True)
    corrected_error = Column(Float, nullable=True)
    accuracy_gain = Column(Float, nullable=True)
    bonus_reward = Column(Float, nullable=True)
    cumulative_accuracy = Column(Float, nullable=True)


# ─── Dependency ──────────────────────────────────────────────────────────────

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_context():
    """Async context manager for non-dependency-injection use (e.g. WebSocket)."""
    async with AsyncSessionLocal() as session:
        yield session

