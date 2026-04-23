"""
app/api/routes/audit.py
========================
Audit & XAI Report API
------------------------
GET  /v1/audit/logs          — Paginated list of all audit log entries
GET  /v1/audit/logs/{id}     — Single audit log with full XAI report
GET  /v1/audit/stats         — Aggregate fairness statistics
GET  /v1/audit/rl-episodes   — RL training episode replay buffer
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func, case
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import AuditLog, RLEpisode, get_db

router = APIRouter()


# ─── Response schemas ─────────────────────────────────────────────────────────

class AuditLogSummary(BaseModel):
    id: int
    request_id: str
    domain: str
    verdict: str
    action_taken: str
    original_score: float
    final_score: float
    counterfactual_variance: float
    total_protected_shap: float
    rl_reward: float
    total_latency_ms: Optional[float]


class AuditLogDetail(AuditLogSummary):
    xai_report: Dict[str, Any]
    fairness_metrics: Dict[str, Any]
    rl_decision: Dict[str, Any]
    payload_hash: str


class FairnessStats(BaseModel):
    total_requests: int
    pass_count: int
    mitigate_count: int
    block_count: int
    pass_rate: float
    mitigate_rate: float
    block_rate: float
    avg_counterfactual_variance: float
    avg_protected_shap: float
    avg_rl_reward: float
    avg_latency_ms: float


class RLEpisodeSummary(BaseModel):
    id: int
    request_id: str
    action: str
    reward: float
    cf_variance: float
    protected_shap_sum: float
    state: List[float]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/logs", response_model=List[AuditLogSummary])
async def list_audit_logs(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    verdict: Optional[str] = Query(None, description="Filter by verdict: PASS | MITIGATE | BLOCK"),
    domain: Optional[str] = Query(None, description="Filter by domain"),
    db: AsyncSession = Depends(get_db),
) -> List[AuditLogSummary]:
    """Paginated list of audit log entries with optional filters."""
    stmt = select(AuditLog).order_by(AuditLog.timestamp.desc()).offset(skip).limit(limit)

    if verdict:
        stmt = stmt.where(AuditLog.verdict == verdict.upper())
    if domain:
        stmt = stmt.where(AuditLog.domain == domain.lower())

    result = await db.execute(stmt)
    logs = result.scalars().all()

    return [
        AuditLogSummary(
            id=log.id,
            request_id=log.request_id,
            domain=log.domain,
            verdict=log.verdict,
            action_taken=log.action_taken,
            original_score=log.original_score,
            final_score=log.final_score,
            counterfactual_variance=log.counterfactual_variance,
            total_protected_shap=log.total_protected_shap,
            rl_reward=log.rl_reward,
            total_latency_ms=log.total_latency_ms,
        )
        for log in logs
    ]


@router.get("/logs/{request_id}", response_model=AuditLogDetail)
async def get_audit_log(
    request_id: str,
    db: AsyncSession = Depends(get_db),
) -> AuditLogDetail:
    """Full audit log entry including XAI report and RL decision."""
    stmt = select(AuditLog).where(AuditLog.request_id == request_id)
    result = await db.execute(stmt)
    log_entry = result.scalar_one_or_none()

    if not log_entry:
        raise HTTPException(status_code=404, detail=f"Audit log not found: {request_id}")

    return AuditLogDetail(
        id=log_entry.id,
        request_id=log_entry.request_id,
        domain=log_entry.domain,
        verdict=log_entry.verdict,
        action_taken=log_entry.action_taken,
        original_score=log_entry.original_score,
        final_score=log_entry.final_score,
        counterfactual_variance=log_entry.counterfactual_variance,
        total_protected_shap=log_entry.total_protected_shap,
        rl_reward=log_entry.rl_reward,
        total_latency_ms=log_entry.total_latency_ms,
        xai_report=json.loads(log_entry.xai_report_json),
        fairness_metrics=json.loads(log_entry.fairness_metrics_json),
        rl_decision=json.loads(log_entry.rl_decision_json),
        payload_hash=log_entry.payload_hash,
    )


@router.get("/stats", response_model=FairnessStats)
async def fairness_stats(
    domain: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> FairnessStats:
    """Aggregate fairness statistics across all processed requests."""
    stmt = select(
        func.count(AuditLog.id).label("total"),
        func.sum(case((AuditLog.verdict == "PASS", 1), else_=0)).label("pass_c"),
        func.sum(case((AuditLog.verdict == "MITIGATE", 1), else_=0)).label("mitigate_c"),
        func.sum(case((AuditLog.verdict == "BLOCK", 1), else_=0)).label("block_c"),
        func.avg(AuditLog.counterfactual_variance).label("avg_cf"),
        func.avg(AuditLog.total_protected_shap).label("avg_shap"),
        func.avg(AuditLog.rl_reward).label("avg_reward"),
        func.avg(AuditLog.total_latency_ms).label("avg_latency"),
    )
    if domain:
        stmt = stmt.where(AuditLog.domain == domain.lower())

    result = await db.execute(stmt)
    row = result.one()

    total = row.total or 1  # Avoid division by zero
    return FairnessStats(
        total_requests=row.total or 0,
        pass_count=row.pass_c or 0,
        mitigate_count=row.mitigate_c or 0,
        block_count=row.block_c or 0,
        pass_rate=(row.pass_c or 0) / total,
        mitigate_rate=(row.mitigate_c or 0) / total,
        block_rate=(row.block_c or 0) / total,
        avg_counterfactual_variance=float(row.avg_cf or 0),
        avg_protected_shap=float(row.avg_shap or 0),
        avg_rl_reward=float(row.avg_reward or 0),
        avg_latency_ms=float(row.avg_latency or 0),
    )


@router.get("/rl-episodes", response_model=List[RLEpisodeSummary])
async def list_rl_episodes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
) -> List[RLEpisodeSummary]:
    """RL training replay buffer — used by the PPO trainer offline."""
    stmt = select(RLEpisode).order_by(RLEpisode.timestamp.desc()).offset(skip).limit(limit)
    result = await db.execute(stmt)
    episodes = result.scalars().all()

    return [
        RLEpisodeSummary(
            id=ep.id,
            request_id=ep.request_id,
            action=ep.action,
            reward=ep.reward,
            cf_variance=ep.cf_variance,
            protected_shap_sum=ep.protected_shap_sum,
            state=json.loads(ep.state_json),
        )
        for ep in episodes
    ]
