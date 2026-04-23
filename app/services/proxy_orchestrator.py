"""
app/services/proxy_orchestrator.py
=====================================
The Fairness Proxy Orchestrator
---------------------------------
This is the central pipeline that coordinates all 5 stages:

  Stage 1 → Dynamic Schema Classifier
  Stage 2 → Counterfactual Twin Generator
  Stage 3 → Parallel Upstream API Execution
  Stage 4 → SHAP XAI + Fairness Evaluation + RL Decision
  Stage 5 → Audit logging + Response assembly

All stages are wired together here. The router delegates
all POST /v1/proxy/infer requests to this service.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.config import settings
from app.db.session import AuditLog, RLEpisode
from app.models.schemas import (
    ClassifiedSchema,
    FairnessMetrics,
    ProxyRequest,
    ProxyResponse,
    RLDecision,
    TwinGenerationResult,
    UpstreamInference,
)
from app.rl.ppo_agent import get_rl_agent
from app.services.fairness_evaluator import get_fairness_evaluator
from app.services.schema_classifier import get_schema_classifier
from app.services.twin_generator import get_twin_generator
from app.services.upstream_client import get_upstream_client
from app.xai.shap_engine import get_shap_engine

log = structlog.get_logger(__name__)


class ProxyOrchestrator:
    """
    Stateless orchestrator. Each request spawns a full pipeline execution.
    """

    async def process(
        self,
        request: ProxyRequest,
        db: AsyncSession,
        mock_upstream: bool = False,
    ) -> ProxyResponse:
        t_start = time.monotonic()

        log.info(
            "proxy.request_received",
            request_id=request.request_id,
            domain_hint=request.domain,
            payload_keys=list(request.payload.keys()),
        )

        # ── Stage 1: Dynamic Schema Classification ────────────────────────
        classifier = get_schema_classifier()
        schema: ClassifiedSchema = await classifier.classify(
            request.payload,
            domain_hint=request.domain,
        )

        # ── Stage 2: Counterfactual Twin Generation ───────────────────────
        generator = get_twin_generator()
        twin_result: TwinGenerationResult = generator.generate(
            original_payload=request.payload,
            merit_features=schema.merit_features,
            protected_features=schema.protected_features,
        )

        # ── Stage 3: Parallel Upstream API Execution ──────────────────────
        upstream = get_upstream_client(mock=mock_upstream)

        # Build list of all payloads to infer (original + all twins)
        all_payloads: List[Dict[str, Any]] = [twin_result.original] + [
            t.payload for t in twin_result.twins
        ]

        inference_tasks = [
            upstream.infer(
                endpoint=request.target_endpoint,
                payload=p,
                method=request.target_method,
            )
            for p in all_payloads
        ]

        all_inferences: List[UpstreamInference] = await asyncio.gather(*inference_tasks)
        original_inference = all_inferences[0]
        twin_inferences = all_inferences[1:]

        log.info(
            "upstream.inferences_complete",
            original_score=round(original_inference.decision_score, 4),
            twin_scores=[round(i.decision_score, 4) for i in twin_inferences],
        )

        # ── Stage 4a: SHAP / XAI Extraction ──────────────────────────────
        shap_engine = get_shap_engine()

        # Build a synchronous prediction function for KernelSHAP
        # (SHAP calls this many times; we use the mock for speed in training)
        upstream_sync = get_upstream_client(mock=True)  # Always use mock for SHAP perturbations

        def sync_predict(payload: Dict[str, Any]) -> float:
            """Blocking wrapper used by KernelSHAP internals."""
            loop = asyncio.new_event_loop()
            try:
                inf = loop.run_until_complete(
                    upstream_sync.infer(request.target_endpoint, payload)
                )
                return inf.decision_score
            finally:
                loop.close()

        xai_report = await shap_engine.compute_shap(
            original_payload=request.payload,
            merit_features=schema.merit_features,
            protected_features=schema.protected_features,
            prediction_fn=sync_predict,
        )

        # ── Stage 4b: Fairness Metrics + RL Reward ────────────────────────
        evaluator = get_fairness_evaluator()

        # Pre-compute final_score for accuracy tracking
        # The RL agent hasn't decided yet, so we estimate using mean-twin correction
        twin_mean = sum(inf.decision_score for inf in twin_inferences) / max(len(twin_inferences), 1)
        estimated_final = (original_inference.decision_score + twin_mean) / 2

        fairness_metrics: FairnessMetrics = evaluator.evaluate(
            original_inference=original_inference,
            twin_inferences=twin_inferences,
            xai_report=xai_report,
            true_label=request.true_label,
            final_score=estimated_final,
        )

        # ── Stage 4c: RL Agent Decision ───────────────────────────────────
        rl_agent = get_rl_agent()
        rl_decision: RLDecision = rl_agent.decide(
            metrics=fairness_metrics,
            original_payload=request.payload,
            protected_features=schema.protected_features,
        )

        # ── Stage 5: Audit Logging ────────────────────────────────────────
        final_score = (
            rl_decision.corrected_score
            if rl_decision.corrected_score is not None
            else original_inference.decision_score
        )
        final_decision = (
            rl_decision.corrected_binary_decision
            if rl_decision.corrected_binary_decision is not None
            else original_inference.binary_decision
        )

        total_latency_ms = (time.monotonic() - t_start) * 1000

        await self._persist_audit_log(
            db=db,
            request=request,
            schema=schema,
            fairness_metrics=fairness_metrics,
            rl_decision=rl_decision,
            original_inference=original_inference,
            final_score=final_score,
            final_decision=final_decision,
            total_latency_ms=total_latency_ms,
        )

        log.info(
            "proxy.request_complete",
            request_id=request.request_id,
            verdict=rl_decision.verdict,
            original_score=round(original_inference.decision_score, 4),
            final_score=round(final_score, 4),
            latency_ms=round(total_latency_ms, 1),
        )

        return ProxyResponse(
            request_id=request.request_id,
            domain=schema.domain,
            domain_confidence=schema.domain_confidence,
            original_decision=original_inference.binary_decision,
            original_score=original_inference.decision_score,
            final_decision=final_decision,
            final_score=final_score,
            fairness_metrics=fairness_metrics,
            rl_decision=rl_decision,
            classified_schema=schema,
            twin_generation=twin_result,
            upstream_inferences=all_inferences,
            accuracy_gain=fairness_metrics.accuracy_gain,
            cumulative_accuracy_improvement=await self._get_cumulative_accuracy(db),
            total_latency_ms=total_latency_ms,
            mitigation_applied=rl_decision.verdict != "PASS",
        )

    async def _persist_audit_log(
        self,
        db: AsyncSession,
        request: ProxyRequest,
        schema: ClassifiedSchema,
        fairness_metrics: FairnessMetrics,
        rl_decision: RLDecision,
        original_inference: UpstreamInference,
        final_score: float,
        final_decision: bool,
        total_latency_ms: float,
    ) -> None:
        """Persist a complete audit record and RL episode to the database."""
        payload_hash = hashlib.sha256(
            json.dumps(request.payload, sort_keys=True).encode()
        ).hexdigest()

        # Audit log entry
        audit_entry = AuditLog(
            request_id=request.request_id,
            domain=schema.domain.value,
            verdict=rl_decision.verdict.value,
            action_taken=rl_decision.action_taken.value,
            original_score=original_inference.decision_score,
            final_score=final_score,
            original_decision=original_inference.binary_decision,
            final_decision=final_decision,
            counterfactual_variance=fairness_metrics.counterfactual_variance,
            total_protected_shap=fairness_metrics.xai_report.total_protected_shap,
            rl_reward=fairness_metrics.rl_reward,
            max_score_delta=fairness_metrics.max_score_delta,
            xai_report_json=fairness_metrics.xai_report.model_dump_json(),
            fairness_metrics_json=fairness_metrics.model_dump_json(),
            rl_decision_json=rl_decision.model_dump_json(),
            payload_hash=payload_hash,
            total_latency_ms=total_latency_ms,
        )

        # RL episode for training replay buffer
        rl_agent = get_rl_agent()
        from app.rl.ppo_agent import FairnessRLAgent
        state_vec = rl_agent.build_state_vector(fairness_metrics).tolist()

        rl_episode = RLEpisode(
            request_id=request.request_id,
            state_json=json.dumps(state_vec),
            action=rl_decision.verdict.value,
            reward=fairness_metrics.rl_reward,
            cf_variance=fairness_metrics.counterfactual_variance,
            protected_shap_sum=fairness_metrics.xai_report.total_protected_shap,
            true_label=request.true_label,
            original_error=fairness_metrics.original_error,
            corrected_error=fairness_metrics.corrected_error,
            accuracy_gain=fairness_metrics.accuracy_gain,
            bonus_reward=fairness_metrics.bonus_reward,
        )

        db.add(audit_entry)
        db.add(rl_episode)

        try:
            await db.commit()
            log.debug("audit.persisted", request_id=request.request_id)
        except Exception as exc:
            await db.rollback()
            log.error("audit.persist_failed", error=str(exc))

    async def _get_cumulative_accuracy(self, db: AsyncSession) -> float:
        """Running average of accuracy_gain across all episodes with true_label."""
        result = await db.execute(
            select(func.avg(RLEpisode.accuracy_gain))
            .where(RLEpisode.accuracy_gain.isnot(None))
        )
        val = result.scalar()
        return round(float(val), 5) if val else 0.0


# Module-level singleton
_orchestrator: ProxyOrchestrator | None = None


def get_orchestrator() -> ProxyOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ProxyOrchestrator()
    return _orchestrator
