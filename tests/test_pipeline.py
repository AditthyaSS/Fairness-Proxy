"""
tests/test_pipeline.py
=======================
Integration tests for the Fairness Proxy pipeline.
Tests each stage independently and the full orchestrated pipeline.

Run with:
    pytest tests/ -v --asyncio-mode=auto
"""

from __future__ import annotations

import pytest
import asyncio
from typing import Dict, Any

# ─── Stage 1: Schema Classifier ──────────────────────────────────────────────

LOAN_PAYLOAD: Dict[str, Any] = {
    "income": 72000,
    "credit_score": 710,
    "loan_amount": 250000,
    "employment_years": 5,
    "age": 34,
    "gender": "female",
    "race": "Hispanic",
}

JOB_PAYLOAD: Dict[str, Any] = {
    "years_experience": 4,
    "education": "Bachelor's",
    "skills": "Python, SQL, ML",
    "current_salary": 65000,
    "age": 28,
    "gender": "male",
    "race": "Black",
}


class TestSchemaClassifier:
    def test_loan_approval_feature_split(self):
        """income, credit_score → merit; age, gender, race → protected"""
        from app.services.schema_classifier import SchemaClassifier
        clf = SchemaClassifier.__new__(SchemaClassifier)
        # Bypass model loading; test rule-based classification only
        merit, protected, ambiguous = clf.classify_features(LOAN_PAYLOAD, domain=None)
        # Simulate domain-aware test
        from app.models.schemas import Domain
        merit, protected, ambiguous = clf.classify_features(LOAN_PAYLOAD, domain=Domain.LOAN_APPROVAL)
        assert "income" in merit
        assert "credit_score" in merit
        assert "age" in protected
        assert "gender" in protected
        assert "race" in protected

    def test_job_application_income_is_protected(self):
        """In job_application, income is a protected attribute (class proxy)"""
        from app.services.schema_classifier import SchemaClassifier
        from app.models.schemas import Domain
        clf = SchemaClassifier.__new__(SchemaClassifier)
        merit, protected, ambiguous = clf.classify_features(JOB_PAYLOAD, domain=Domain.JOB_APPLICATION)
        assert "current_salary" in protected or "income" in protected
        assert "years_experience" in merit

    def test_universal_protected_fallback(self):
        """Unknown domain: SHAP protected attributes fall back to universal list"""
        from app.services.schema_classifier import SchemaClassifier
        from app.models.schemas import Domain
        clf = SchemaClassifier.__new__(SchemaClassifier)
        payload = {"credit_score": 700, "religion": "Islam", "some_merit_field": 42}
        merit, protected, ambiguous = clf.classify_features(payload, domain=Domain.UNKNOWN)
        assert "religion" in protected


# ─── Stage 2: Twin Generator ──────────────────────────────────────────────────

class TestTwinGenerator:
    def test_merit_features_unchanged(self):
        """Merit features must be IDENTICAL across all twins"""
        from app.services.twin_generator import TwinGenerator
        gen = TwinGenerator(num_twins=5, seed=42)
        merit = {"income": 72000, "credit_score": 710}
        protected = {"age": 34, "gender": "female"}
        result = gen.generate(LOAN_PAYLOAD, merit, protected)

        for twin in result.twins:
            assert twin.payload["income"] == merit["income"], \
                f"Twin {twin.twin_id}: income was modified!"
            assert twin.payload["credit_score"] == merit["credit_score"], \
                f"Twin {twin.twin_id}: credit_score was modified!"

    def test_protected_features_differ(self):
        """At least one protected feature must differ in each twin"""
        from app.services.twin_generator import TwinGenerator
        gen = TwinGenerator(num_twins=5, seed=42)
        merit = {"income": 72000, "credit_score": 710}
        protected = {"age": 34, "gender": "female"}
        result = gen.generate(LOAN_PAYLOAD, merit, protected)

        for twin in result.twins:
            assert twin.swapped_features, f"Twin {twin.twin_id} has no swapped features"

    def test_num_twins_correct(self):
        """Exactly NUM_TWINS twins are generated"""
        from app.services.twin_generator import TwinGenerator
        gen = TwinGenerator(num_twins=5, seed=42)
        merit = {"income": 72000}
        protected = {"gender": "female", "race": "Hispanic"}
        result = gen.generate(LOAN_PAYLOAD, merit, protected)
        assert len(result.twins) == 5

    def test_no_protected_features_returns_identity_twins(self):
        """If no protected features, twins are identical to original"""
        from app.services.twin_generator import TwinGenerator
        gen = TwinGenerator(num_twins=3, seed=42)
        result = gen.generate({"income": 50000}, merit_features={"income": 50000}, protected_features={})
        for twin in result.twins:
            assert twin.swapped_features == {}


# ─── Stage 3: Fairness Metrics ───────────────────────────────────────────────

class TestFairnessEvaluator:
    def _make_inference(self, score: float) -> Any:
        from app.models.schemas import UpstreamInference
        return UpstreamInference(
            input_payload={},
            raw_response={"score": score},
            decision_score=score,
            binary_decision=score >= 0.5,
            latency_ms=50.0,
        )

    def _make_xai_report(self, protected_shap: float) -> Any:
        from app.models.schemas import XAIReport, FeatureImportance
        return XAIReport(
            method="KernelSHAP",
            feature_importances=[
                FeatureImportance(
                    feature_name="race",
                    shap_value=-protected_shap,
                    abs_shap_value=protected_shap,
                    is_protected=True,
                ),
                FeatureImportance(
                    feature_name="income",
                    shap_value=0.3,
                    abs_shap_value=0.3,
                    is_protected=False,
                ),
            ],
            total_protected_shap=protected_shap,
            top_protected_feature="race",
            top_protected_shap=protected_shap,
        )

    def test_zero_variance_for_identical_twins(self):
        """L_CF = 0 when all twins produce the same score as original"""
        from app.services.fairness_evaluator import FairnessEvaluator
        evaluator = FairnessEvaluator(alpha=1.0, beta=2.0)
        original = self._make_inference(0.75)
        twins = [self._make_inference(0.75) for _ in range(5)]
        xai = self._make_xai_report(0.01)
        metrics = evaluator.evaluate(original, twins, xai)
        assert metrics.counterfactual_variance == pytest.approx(0.0, abs=1e-10)

    def test_l_cf_formula(self):
        """L_CF = (1/N) * sum((y - y'_i)^2) — manual verification"""
        from app.services.fairness_evaluator import FairnessEvaluator
        evaluator = FairnessEvaluator(alpha=1.0, beta=2.0)
        y_hat = 0.8
        twin_scores = [0.6, 0.7, 0.5]  # deltas: 0.2, 0.1, 0.3
        expected_L_cf = ((0.2**2) + (0.1**2) + (0.3**2)) / 3  # = 0.04667

        original = self._make_inference(y_hat)
        twins = [self._make_inference(s) for s in twin_scores]
        xai = self._make_xai_report(0.0)
        metrics = evaluator.evaluate(original, twins, xai)
        assert metrics.counterfactual_variance == pytest.approx(expected_L_cf, rel=1e-5)

    def test_rl_reward_formula(self):
        """R = -(alpha * L_CF + beta * sum|w_p|) — manual verification"""
        from app.services.fairness_evaluator import FairnessEvaluator
        alpha, beta = 1.0, 2.0
        evaluator = FairnessEvaluator(alpha=alpha, beta=beta)
        original = self._make_inference(0.8)
        twins = [self._make_inference(0.6), self._make_inference(0.7)]
        L_cf = ((0.2**2) + (0.1**2)) / 2  # = 0.025
        shap_p = 0.20
        xai = self._make_xai_report(shap_p)
        metrics = evaluator.evaluate(original, twins, xai)
        expected_R = -(alpha * L_cf + beta * shap_p)
        assert metrics.rl_reward == pytest.approx(expected_R, rel=1e-5)


# ─── Stage 4: RL Agent ───────────────────────────────────────────────────────

class TestRLAgent:
    def _make_metrics(self, L_cf: float, shap_p: float) -> Any:
        from app.models.schemas import FairnessMetrics, XAIReport, FeatureImportance
        xai = XAIReport(
            method="KernelSHAP",
            feature_importances=[
                FeatureImportance(
                    feature_name="race", shap_value=-shap_p,
                    abs_shap_value=shap_p, is_protected=True,
                )
            ],
            total_protected_shap=shap_p,
            top_protected_feature="race",
            top_protected_shap=shap_p,
        )
        return FairnessMetrics(
            original_score=0.7,
            twin_scores=[0.5, 0.4, 0.6, 0.3, 0.5],
            counterfactual_variance=L_cf,
            max_score_delta=0.4,
            mean_score_delta=0.2,
            demographic_parity_ratio=0.5,
            xai_report=xai,
            rl_reward=-(L_cf + 2 * shap_p),
        )

    def test_heuristic_pass(self):
        """Low L_CF + low SHAP → PASS"""
        from app.rl.ppo_agent import FairnessRLAgent
        from app.models.schemas import FairnessVerdict
        agent = FairnessRLAgent.__new__(FairnessRLAgent)
        agent.policy = None
        metrics = self._make_metrics(L_cf=0.01, shap_p=0.02)
        verdict = agent._heuristic_decision(metrics)
        assert verdict == FairnessVerdict.PASS

    def test_heuristic_mitigate(self):
        """Moderate bias → MITIGATE"""
        from app.rl.ppo_agent import FairnessRLAgent
        from app.models.schemas import FairnessVerdict
        agent = FairnessRLAgent.__new__(FairnessRLAgent)
        agent.policy = None
        metrics = self._make_metrics(L_cf=0.06, shap_p=0.10)
        verdict = agent._heuristic_decision(metrics)
        assert verdict == FairnessVerdict.MITIGATE

    def test_heuristic_block(self):
        """Severe bias → BLOCK"""
        from app.rl.ppo_agent import FairnessRLAgent
        from app.models.schemas import FairnessVerdict
        agent = FairnessRLAgent.__new__(FairnessRLAgent)
        agent.policy = None
        metrics = self._make_metrics(L_cf=0.20, shap_p=0.40)
        verdict = agent._heuristic_decision(metrics)
        assert verdict == FairnessVerdict.BLOCK

    def test_mitigate_corrects_score_toward_twin_mean(self):
        """MITIGATE: corrected_score = mean of twin scores (merit-only signal)"""
        from app.rl.ppo_agent import FairnessRLAgent
        from app.models.schemas import FairnessVerdict
        import numpy as np
        agent = FairnessRLAgent.__new__(FairnessRLAgent)
        agent.policy = None
        metrics = self._make_metrics(L_cf=0.06, shap_p=0.10)
        decision = agent._build_rl_decision(
            FairnessVerdict.MITIGATE, metrics, LOAN_PAYLOAD, {"gender": "female"}
        )
        expected_corrected = float(np.mean(metrics.twin_scores))
        assert decision.corrected_score == pytest.approx(expected_corrected, abs=1e-6)


# ─── Mock Upstream Client ────────────────────────────────────────────────────

class TestMockUpstreamClient:
    @pytest.mark.asyncio
    async def test_bias_is_detectable(self):
        """Mock upstream applies measurable bias: female+Black applicant scores lower"""
        from app.services.upstream_client import MockUpstreamClient
        client = MockUpstreamClient()

        qualified_payload = {
            "income": 90000,
            "credit_score": 750,
            "loan_amount": 200000,
            "employment_years": 8,
        }

        # Privileged group
        inf_privileged = await client.infer(
            "/test", {**qualified_payload, "gender": "male", "race": "White", "age": 35}
        )
        # Disadvantaged group (same merit)
        inf_disadvantaged = await client.infer(
            "/test", {**qualified_payload, "gender": "female", "race": "Black", "age": 35}
        )

        score_gap = inf_privileged.decision_score - inf_disadvantaged.decision_score
        assert score_gap > 0.1, (
            f"Expected >0.1 gap; got {score_gap:.4f}. "
            "Mock bias may not be functioning correctly."
        )
