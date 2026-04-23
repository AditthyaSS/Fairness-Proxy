"""
app/models/schemas.py
======================
All request/response Pydantic v2 schemas used across the Fairness Proxy.
These are the strict-typed contracts between every layer of the system.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class Domain(str, Enum):
    LOAN_APPROVAL = "loan_approval"
    JOB_APPLICATION = "job_application"
    HEALTHCARE_TRIAGE = "healthcare_triage"
    INSURANCE_UNDERWRITING = "insurance_underwriting"
    CREDIT_SCORING = "credit_scoring"
    RENTAL_APPLICATION = "rental_application"
    ADULT_INCOME = "adult_income"
    UNKNOWN = "unknown"


class FairnessVerdict(str, Enum):
    PASS = "PASS"          # No significant bias detected
    MITIGATE = "MITIGATE"  # Bias detected; RL agent applied corrective action
    BLOCK = "BLOCK"        # Bias too severe; request blocked entirely


class MitigationAction(str, Enum):
    NONE = "none"
    INPUT_SCRUB = "input_scrub"          # RL removed protected features from upstream call
    CONFIDENCE_ADJUSTMENT = "confidence_adjustment"  # RL scaled down biased confidence
    OUTPUT_OVERRIDE = "output_override"  # RL replaced upstream output with corrected value


# ─────────────────────────────────────────────────────────────────────────────
# Incoming client proxy request
# ─────────────────────────────────────────────────────────────────────────────

class ProxyRequest(BaseModel):
    """
    The payload a client sends to the Fairness Proxy instead of calling
    the third-party AI directly. The `payload` dict is forwarded (after
    scrubbing) to the upstream AI.
    """
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    domain: Optional[Domain] = Field(
        default=None,
        description="Override auto-detected domain. Leave null for auto-detection.",
    )
    payload: Dict[str, Any] = Field(
        ...,
        description="The raw input dictionary that would have been sent to the 3rd-party AI.",
    )
    target_endpoint: str = Field(
        ...,
        description="The upstream AI endpoint path, e.g. '/chat/completions'.",
    )
    true_label: Optional[float] = Field(
        default=None,
        description="Ground truth label (0.0 or 1.0) from UCI Adult dataset for accuracy tracking.",
    )
    target_method: str = Field(default="POST")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"json_schema_extra": {
        "example": {
            "domain": "loan_approval",
            "target_endpoint": "/v1/decisions/loan",
            "payload": {
                "applicant_name": "Alex Johnson",
                "income": 72000,
                "credit_score": 710,
                "loan_amount": 250000,
                "employment_years": 5,
                "age": 34,
                "gender": "female",
                "race": "Hispanic",
                "zip_code": "90210",
            },
        }
    }}


# ─────────────────────────────────────────────────────────────────────────────
# Parameter classification output
# ─────────────────────────────────────────────────────────────────────────────

class ClassifiedSchema(BaseModel):
    """Output of the Dynamic Schema Classifier."""
    domain: Domain
    domain_confidence: float = Field(ge=0.0, le=1.0)
    merit_features: Dict[str, Any] = Field(
        description="Judgemental/merit parameters that are legitimate decision inputs."
    )
    protected_features: Dict[str, Any] = Field(
        description="Protected attributes that must NOT drive the decision."
    )
    ambiguous_features: Dict[str, Any] = Field(
        default_factory=dict,
        description="Features whose classification is domain-dependent or unclear.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Counterfactual twins
# ─────────────────────────────────────────────────────────────────────────────

class CounterfactualTwin(BaseModel):
    twin_id: int
    payload: Dict[str, Any]
    swapped_features: Dict[str, Any] = Field(
        description="The protected features that were permuted from the original."
    )


class TwinGenerationResult(BaseModel):
    original: Dict[str, Any]
    twins: List[CounterfactualTwin]


# ─────────────────────────────────────────────────────────────────────────────
# Upstream AI inference result
# ─────────────────────────────────────────────────────────────────────────────

class UpstreamInference(BaseModel):
    """Raw result from the third-party AI for one input."""
    input_payload: Dict[str, Any]
    raw_response: Dict[str, Any]
    decision_score: float = Field(
        description="Normalised probability/score in [0,1] extracted from the raw response."
    )
    binary_decision: bool = Field(
        description="Approved/Hired/Triaged = True; Denied/Rejected = False."
    )
    latency_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# SHAP / XAI report
# ─────────────────────────────────────────────────────────────────────────────

class FeatureImportance(BaseModel):
    feature_name: str
    shap_value: float
    abs_shap_value: float
    is_protected: bool


class XAIReport(BaseModel):
    method: str = "KernelSHAP"
    feature_importances: List[FeatureImportance]
    total_protected_shap: float = Field(
        description="Sum of |SHAP| values for all protected features."
    )
    top_protected_feature: Optional[str]
    top_protected_shap: float


# ─────────────────────────────────────────────────────────────────────────────
# Fairness metrics
# ─────────────────────────────────────────────────────────────────────────────

class FairnessMetrics(BaseModel):
    """
    Core fairness statistics computed from the original + twin predictions.
    """
    original_score: float
    twin_scores: List[float]
    counterfactual_variance: float = Field(
        description=(
            "L_CF = (1/N) * sum((y_hat - y_hat'_i)^2). "
            "Measures how much the decision changes when only protected attrs change."
        )
    )
    max_score_delta: float = Field(
        description="Max |y_hat - y_hat'_i| across all twins."
    )
    mean_score_delta: float
    demographic_parity_ratio: Optional[float] = Field(
        default=None,
        description="P(positive|group_a) / P(positive|group_b) across twin groups.",
    )
    xai_report: XAIReport
    rl_reward: float = Field(
        description="Computed RL reward signal: -(alpha*L_CF + beta*sum|w_p|) + gamma*accuracy_gain"
    )
    original_error: Optional[float] = Field(
        default=None, description="|original_score - true_label|"
    )
    corrected_error: Optional[float] = Field(
        default=None, description="|final_score - true_label|"
    )
    accuracy_gain: Optional[float] = Field(
        default=None, description="original_error - corrected_error (positive = proxy improved)"
    )
    bonus_reward: Optional[float] = Field(
        default=None, description="gamma * accuracy_gain added to RL reward"
    )


# ─────────────────────────────────────────────────────────────────────────────
# RL agent decision
# ─────────────────────────────────────────────────────────────────────────────

class RLDecision(BaseModel):
    verdict: FairnessVerdict
    action_taken: MitigationAction
    corrected_score: Optional[float] = Field(
        default=None,
        description="If MITIGATE, the RL-adjusted decision score.",
    )
    corrected_binary_decision: Optional[bool] = None
    scrubbed_payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="If INPUT_SCRUB, the sanitised payload sent to upstream.",
    )
    explanation: str


# ─────────────────────────────────────────────────────────────────────────────
# Final proxy response to client
# ─────────────────────────────────────────────────────────────────────────────

class ProxyResponse(BaseModel):
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    domain: Domain
    domain_confidence: float

    # Original upstream decision (before any mitigation)
    original_decision: bool
    original_score: float

    # Final decision after Fairness Proxy intervention
    final_decision: bool
    final_score: float

    # Fairness analysis
    fairness_metrics: FairnessMetrics
    rl_decision: RLDecision

    # Full transparency report
    classified_schema: ClassifiedSchema
    twin_generation: TwinGenerationResult
    upstream_inferences: List[UpstreamInference]

    # Accuracy tracking
    accuracy_gain: Optional[float] = None
    cumulative_accuracy_improvement: Optional[float] = None

    # Performance
    total_latency_ms: float
    mitigation_applied: bool


# ─────────────────────────────────────────────────────────────────────────────
# Audit log DB schema (used by SQLAlchemy ORM model)
# ─────────────────────────────────────────────────────────────────────────────

class AuditLogEntry(BaseModel):
    request_id: str
    timestamp: datetime
    domain: str
    verdict: str
    action: str
    original_score: float
    final_score: float
    counterfactual_variance: float
    total_protected_shap: float
    rl_reward: float
    payload_hash: str  # SHA-256 of original payload (PII-safe)
    xai_report_json: str
    metrics_json: str
