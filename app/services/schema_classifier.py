"""
app/services/schema_classifier.py
===================================
Dynamic Contextual Parameterisation
------------------------------------
Uses a zero-shot NLI model to:
  1. Detect the domain (loan_approval, job_application, etc.)
  2. Split payload keys into merit_features vs protected_features
     based on domain-specific ontologies.

The classification is deliberately domain-aware: e.g., `income` is
a MERIT feature in loan approval but a PROTECTED (proxy for class)
feature in job applications.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import structlog
from transformers import pipeline

from app.core.config import settings
from app.models.schemas import ClassifiedSchema, Domain

log = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Hard-coded domain ontologies
# Each domain maps feature name patterns → "merit" | "protected"
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_ONTOLOGIES: Dict[str, Dict[str, str]] = {
    "loan_approval": {
        r"income|salary|annual_income|gross_income": "merit",
        r"credit_score|fico|credit_rating": "merit",
        r"loan_amount|requested_amount|principal": "merit",
        r"employment_years|years_employed|tenure": "merit",
        r"debt_to_income|dti|existing_debt": "merit",
        r"age|date_of_birth|dob": "protected",
        r"gender|sex": "protected",
        r"race|ethnicity|nationality": "protected",
        r"zip_code|postal_code": "protected",    # redlining proxy
        r"marital_status|married": "protected",
        r"disability|handicap": "protected",
        r"name|applicant_name": "ambiguous",     # PII but not direct merit
    },
    "job_application": {
        r"years_experience|experience_years|experience": "merit",
        r"education|degree|qualification|gpa": "merit",
        r"skill[s]?|certification[s]?|license[s]?": "merit",
        r"portfolio|github|linkedin": "merit",
        r"income|salary|current_salary|expected_salary": "protected",  # proxy for class
        r"age|date_of_birth|dob": "protected",
        r"gender|sex": "protected",
        r"race|ethnicity": "protected",
        r"zip_code|postal_code|address": "protected",
        r"name|applicant_name|candidate_name": "ambiguous",
        r"university|college|school": "ambiguous",  # can be prestige proxy
    },
    "healthcare_triage": {
        r"symptoms|chief_complaint|presenting_complaint": "merit",
        r"vital[s]?|blood_pressure|heart_rate|temperature|spo2": "merit",
        r"medical_history|comorbidity|comorbidities": "merit",
        r"lab_results|test_results|bmi": "merit",
        r"age": "merit",                          # clinically relevant
        r"gender|sex": "ambiguous",               # sometimes clinically relevant
        r"race|ethnicity": "protected",
        r"insurance|coverage|payer": "protected",
        r"income|socioeconomic": "protected",
        r"zip_code|address": "protected",
        r"name|patient_name": "ambiguous",
    },
    "insurance_underwriting": {
        r"driving_record|claims_history|accident[s]?": "merit",
        r"property_value|home_value|asset_value": "merit",
        r"coverage_amount|deductible": "merit",
        r"credit_score": "merit",
        r"age": "protected",
        r"gender|sex": "protected",
        r"race|ethnicity": "protected",
        r"zip_code|postal_code": "protected",
        r"marital_status": "protected",
        r"name": "ambiguous",
    },
    "credit_scoring": {
        r"payment_history|on_time_payments|delinquencies": "merit",
        r"credit_utilization|utilization_ratio": "merit",
        r"credit_age|account_age|length_of_history": "merit",
        r"num_accounts|account_count|total_accounts": "merit",
        r"hard_inquiries|recent_inquiries": "merit",
        r"income|salary": "merit",
        r"age": "protected",
        r"gender|sex": "protected",
        r"race|ethnicity": "protected",
        r"zip_code": "protected",
        r"name": "ambiguous",
    },
    "rental_application": {
        r"income|monthly_income|annual_income": "merit",
        r"credit_score": "merit",
        r"rental_history|eviction_history": "merit",
        r"employment_status|employer": "merit",
        r"references|landlord_reference": "merit",
        r"age": "protected",
        r"gender|sex": "protected",
        r"race|ethnicity|nationality": "protected",
        r"familial_status|children|family_size": "protected",
        r"disability|handicap": "protected",
        r"zip_code": "protected",
        r"name": "ambiguous",
    },
    "adult_income": {
        r"education|education-num|education_num": "merit",
        r"hours-per-week|hours_per_week": "merit",
        r"workclass": "merit",
        r"occupation": "merit",
        r"capital-gain|capital-loss|fnlwgt": "merit",
        r"sex|gender": "protected",
        r"race|ethnicity": "protected",
        r"age": "protected",
        r"native-country|native_country": "protected",
        r"relationship|marital-status|marital_status": "ambiguous",
    },
}

# Fallback universal protected attributes for unknown domains
UNIVERSAL_PROTECTED = re.compile(
    r"^(age|gender|sex|race|ethnicity|nationality|religion|disability|"
    r"marital_status|familial_status|sexual_orientation|gender_identity|"
    r"national_origin|zip_code|postal_code)$",
    re.IGNORECASE,
)


class SchemaClassifier:
    """
    Two-stage classifier:
      Stage 1 → Zero-shot NLI to detect domain.
      Stage 2 → Rule-based ontology lookup to split features.
    """

    def __init__(self) -> None:
        log.info("schema_classifier.loading_model", model=settings.CLASSIFIER_MODEL)
        self._classifier = pipeline(
            "zero-shot-classification",
            model=settings.CLASSIFIER_MODEL,
            device=-1,  # CPU; set to 0 for GPU
        )
        log.info("schema_classifier.model_ready")

    def detect_domain(
        self, payload: Dict[str, Any], domain_hint: Optional[Domain]
    ) -> Tuple[Domain, float]:
        """
        If a domain override is provided, trust it.
        Otherwise run zero-shot classification over the payload's key-value
        pairs serialised as a natural-language hypothesis.
        """
        if domain_hint and domain_hint != Domain.UNKNOWN:
            log.debug("domain.override_used", domain=domain_hint)
            return domain_hint, 1.0

        # Build a short natural-language description of the payload
        feature_list = ", ".join(
            f"{k}={v}" for k, v in list(payload.items())[:12]
        )
        hypothesis = f"This is a decision-making request with features: {feature_list}"

        result = self._classifier(
            hypothesis,
            candidate_labels=[d.value for d in Domain if d != Domain.UNKNOWN],
            multi_label=False,
        )

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])

        try:
            domain = Domain(top_label)
        except ValueError:
            domain = Domain.UNKNOWN

        log.info(
            "domain.detected",
            domain=domain,
            confidence=round(top_score, 4),
        )
        return domain, top_score

    def classify_features(
        self,
        payload: Dict[str, Any],
        domain: Domain,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """
        Split payload into (merit_features, protected_features, ambiguous_features)
        using the domain ontology.
        """
        ontology = DOMAIN_ONTOLOGIES.get(domain.value, {})

        merit: Dict[str, Any] = {}
        protected: Dict[str, Any] = {}
        ambiguous: Dict[str, Any] = {}

        for key, value in payload.items():
            classification = self._classify_key(key, ontology)
            if classification == "merit":
                merit[key] = value
            elif classification == "protected":
                protected[key] = value
            else:
                ambiguous[key] = value

        log.info(
            "features.classified",
            domain=domain,
            merit_count=len(merit),
            protected_count=len(protected),
            ambiguous_count=len(ambiguous),
        )
        return merit, protected, ambiguous

    def _classify_key(self, key: str, ontology: Dict[str, str]) -> str:
        """Match a feature key against the domain ontology patterns."""
        key_lower = key.lower().strip()

        for pattern, label in ontology.items():
            if re.search(pattern, key_lower):
                return label

        # Universal fallback
        if UNIVERSAL_PROTECTED.match(key_lower):
            return "protected"

        return "merit"  # Default: assume legitimate if not explicitly protected

    async def classify(
        self,
        payload: Dict[str, Any],
        domain_hint: Optional[Domain] = None,
    ) -> ClassifiedSchema:
        """Full pipeline: detect domain → split features → return schema."""
        domain, confidence = self.detect_domain(payload, domain_hint)
        merit, protected, ambiguous = self.classify_features(payload, domain)

        return ClassifiedSchema(
            domain=domain,
            domain_confidence=confidence,
            merit_features=merit,
            protected_features=protected,
            ambiguous_features=ambiguous,
        )


# Module-level singleton (loaded once at startup)
_classifier_instance: Optional[SchemaClassifier] = None


def get_schema_classifier() -> SchemaClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = SchemaClassifier()
    return _classifier_instance
