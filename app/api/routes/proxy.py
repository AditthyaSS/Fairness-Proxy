"""
app/api/routes/proxy.py
========================
Primary Fairness Proxy Router
-------------------------------
POST /v1/proxy/infer    — Full pipeline: classify → twin → infer → SHAP → RL → respond
POST /v1/proxy/classify — Dry-run: only domain detection + feature classification
POST /v1/proxy/twins    — Dry-run: generate counterfactual twins without calling upstream
"""

from fastapi import APIRouter, Depends, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.schemas import (
    ClassifiedSchema,
    ProxyRequest,
    ProxyResponse,
    TwinGenerationResult,
)
from app.services.proxy_orchestrator import get_orchestrator
from app.services.schema_classifier import get_schema_classifier
from app.services.twin_generator import get_twin_generator

router = APIRouter()


@router.post(
    "/infer",
    response_model=ProxyResponse,
    summary="Full Fairness Pipeline",
    description=(
        "Intercepts a request destined for a third-party AI. Runs the complete "
        "5-stage fairness pipeline: domain classification → counterfactual twin "
        "generation → parallel upstream inference → SHAP XAI extraction → PPO RL "
        "agent decision. Returns the original decision, corrected decision, full "
        "XAI transparency report, and RL verdict."
    ),
)
async def infer(
    request: ProxyRequest,
    db: AsyncSession = Depends(get_db),
    mock: bool = Query(
        default=False,
        description="Use mock upstream AI (intentionally biased). Useful for demo / testing.",
    ),
) -> ProxyResponse:
    orchestrator = get_orchestrator()
    return await orchestrator.process(request, db, mock_upstream=mock)


@router.post(
    "/classify",
    response_model=ClassifiedSchema,
    summary="Domain Classification Only",
    description=(
        "Dry-run endpoint. Detects the domain from the payload and splits "
        "features into merit vs protected. Does NOT call the upstream AI."
    ),
)
async def classify(request: ProxyRequest) -> ClassifiedSchema:
    classifier = get_schema_classifier()
    return await classifier.classify(request.payload, domain_hint=request.domain)


@router.post(
    "/twins",
    response_model=TwinGenerationResult,
    summary="Counterfactual Twin Generation Only",
    description=(
        "Dry-run endpoint. Generates N counterfactual twins by permuting "
        "protected attributes while locking merit features. "
        "Does NOT call the upstream AI or run XAI."
    ),
)
async def generate_twins(request: ProxyRequest) -> TwinGenerationResult:
    classifier = get_schema_classifier()
    schema = await classifier.classify(request.payload, domain_hint=request.domain)

    generator = get_twin_generator()
    return generator.generate(
        original_payload=request.payload,
        merit_features=schema.merit_features,
        protected_features=schema.protected_features,
    )
