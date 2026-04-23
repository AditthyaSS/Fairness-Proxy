"""
app/api/routes/health.py
=========================
Health & Readiness Checks
"""

from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    from app.core.config import settings
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.APP_VERSION,
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """Used by Kubernetes readiness probes."""
    return {"ready": True}
