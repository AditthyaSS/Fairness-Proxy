"""
Fairness Proxy — Main Application Entry Point
==============================================
Reverse-proxy middleware that intercepts AI decisions,
runs counterfactual fairness analysis, and applies RL-based
mitigation before returning results to the client.
"""

import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.routes import proxy, audit, health, dataset, rl_status, stream, tunnel
from app.core.config import settings
from app.core.tunnel import start_tunnel, stop_tunnel
from app.db.session import engine, Base
from app.core.logging import configure_logging

configure_logging()
log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    log.info("fairness_proxy.startup", version=settings.APP_VERSION)
    # Create all DB tables on startup (use Alembic for production migrations)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Auto-train model if not present
    from pathlib import Path
    if not Path("models/xgboost_upstream.joblib").exists():
        log.info("upstream.model_missing", hint="triggering background training")
        import subprocess
        subprocess.Popen(["python", "scripts/train_upstream_model.py"])
        log.info("upstream.training_started_in_background")

    log.info("database.tables_ready")

    # Open ngrok tunnel if configured
    tunnel_url = start_tunnel(port=8000)
    if tunnel_url:
        app.state.public_url = tunnel_url
    else:
        app.state.public_url = "http://localhost:8000"

    yield

    # ── Shutdown ──
    stop_tunnel()
    log.info("fairness_proxy.shutdown")
    await engine.dispose()


app = FastAPI(
    title="Fairness Proxy — AI Bias Interception Middleware",
    description=(
        "A man-in-the-middle fairness layer that intercepts third-party AI decisions, "
        "runs counterfactual twin analysis, computes SHAP-based bias scores, and applies "
        "a Proximal Policy Optimization (PPO) RL agent to mitigate discriminatory outputs."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Prometheus metrics ────────────────────────────────────────────────────────
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(proxy.router, prefix="/v1/proxy", tags=["Fairness Proxy"])
app.include_router(audit.router, prefix="/v1/audit", tags=["Audit & XAI Reports"])
app.include_router(dataset.router, prefix="/v1/dataset", tags=["Dataset"])
app.include_router(rl_status.router, prefix="/v1/rl", tags=["RL Agent"])
app.include_router(tunnel.router, prefix="/v1", tags=["Tunnel"])
app.include_router(stream.router, tags=["Streaming"])


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )
