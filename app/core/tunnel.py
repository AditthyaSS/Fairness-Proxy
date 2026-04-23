"""
app/core/tunnel.py
===================
Ngrok tunnel manager.
Opens a public HTTPS tunnel to localhost:8000 on startup.
Prints the public URL to console and stores it for API access.
"""

from __future__ import annotations
import structlog
from app.core.config import get_settings

log = structlog.get_logger(__name__)

_tunnel_url: str | None = None


def start_tunnel(port: int = 8000) -> str | None:
    """
    Opens ngrok tunnel. Returns public HTTPS URL or None if disabled.
    Call this inside FastAPI lifespan on startup.
    """
    settings = get_settings()

    if not settings.NGROK_ENABLED or not settings.NGROK_AUTHTOKEN:
        log.info("tunnel.disabled", hint="set NGROK_AUTHTOKEN and NGROK_ENABLED=true in .env to enable")
        return None

    try:
        from pyngrok import ngrok, conf

        conf.get_default().auth_token = settings.NGROK_AUTHTOKEN

        # Use static domain if configured, else random
        if settings.NGROK_DOMAIN:
            tunnel = ngrok.connect(
                port,
                domain=settings.NGROK_DOMAIN,
            )
        else:
            tunnel = ngrok.connect(port)

        global _tunnel_url
        _tunnel_url = tunnel.public_url

        # Force HTTPS
        if _tunnel_url.startswith("http://"):
            _tunnel_url = _tunnel_url.replace("http://", "https://", 1)

        log.info(
            "tunnel.open",
            public_url=_tunnel_url,
            docs=f"{_tunnel_url}/docs",
            health=f"{_tunnel_url}/health/",
            websocket=f"{_tunnel_url.replace('https', 'wss')}/ws/pipeline",
        )

        # Print clearly for copy-paste
        print("\n" + "=" * 60)
        print("🚀  FAIRNESS PROXY — LIVE API")
        print("=" * 60)
        print(f"  Public URL  :  {_tunnel_url}")
        print(f"  API Docs    :  {_tunnel_url}/docs")
        print(f"  Health      :  {_tunnel_url}/health/")
        print(f"  WebSocket   :  {_tunnel_url.replace('https', 'wss')}/ws/pipeline")
        print("=" * 60)
        print("  SDK usage:")
        print(f"  from fairness_proxy import FairnessProxy")
        print(f"  fp = FairnessProxy(api_key='fp_...', base_url='{_tunnel_url}')")
        print("=" * 60 + "\n")

        return _tunnel_url

    except Exception as e:
        log.error("tunnel.failed", error=str(e))
        return None


def stop_tunnel() -> None:
    """Closes all ngrok tunnels. Call on shutdown."""
    try:
        from pyngrok import ngrok
        ngrok.kill()
        log.info("tunnel.closed")
    except Exception:
        pass


def get_tunnel_url() -> str | None:
    """Returns current public URL, or None if tunnel not open."""
    return _tunnel_url
