"""
app/api/routes/tunnel.py
==========================
Exposes the current ngrok public URL programmatically.
Frontend calls this on load to auto-configure the API base URL.
"""

from fastapi import APIRouter
from app.core.tunnel import get_tunnel_url

router = APIRouter()


@router.get("/tunnel/info")
def tunnel_info():
    """
    Returns the current public ngrok URL.
    Frontend calls this on load to auto-set the API base URL.
    """
    url = get_tunnel_url()
    return {
        "public_url": url or "http://localhost:8000",
        "tunnel_active": url is not None,
        "docs_url": f"{url}/docs" if url else "http://localhost:8000/docs",
        "ws_url": (
            f"{url.replace('https', 'wss')}/ws/pipeline"
            if url
            else "ws://localhost:8000/ws/pipeline"
        ),
        "sdk_snippet": (
            f"FairnessProxy(api_key='fp_...', base_url='{url}')"
            if url
            else ""
        ),
    }
