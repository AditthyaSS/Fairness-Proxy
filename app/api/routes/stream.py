"""
app/api/routes/stream.py
=========================
WebSocket endpoint that streams pipeline stage progress
as each stage completes, so the React Flow UI updates in real time
rather than waiting for the full response.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.proxy_orchestrator import get_orchestrator
from app.models.schemas import ProxyRequest
from app.db.session import get_db_context
import json

router = APIRouter()


@router.websocket("/ws/pipeline")
async def pipeline_websocket(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)

            async def emit(stage: str, payload: dict):
                await websocket.send_text(json.dumps({
                    "stage": stage,
                    "data": payload
                }))

            # Stage 1: classify
            await emit("classifying", {"status": "running"})
            from app.services.schema_classifier import get_schema_classifier

            # Extract true_label from frontend before building ProxyRequest
            true_label = request_data.pop("true_label", None)
            # Remove non-model fields but save mock_upstream for later use
            mock_upstream = request_data.pop("mock_upstream", False)
            req = ProxyRequest(**request_data, true_label=true_label)
            classifier = get_schema_classifier()
            schema = await classifier.classify(req.payload, domain_hint=req.domain)
            await emit("classified", {
                "domain": schema.domain.value,
                "confidence": schema.domain_confidence,
                "merit_features": list(schema.merit_features.keys()),
                "protected_features": list(schema.protected_features.keys()),
            })

            # Stage 2: twins
            await emit("generating_twins", {"status": "running"})
            from app.services.twin_generator import get_twin_generator
            gen = get_twin_generator()
            twin_result = gen.generate(req.payload, schema.merit_features, schema.protected_features)
            await emit("twins_generated", {
                "num_twins": len(twin_result.twins),
                "swapped_features": [
                    {k: str(v) for k, v in t.swapped_features.items()}
                    for t in twin_result.twins
                ],
            })

            # Stage 3: upstream (stream each inference as it completes)
            await emit("upstream_running", {"status": "running", "total": 6})
            from app.services.upstream_client import get_upstream_client
            upstream = get_upstream_client(mock=mock_upstream)
            all_payloads = [twin_result.original] + [t.payload for t in twin_result.twins]

            for i, payload in enumerate(all_payloads):
                inf = await upstream.infer(req.target_endpoint, payload)
                await emit("upstream_score", {
                    "index": i,
                    "label": "orig" if i == 0 else f"T{i}",
                    "score": round(inf.decision_score, 4),
                    "decision": inf.binary_decision,
                })

            # Stage 4: full pipeline for final result
            await emit("evaluating", {"status": "running"})
            async with get_db_context() as db:
                orchestrator = get_orchestrator()
                result = await orchestrator.process(
                    req, db, mock_upstream=mock_upstream
                )

            # Send complete result
            result_json = json.loads(result.model_dump_json())
            await emit("complete", result_json)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({
                "stage": "error",
                "data": {"message": str(e)}
            }))
        except Exception:
            pass
