"""
app/api/routes/dataset.py
==========================
Endpoints for real-time dataset streaming.
Frontend calls /v1/dataset/next to get the next real UCI Adult row.
Frontend calls /v1/dataset/stats to get dataset metadata.
"""

from fastapi import APIRouter
import json
import random
from pathlib import Path
from typing import Any, Dict, List

router = APIRouter()

_rows: List[Dict[str, Any]] = []
_cursor: int = 0
_shuffled: List[int] = []


def _load_rows():
    global _rows, _shuffled
    path = Path("models/sample_rows.json")
    if path.exists():
        with open(path) as f:
            _rows = json.load(f)
        _shuffled = list(range(len(_rows)))
        random.shuffle(_shuffled)


_load_rows()


@router.get("/next")
def get_next_row(bias_demo: bool = False):
    """
    Returns the next real UCI Adult row as a proxy-ready payload.
    If bias_demo=True, selects a row with a disadvantaged protected profile.
    """
    global _cursor
    if not _rows:
        _load_rows()  # Retry loading
        if not _rows:
            return {"error": "Dataset not loaded — run scripts/train_upstream_model.py"}

    if bias_demo:
        candidates = [
            r for r in _rows
            if str(r.get("race", "")).strip() in ("Black", "Amer-Indian-Eskimo", "Other")
            or str(r.get("sex", "")).strip() == "Female"
        ]
        row = random.choice(candidates) if candidates else random.choice(_rows)
    else:
        idx = _shuffled[_cursor % len(_shuffled)]
        row = _rows[idx]
        _cursor += 1

    payload = {k: v for k, v in row.items() if k != "true_label"}
    return {
        "payload": payload,
        "true_label": float(row.get("true_label", 0)),
        "row_index": _cursor,
        "total_rows": len(_rows),
    }


@router.get("/stats")
def get_dataset_stats():
    path = Path("models/metadata.json")
    if not path.exists():
        return {"error": "Model not trained yet"}
    with open(path) as f:
        return json.load(f)


@router.get("/sample")
def get_sample_rows(n: int = 10, bias_demo: bool = False):
    """Returns n rows for batch display."""
    if not _rows:
        _load_rows()
        if not _rows:
            return []
    if bias_demo:
        candidates = [
            r for r in _rows
            if str(r.get("race", "")).strip() in ("Black", "Amer-Indian-Eskimo", "Other")
            or str(r.get("sex", "")).strip() == "Female"
        ]
        pool = candidates if candidates else _rows
    else:
        pool = _rows
    return random.sample(pool, min(n, len(pool)))
