"""Process-local cache for the dual-loop RAGAS threshold config.

The InlineEvaluator reads thresholds on every diagnostic session, so a 5-min
in-memory cache cuts the per-request DB hit. Cloud Run min-instances=0 means
cold starts naturally invalidate; admins can also force-invalidate after a
threshold update via `invalidate()`.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from core.crud import get_evaluation_config

_TTL_SECONDS = 300

_lock = threading.Lock()
_state: dict = {"value": None, "expires_at": 0.0}


def get_thresholds() -> dict:
    """Return `{"green": {...}, "red": {...}}` — fetches once per TTL."""
    now = time.monotonic()
    with _lock:
        if _state["value"] is None or now >= _state["expires_at"]:
            cfg = get_evaluation_config()
            _state["value"] = cfg["thresholds"]
            _state["expires_at"] = now + _TTL_SECONDS
        return _state["value"]


def invalidate() -> None:
    """Drop the cached value so the next read refetches from the DB."""
    with _lock:
        _state["expires_at"] = 0.0
        _state["value"] = None


def peek() -> Optional[dict]:
    """Return the currently-cached thresholds (or None) without refetching.

    Used by tests and by the recommendations engine when it wants the
    last-known good config without forcing a DB roundtrip."""
    with _lock:
        return _state["value"]
