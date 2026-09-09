"""Telemetry wrappers for diagnostic pipeline LLM calls.

Every LLM invocation in the diagnostic graph should go through one of these
wrappers so that token counts, latency, and error state are captured in a
structured telemetry record.  Records accumulate in DiagnosticState via the
``execution_telemetry`` field and are batch-persisted by ``agent_manager``
after the graph completes.

All wrappers return ``(response_or_none, telemetry_record)`` and never raise.
Callers check ``if response is None`` for error handling.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge-decision collector (thread-safe dict keyed by thread_id)
# ---------------------------------------------------------------------------

_edge_lock = threading.Lock()
_edge_telemetry_store: dict[str, list[dict]] = {}

# Module-level active thread_id — set by agent_manager before graph.stream().
# Safe for single-worker uvicorn (one SSE stream at a time).
_active_thread_id: str = ""


def set_edge_telemetry_thread_id(thread_id: str) -> None:
    """Set the active thread_id for edge telemetry collection.
    Called by agent_manager before starting the graph stream."""
    global _active_thread_id
    _active_thread_id = thread_id
    with _edge_lock:
        _edge_telemetry_store[thread_id] = []


def record_edge_decision(
    edge_name: str, decision: str, metadata: dict | None = None
) -> None:
    """Append an edge-decision telemetry record (called from workflow.py routing fns)."""
    now = datetime.now(timezone.utc)
    thread_id = _active_thread_id
    record = {
        "node_name": f"edge:{edge_name}",
        "operation_type": "edge_decision",
        "model": None,
        "input_tokens": None,
        "output_tokens": None,
        "latency_ms": 0,
        "started_at": now,
        "completed_at": now,
        "error": None,
        "metadata": {"decision": decision, **(metadata or {})},
    }
    with _edge_lock:
        _edge_telemetry_store.setdefault(thread_id, []).append(record)


def collect_edge_telemetry(thread_id: str = "") -> list[dict]:
    """Return accumulated edge records for the given thread_id and reset."""
    with _edge_lock:
        records = _edge_telemetry_store.pop(thread_id, [])
    return records


# ---------------------------------------------------------------------------
# Telemetry record builder
# ---------------------------------------------------------------------------


def make_record(
    node_name: str,
    model: str | None,
    operation_type: str,
    input_tokens: int | None,
    output_tokens: int | None,
    latency_ms: int,
    started_at: datetime,
    completed_at: datetime,
    error: str | None = None,
    metadata: dict | None = None,
) -> dict:
    return {
        "node_name": node_name,
        "model": model,
        "operation_type": operation_type,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "latency_ms": latency_ms,
        "started_at": started_at,
        "completed_at": completed_at,
        "error": error,
        "metadata": metadata,
    }


def _extract_genai_tokens(response: Any) -> tuple[int | None, int | None]:
    """Extract token counts from a google-genai GenerateContentResponse."""
    try:
        usage = response.usage_metadata
        return (
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )
    except (AttributeError, TypeError):
        return None, None


# ---------------------------------------------------------------------------
# Wrapper: raw genai_client.models.generate_content()
# ---------------------------------------------------------------------------


def tracked_generate(
    client: Any,
    model: str,
    contents: Any,
    config: Any,
    node_name: str,
    operation_type: str = "llm_call",
) -> tuple[Any | None, dict]:
    """Wrap ``client.models.generate_content`` with telemetry capture.

    Returns ``(response, telemetry)`` on success, ``(None, telemetry)`` on error.
    Never raises.
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        inp, out = _extract_genai_tokens(response)
        return response, make_record(
            node_name=node_name,
            model=model,
            operation_type=operation_type,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(f"tracked_generate({node_name}) failed: {exc}")
        return None, make_record(
            node_name=node_name,
            model=model,
            operation_type=operation_type,
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Wrapper: generate_with_retry() from core.config
# ---------------------------------------------------------------------------


def tracked_generate_with_retry(
    model: str,
    contents: Any,
    config: Any,
    node_name: str,
    operation_type: str = "llm_call",
) -> tuple[Any | None, dict]:
    """Wrap ``generate_with_retry`` (built-in 429 backoff) with telemetry.

    Returns ``(response, telemetry)`` on success, ``(None, telemetry)`` on error.
    Never raises.
    """
    from core.config import generate_with_retry

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        response = generate_with_retry(model=model, contents=contents, config=config)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        inp, out = _extract_genai_tokens(response)
        return response, make_record(
            node_name=node_name,
            model=model,
            operation_type=operation_type,
            input_tokens=inp,
            output_tokens=out,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(f"tracked_generate_with_retry({node_name}) failed: {exc}")
        return None, make_record(
            node_name=node_name,
            model=model,
            operation_type=operation_type,
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Wrapper: LangChain .invoke()
# ---------------------------------------------------------------------------


def tracked_invoke(
    chain: Any,
    input_value: Any,
    node_name: str,
    model_name: str,
    operation_type: str = "llm_call",
) -> tuple[Any | None, dict]:
    """Wrap a LangChain chain's ``.invoke()`` with telemetry.

    Returns ``(response, telemetry)`` on success, ``(None, telemetry)`` on error.
    Never raises.
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        response = chain.invoke(input_value)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        input_tokens: int | None = None
        output_tokens: int | None = None
        if hasattr(response, "response_metadata"):
            meta = response.response_metadata or {}
            usage = meta.get("usage_metadata") or meta.get("token_usage") or {}
            input_tokens = usage.get("prompt_token_count") or usage.get("input_tokens")
            output_tokens = usage.get("candidates_token_count") or usage.get(
                "output_tokens"
            )
        return response, make_record(
            node_name=node_name,
            model=model_name,
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(f"tracked_invoke({node_name}) failed: {exc}")
        return None, make_record(
            node_name=node_name,
            model=model_name,
            operation_type=operation_type,
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Wrapper: LangChain .stream()
# ---------------------------------------------------------------------------


def tracked_stream(
    llm: Any,
    messages: Any,
    node_name: str,
    model_name: str,
    *,
    writer: Any | None = None,
    writer_payload_fn: Any | None = None,
    extract_fn: Any | None = None,
    operation_type: str = "llm_call",
) -> tuple[str | None, dict]:
    """Wrap a LangChain ``.stream()`` call with telemetry.

    *writer*: SSE stream writer (optional). *writer_payload_fn* builds the
    chunk payload for the writer (defaults to ``{"type": "rca_token", "token": t}``).
    *extract_fn* converts each chunk to a string token.

    Returns ``(accumulated_text, telemetry)`` or ``(None, telemetry)`` on error.
    Never raises.
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    accumulated: list[str] = []

    if extract_fn is None:
        from pipeline.nodes import extract_clean_string

        extract_fn = extract_clean_string

    try:
        for chunk in llm.stream(messages):
            token = extract_fn(chunk)
            if token:
                accumulated.append(token)
                if writer:
                    if writer_payload_fn:
                        writer(writer_payload_fn(token))
                    else:
                        writer({"type": "rca_token", "token": token})
        text_result = "".join(accumulated)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return text_result, make_record(
            node_name=node_name,
            model=model_name,
            operation_type=operation_type,
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(f"tracked_stream({node_name}) failed: {exc}")
        return None, make_record(
            node_name=node_name,
            model=model_name,
            operation_type=operation_type,
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Wrapper: stream_with_writer() from core.config
# ---------------------------------------------------------------------------


def tracked_stream_with_writer(
    model: str,
    contents: Any,
    config: Any,
    node_name: str,
    *,
    writer: Any,
    max_retries: int = 3,
    base_delay: float = 2.0,
    operation_type: str = "llm_call",
) -> tuple[str | None, dict]:
    """Wrap ``stream_with_writer`` from ``core.config`` with telemetry.

    Returns ``(accumulated_text, telemetry)`` or ``(None, telemetry)`` on error.
    Never raises.
    """
    from core.config import stream_with_writer

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        result = stream_with_writer(
            model=model,
            contents=contents,
            config=config,
            writer=writer,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return result, make_record(
            node_name=node_name,
            model=model,
            operation_type=operation_type,
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        logger.warning(f"tracked_stream_with_writer({node_name}) failed: {exc}")
        return None, make_record(
            node_name=node_name,
            model=model,
            operation_type=operation_type,
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Node-level telemetry helper (for pass-through / non-LLM nodes)
# ---------------------------------------------------------------------------


def make_node_telemetry(
    node_name: str,
    started_at: datetime,
    latency_ms: int,
    *,
    error: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a telemetry record for a non-LLM node (pass-through, rule check, etc.)."""
    return make_record(
        node_name=node_name,
        model=None,
        operation_type="node",
        input_tokens=None,
        output_tokens=None,
        latency_ms=latency_ms,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        error=error,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# State fingerprint (Phase 4.2 — Audit State Snapshots)
# ---------------------------------------------------------------------------

_STATE_FINGERPRINT_KEYS = [
    "route",
    "search_queries",
    "retrieved_chunk_ids",
    "root_causes",
    "safety_protocols",
    "inline_eval",
]


def state_fingerprint(state: dict) -> str:
    """Compute a short hash of key state fields for change detection.

    Returns a 16-char hex prefix of SHA-256 over the JSON-serialized subset
    of state keys that matter for audit trail purposes.
    """
    subset = {k: state.get(k) for k in _STATE_FINGERPRINT_KEYS if state.get(k)}
    raw = json.dumps(subset, default=str, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]
