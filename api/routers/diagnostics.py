# api/routers/diagnostics.py
import datetime as dt
import json
import logging
import time
import uuid

from fastapi.responses import StreamingResponse

from api.schemas import DiagnoseRequest, EscalateRequest
from api.agent_manager import agent_manager

from sqlalchemy.orm import Session
from fastapi import APIRouter, HTTPException, Request, Depends
from core.database import get_db
from core import crud
from core.eval_config import get_thresholds

from pipeline.eval_logic import (
    ArtifactType,
    classify_artifact,
    extract_question,
    extract_evaluable_text,
    extract_context,
    extract_image_parts,
    eval_faithfulness_sync,
    eval_answer_relevance_sync,
    evaluate_per_step_faithfulness,
    compute_quality_badge,
    INLINE_METRIC_KEYS,
    is_plan_failure,
)
from pipeline.custom_ragas import run_shadow_evaluation

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Diagnostics"])


@router.post("/diagnose/stream")
async def start_and_stream_diagnosis(
    request: Request,
    diagnosis: DiagnoseRequest,
    db: Session = Depends(get_db),
):
    thread_id = diagnosis.thread_id or str(uuid.uuid4())

    crud.create_or_update_thread(db, thread_id, device_id=diagnosis.device_id)

    input_data = agent_manager.build_input_data(diagnosis)

    def event_generator():
        yield f"data: {json.dumps({'event': 'THREAD_CREATED', 'thread_id': thread_id})}\n\n"
        yield from agent_manager.stream_graph(thread_id, input_data)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/diagnose/{thread_id}/stream")
async def stream_diagnosis(
    thread_id: str,
):
    return StreamingResponse(
        agent_manager.stream_graph(thread_id), media_type="text/event-stream"
    )


@router.post("/diagnose/escalate", status_code=200)
async def escalate_to_supervisor(
    body: EscalateRequest,
):
    """Stub: enqueues an audit-queue entry once Task 8's audit system lands.
    Records the escalation against the metric row regardless."""
    if body.response_id:
        try:
            crud.record_override(body.response_id, f"escalate: {body.reason or ''}")
        except Exception:
            pass
    return {"status": "queued", "thread_id": body.thread_id}


@router.post("/diagnose/{thread_id}/evaluate")
async def evaluate_thread(
    thread_id: str,
):
    """Run (or return cached) inline + shadow evaluation for a thread.

    If evaluations already exist in the DB, returns them directly.
    Otherwise, reads the LangGraph checkpoint, runs all 4 RAGAS metrics,
    persists them, and returns the same shape the SSE stream would emit.
    """
    # Check if eval already exists
    try:
        existing = crud.get_evaluations_for_thread(thread_id)
    except Exception:
        existing = []

    if existing:
        # Reconstruct inline + shadow from the persisted row (use last row)
        row = existing[-1]
        rid = row.get("response_id", "unknown")
        badge = row.get("badge", "red")
        if badge not in ("green", "yellow", "red"):
            badge = "red"
        inline = {
            "scores": {
                "faithfulness": row.get("sync_faithfulness"),
                "answer_relevance": row.get("sync_answer_relevance"),
            },
            "reasons": row.get("sync_reasons") or {},
            "badge": badge,
            "quality_badge": row.get("quality_badge") or badge,
            "safety_badge": row.get("safety_badge"),
            "duration_ms": row.get("sync_duration_ms"),
            "error": None,
            "response_id": rid,
            "step_verdicts": row.get("step_verdicts"),
        }
        shadow = {
            "context_relevance": {
                "score": row.get("async_context_relevance"),
                "reason": "",
            },
            "completeness": {"score": row.get("async_completeness"), "reason": ""},
            "evidence": row.get("async_evidence"),
            "duration_ms": row.get("async_duration_ms"),
            "response_id": rid,
        }
        return {"inline": inline, "shadow": shadow}

    # No persisted eval — read LangGraph state and run evaluation
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = agent_manager.graph.get_state(config)
        if not snapshot or not snapshot.values:
            raise HTTPException(status_code=404, detail="Thread state not found")
        state = dict(snapshot.values)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to read state for {thread_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to read thread state")

    # Extract inputs
    question = extract_question(state)
    final_response = state.get("final_response") or {}
    context = extract_context(state)
    image_parts = extract_image_parts(state)
    response_id = state.get("response_id") or "unknown"

    artifact_type = classify_artifact(final_response, bool(context))

    if artifact_type == ArtifactType.NON_RETRIEVAL:
        raise HTTPException(
            status_code=422,
            detail="Non-retrieval response — evaluation not applicable",
        )

    plan_text = extract_evaluable_text(final_response, artifact_type)
    if not plan_text or is_plan_failure(plan_text):
        raise HTTPException(status_code=422, detail="No valid content to evaluate")

    # Run inline metrics (faithfulness + answer_relevance)
    start = time.time()
    faith_result, _ = eval_faithfulness_sync(plan_text, context, image_parts)
    relevance_result, _ = eval_answer_relevance_sync(question, plan_text)
    inline_duration_ms = int((time.time() - start) * 1000)

    scores = {
        "faithfulness": faith_result.score,
        "answer_relevance": relevance_result.score,
    }
    reasons = {
        "faithfulness": faith_result.reason,
        "answer_relevance": relevance_result.reason,
    }

    # Per-step faithfulness (Group 2) — only for plan artifacts
    step_verdicts = None
    repair_steps = final_response.get("repair_steps") or []
    if repair_steps:
        eval_steps = []
        for s in repair_steps:
            if isinstance(s, dict):
                eval_steps.append(
                    {"step_id": s.get("step_id", ""), "text": s.get("text", "")}
                )
            else:
                eval_steps.append({"step_id": "", "text": str(s)})
        img_map = state.get("retrieved_image_map") or {}
        ordered_entries = img_map.get("ordered", [])
        retrieved_chunks = [
            {"chunk_id": e.get("id", ""), "text": e.get("text", "")}
            for e in ordered_entries
            if e.get("id") and e.get("text")
        ]
        if eval_steps and retrieved_chunks:
            per_step_result, _ = evaluate_per_step_faithfulness(
                eval_steps, retrieved_chunks, image_parts
            )
            if per_step_result:
                step_verdicts = [v.model_dump() for v in per_step_result.steps]

    thresholds = get_thresholds()
    quality_badge = compute_quality_badge(
        scores, thresholds, INLINE_METRIC_KEYS, step_verdicts=step_verdicts
    )
    safety_badge = None  # Safety badge removed; SafetyExtractor v2 is informational only
    badge = quality_badge

    # Persist inline eval
    try:
        crud.insert_evaluation_metric(
            {
                "thread_id": thread_id,
                "response_id": response_id,
                "device_id": state.get("device_id", ""),
                "user_id": None,
                "badge": badge,
                "quality_badge": quality_badge,
                "safety_badge": safety_badge,
                "sync_faithfulness": scores["faithfulness"],
                "sync_answer_relevance": scores["answer_relevance"],
                "sync_context_relevance": None,
                "sync_completeness": None,
                "sync_reasons": reasons,
                "sync_duration_ms": inline_duration_ms,
                "thresholds_snapshot": thresholds,
                "step_verdicts": step_verdicts,
            }
        )
    except Exception as e:
        logger.error(f"Failed to persist inline eval for {thread_id}: {e}")

    # Run shadow metrics (context_relevance + completeness)
    state["response_id"] = response_id
    shadow_result = run_shadow_evaluation(state)

    # Persist shadow eval
    try:
        crud.update_evaluation_metric_async(
            response_id,
            {
                "async_context_relevance": (
                    shadow_result.get("context_relevance") or {}
                ).get("score"),
                "async_completeness": (shadow_result.get("completeness") or {}).get(
                    "score"
                ),
                "async_evidence": shadow_result.get("evidence"),
                "async_duration_ms": shadow_result.get("duration_ms"),
                "async_completed_at": shadow_result.get("completed_at")
                or dt.datetime.now(dt.timezone.utc),
            },
        )
    except Exception as e:
        logger.error(f"Failed to persist shadow eval for {thread_id}: {e}")

    inline = {
        "scores": scores,
        "reasons": reasons,
        "badge": badge,
        "quality_badge": quality_badge,
        "safety_badge": safety_badge,
        "duration_ms": inline_duration_ms,
        "error": None,
        "response_id": response_id,
        "step_verdicts": step_verdicts,
    }
    shadow = {
        "context_relevance": shadow_result.get(
            "context_relevance", {"score": None, "reason": ""}
        ),
        "completeness": shadow_result.get(
            "completeness", {"score": None, "reason": ""}
        ),
        "evidence": shadow_result.get("evidence"),
        "duration_ms": shadow_result.get("duration_ms"),
        "response_id": response_id,
    }
    return {"inline": inline, "shadow": shadow}



# Audio transcription and PDF report endpoints removed — not needed for eval pipeline.
