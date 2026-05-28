"""Shadow evaluation router — Cloud Tasks webhook + polling endpoint."""

import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.config import AUDIT_LOGGING_ENABLED
from core.crud import bulk_insert_execution_logs, update_evaluation_metric_async
from core.eval_config import get_thresholds
from pipeline.custom_ragas import run_shadow_evaluation
from pipeline.eval_logic import compute_quality_badge, _BADGE_ORDER

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/evaluation", tags=["evaluation"])


# --- Schemas ---


class ShadowEvalRequest(BaseModel):
    """Payload dispatched via Cloud Tasks or called directly in local mode."""

    response_id: str
    thread_id: str
    device_id: str
    user_id: str | None = None
    trace_id: str
    question: str
    plan_text: str
    repair_steps: list[dict | str]
    retrieved_context: str
    retrieved_image_uris: list[str]
    inline_step_verdicts: list[dict] | None = None
    inline_scores: dict
    inline_badge: str
    thresholds_snapshot: dict


class ShadowEvalPollResponse(BaseModel):
    completed: bool
    scores: dict | None = None
    step_agreement_rate: float | None = None
    badge_escalated: bool = False
    new_badge: str | None = None


# --- Dispatch function (called from agent_manager) ---


def dispatch_shadow_eval(final_state: dict, trace_id: str) -> None:
    """Dispatch shadow evaluation based on EXECUTION_MODE.

    - cloud: Creates a Cloud Tasks HTTP task targeting the webhook endpoint.
    - local: Runs shadow evaluation in a background thread.
    """
    response_id = final_state.get("response_id", "unknown")
    if response_id == "unknown":
        logger.debug("Shadow eval skipped: no response_id")
        return
    if not AUDIT_LOGGING_ENABLED:
        logger.debug("Shadow eval skipped: audit logging disabled")
        return

    inline_eval = final_state.get("inline_eval") or {}
    execution_mode = os.getenv("EXECUTION_MODE", "cloud").lower()

    # Build the payload for the shadow evaluator
    from pipeline.eval_logic import (
        extract_question,
        extract_manual_context,
        extract_evaluable_text,
        classify_artifact,
        ArtifactType,
    )

    final_response = final_state.get("final_response") or {}
    has_context = bool(final_state.get("retrieved_manuals"))
    artifact_type = classify_artifact(final_response, has_context)
    if artifact_type == ArtifactType.NON_RETRIEVAL:
        logger.debug("Shadow eval skipped: non-retrieval artifact")
        return

    question = extract_question(final_state)
    plan_text = extract_evaluable_text(final_response, artifact_type)
    manual_context = extract_manual_context(final_state)

    # Collect image URIs from retrieved_image_map
    image_map = final_state.get("retrieved_image_map") or {}
    image_uris: list[str] = []
    for entry in image_map.get("ordered", []):
        for img in entry.get("images", []):
            url = img.get("url", "")
            if url:
                image_uris.append(url)

    payload = ShadowEvalRequest(
        response_id=response_id,
        thread_id=final_state.get("_thread_id", ""),
        device_id=final_state.get("device_id", ""),
        user_id=None,
        trace_id=trace_id,
        question=question,
        plan_text=plan_text,
        repair_steps=final_response.get("repair_steps", []),
        retrieved_context=manual_context,
        retrieved_image_uris=image_uris[:8],
        inline_step_verdicts=inline_eval.get("step_verdicts"),
        inline_scores=inline_eval.get("scores", {}),
        inline_badge=inline_eval.get("badge", "gray"),
        thresholds_snapshot=inline_eval.get("thresholds_snapshot") or get_thresholds(),
    )

    if execution_mode == "local":
        _dispatch_local(payload)
    else:
        _dispatch_cloud_tasks(payload)


def _persist_dispatch_record(
    payload: ShadowEvalRequest,
    *,
    dispatch_method: str,
    dispatch_latency_ms: int,
    cloud_tasks_task_id: str | None = None,
    webhook_url: str | None = None,
    payload_size_bytes: int | None = None,
    success: bool = True,
    error_msg: str | None = None,
) -> None:
    """L2-E11: Persist ShadowEvaluator dispatch record to agent_execution_logs."""
    now = datetime.now(timezone.utc)
    try:
        bulk_insert_execution_logs(
            [
                {
                    "trace_id": payload.trace_id,
                    "thread_id": payload.thread_id,
                    "response_id": payload.response_id,
                    "operation_type": "shadow_dispatch",
                    "node_name": "ShadowEvaluator_dispatch",
                    "model": None,
                    "latency_ms": dispatch_latency_ms,
                    "started_at": now,
                    "completed_at": now,
                    "error": error_msg,
                    "metadata": {
                        "dispatch_method": dispatch_method,
                        "dispatch_success": success,
                        "cloud_tasks_task_id": cloud_tasks_task_id,
                        "webhook_url": webhook_url,
                        "payload_size_bytes": payload_size_bytes,
                    },
                }
            ]
        )
    except Exception:
        logger.warning(
            f"Failed to persist shadow dispatch record for {payload.response_id}",
            exc_info=True,
        )


def _dispatch_local(payload: ShadowEvalRequest) -> None:
    """Run shadow evaluation in a background thread (local dev mode)."""

    def _run():
        try:
            _execute_shadow_eval(payload)
        except Exception as e:
            logger.error(f"Local shadow eval failed: {e}")

    thread = threading.Thread(target=_run, name="shadow-eval-local", daemon=True)
    thread.start()
    logger.info(f"Shadow eval dispatched locally for response_id={payload.response_id}")
    _persist_dispatch_record(
        payload,
        dispatch_method="local",
        dispatch_latency_ms=0,
    )


def _dispatch_cloud_tasks(payload: ShadowEvalRequest) -> None:
    """Dispatch shadow eval via the provider-agnostic task queue."""
    try:
        from core.task_queue import get_task_queue_dispatcher

        # Determine the backend service URL for the webhook.
        # BACKEND_SERVICE_URL must be set on all cloud deployments (GCP and Azure).
        backend_url = os.getenv("BACKEND_SERVICE_URL", "")
        if not backend_url:
            # Fallback for GCP only — Azure has no equivalent auto-discovery
            from core.config import CLOUD_PROVIDER as _cp

            if _cp == "gcp":
                project = os.getenv("GOOGLE_CLOUD_PROJECT", "")
                service = os.getenv("K_SERVICE", "hitl-backend")
                backend_url = f"https://{service}-{project}.a.run.app"
            else:
                raise RuntimeError(
                    "BACKEND_SERVICE_URL must be set for shadow eval dispatch "
                    f"on CLOUD_PROVIDER={_cp}"
                )

        payload_bytes = payload.model_dump_json().encode()
        webhook_url = f"{backend_url}/api/v1/evaluation/shadow/run"

        dispatcher = get_task_queue_dispatcher()

        dispatch_t0 = time.perf_counter()
        task_id = dispatcher.dispatch_task(
            payload=payload_bytes,
            target_url=webhook_url,
        )
        dispatch_latency = int((time.perf_counter() - dispatch_t0) * 1000)

        logger.info(
            f"Shadow eval task dispatched: task_id={task_id}, "
            f"response_id={payload.response_id}, "
            f"dispatch_latency_ms={dispatch_latency}, "
            f"payload_size_bytes={len(payload_bytes)}"
        )
        _persist_dispatch_record(
            payload,
            dispatch_method="task_queue",
            dispatch_latency_ms=dispatch_latency,
            cloud_tasks_task_id=task_id,
            webhook_url=webhook_url,
            payload_size_bytes=len(payload_bytes),
        )
    except Exception as e:
        logger.error(f"Task queue dispatch failed: {e}")
        # Fallback to local execution on dispatch failure
        logger.info("Falling back to local shadow eval")
        _dispatch_local(payload)


# --- Webhook endpoint (receives Cloud Tasks dispatch) ---


@router.post("/shadow/run")
async def shadow_eval_webhook(request: ShadowEvalRequest):
    """Execute shadow evaluation. Called by Cloud Tasks or directly in tests."""
    try:
        _execute_shadow_eval(request)
        return {"status": "completed", "response_id": request.response_id}
    except Exception as e:
        logger.error(f"Shadow webhook execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Polling endpoint (frontend queries after AGENT_DONE) ---


@router.get("/shadow/{response_id}", response_model=ShadowEvalPollResponse)
async def poll_shadow_result(response_id: str):
    """Poll for shadow evaluation results. Returns completed=false if pending."""
    from core.crud import get_evaluation_metric_by_response_id

    row = get_evaluation_metric_by_response_id(response_id)
    if not row:
        return ShadowEvalPollResponse(completed=False)

    if row.async_completed_at is None:
        return ShadowEvalPollResponse(completed=False)

    scores = {
        "faithfulness": row.async_faithfulness,
        "answer_relevance": row.async_answer_relevance,
        "context_relevance": row.async_context_relevance,
        "completeness": row.async_completeness,
    }

    # Check badge escalation
    badge_escalated = False
    new_badge = None
    badge_source = getattr(row, "badge_source", "inline")
    if badge_source == "shadow":
        badge_escalated = True
        new_badge = row.badge

    # Extract step agreement rate from evidence
    step_agreement_rate = None
    evidence = row.async_evidence or {}
    cross_val = evidence.get("step_cross_validation")
    if cross_val:
        step_agreement_rate = cross_val.get("agreement_rate")

    return ShadowEvalPollResponse(
        completed=True,
        scores=scores,
        step_agreement_rate=step_agreement_rate,
        badge_escalated=badge_escalated,
        new_badge=new_badge,
    )


# --- Core execution logic ---


def _execute_shadow_eval(payload: ShadowEvalRequest) -> None:
    """Run the full shadow evaluation and persist results."""
    # Build a minimal state dict for run_shadow_evaluation
    state: dict[str, Any] = {
        "response_id": payload.response_id,
        "device_id": payload.device_id,
        "retrieved_manuals": [payload.retrieved_context]
        if payload.retrieved_context
        else [],
        "final_response": {"repair_steps": payload.repair_steps},
        "_seed_question": payload.question,
        "messages": [{"role": "user", "content": payload.question}],
        "retrieved_image_map": {
            "ordered": [
                {"images": [{"url": uri} for uri in payload.retrieved_image_uris]}
            ]
        }
        if payload.retrieved_image_uris
        else {},
    }

    eval_result = run_shadow_evaluation(state)
    if not eval_result:
        return

    response_id = payload.response_id

    # Persist shadow scores to DB
    ctx_meta = eval_result.get("context_relevance") or {}
    comp_meta = eval_result.get("completeness") or {}
    faith_meta = eval_result.get("faithfulness") or {}
    ar_meta = eval_result.get("answer_relevance") or {}
    raw_evidence = eval_result.get("evidence") or {}

    persisted_evidence: dict[str, Any] = {
        "context_relevance": {
            "reason": ctx_meta.get("reason", ""),
            "evidence": raw_evidence.get("context_relevance"),
        },
        "completeness": {
            "reason": comp_meta.get("reason", ""),
            "evidence": raw_evidence.get("completeness"),
        },
    }

    # Step cross-validation
    shadow_step_verdicts = eval_result.get("step_verdicts")
    if shadow_step_verdicts and payload.inline_step_verdicts:
        agreement = _compute_step_agreement(
            payload.inline_step_verdicts, shadow_step_verdicts
        )
        persisted_evidence["step_cross_validation"] = agreement

    async_payload = {
        "async_faithfulness": faith_meta.get("score"),
        "async_answer_relevance": ar_meta.get("score"),
        "async_context_relevance": ctx_meta.get("score"),
        "async_completeness": comp_meta.get("score"),
        "async_evidence": persisted_evidence,
        "async_duration_ms": eval_result.get("duration_ms"),
        "async_completed_at": eval_result.get("completed_at"),
        # L2-EA09: Safety coverage from shadow evaluation
        "async_safety_coverage": eval_result.get("safety_coverage"),
    }

    # Badge escalation check — exclude null-scored metrics (vague query
    # → answer_relevance N/A) so they don't force a gray badge.
    shadow_scores = {
        "faithfulness": faith_meta.get("score"),
        "answer_relevance": ar_meta.get("score"),
        "context_relevance": ctx_meta.get("score"),
        "completeness": comp_meta.get("score"),
    }
    thresholds = payload.thresholds_snapshot
    shadow_badge_keys = tuple(
        k for k in shadow_scores if shadow_scores[k] is not None
    ) or tuple(shadow_scores.keys())
    shadow_badge = compute_quality_badge(
        shadow_scores, thresholds, keys=shadow_badge_keys,
        step_verdicts=shadow_step_verdicts,
    )
    inline_badge = payload.inline_badge

    if _BADGE_ORDER.get(shadow_badge, 3) < _BADGE_ORDER.get(inline_badge, 3):
        # Shadow found worse quality than inline — escalate
        async_payload["badge"] = shadow_badge
        async_payload["badge_source"] = "shadow"
        if shadow_step_verdicts:
            async_payload["shadow_step_verdicts"] = shadow_step_verdicts

        # Create review task for escalation
        try:
            from core.crud import create_review_task
            from core.models import ReviewTaskSource, ReviewTaskPriority

            create_review_task(
                thread_id=payload.thread_id,
                flagged_by=None,
                source=ReviewTaskSource.AUTO_EVAL,
                priority=ReviewTaskPriority.P2,
                response_id=response_id,
                reason="shadow_escalation",
            )
        except Exception as e:
            logger.warning(f"Failed to create shadow escalation review task: {e}")
    else:
        if shadow_step_verdicts:
            async_payload["shadow_step_verdicts"] = shadow_step_verdicts

    # Check step agreement threshold
    cross_val = persisted_evidence.get("step_cross_validation")
    if cross_val and cross_val.get("agreement_rate", 1.0) < 0.8:
        try:
            from core.crud import create_review_task
            from core.models import ReviewTaskSource, ReviewTaskPriority

            create_review_task(
                thread_id=payload.thread_id,
                flagged_by=None,
                source=ReviewTaskSource.AUTO_EVAL,
                priority=ReviewTaskPriority.P1,
                response_id=response_id,
                reason="low_step_agreement",
            )
        except Exception as e:
            logger.warning(f"Failed to create step agreement review task: {e}")

    try:
        update_evaluation_metric_async(response_id, async_payload)
    except Exception as e:
        logger.error(f"Shadow eval DB persist failed: {e}")


def _compute_step_agreement(
    inline_verdicts: list[dict], shadow_verdicts: list[dict]
) -> dict:
    """Compare inline vs shadow step verdicts and compute agreement rate."""
    inline_map = {v.get("step_id"): v.get("faithful", True) for v in inline_verdicts}
    shadow_map = {v.get("step_id"): v.get("faithful", True) for v in shadow_verdicts}

    all_step_ids = set(inline_map.keys()) | set(shadow_map.keys())
    if not all_step_ids:
        return {"agreement_rate": 1.0, "disagreements": []}

    matching = 0
    disagreements = []
    for step_id in sorted(all_step_ids):
        inline_f = inline_map.get(step_id)
        shadow_f = shadow_map.get(step_id)
        if inline_f is None or shadow_f is None:
            continue
        if inline_f == shadow_f:
            matching += 1
        else:
            shadow_reason = next(
                (
                    v.get("reason", "")
                    for v in shadow_verdicts
                    if v.get("step_id") == step_id
                ),
                "",
            )
            disagreements.append(
                {
                    "step_id": step_id,
                    "inline_faithful": inline_f,
                    "shadow_faithful": shadow_f,
                    "shadow_reason": shadow_reason,
                }
            )

    total = matching + len(disagreements)
    agreement_rate = matching / total if total > 0 else 1.0

    return {"agreement_rate": round(agreement_rate, 3), "disagreements": disagreements}
