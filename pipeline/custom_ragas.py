"""Shadow Evaluator — runs full 5-metric RAGAS evaluation asynchronously.

Dispatched via Cloud Tasks (cloud) or BackgroundTask (local) after the inline
evaluator gates the user. Runs: faithfulness, answer_relevance, context_relevance,
completeness, and per-step faithfulness cross-validation — all on gemini-2.5-flash.

Results are written to evaluation_metrics via update_evaluation_metric_async.
Badge escalation and review task creation handled by the evaluation router.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types as genai_types

from core.config import MODEL_FLASH, genai_client
from pipeline.telemetry import _extract_genai_tokens, make_record
from pipeline.eval_logic import (
    ArtifactType,
    classify_artifact,
    extract_context,
    extract_evaluable_text,
    extract_image_parts,
    extract_question,
)
from pipeline.eval_schemas import (
    CompletenessAsyncResult,
    ContextRelevanceAsyncResult,
    FaithfulnessSyncResult,
    AnswerRelevanceSyncResult,
    PerStepFaithfulnessResult,
)
from prompts.system_prompts import (
    PROMPT_ASYNC_COMPLETENESS,
    PROMPT_ASYNC_CONTEXT_RELEVANCE,
    PROMPT_SYNC_FAITHFULNESS,
    PROMPT_SYNC_ANSWER_RELEVANCE,
    PROMPT_SYNC_PER_STEP_FAITHFULNESS,
)

logger = logging.getLogger("ShadowEvaluator")

# Full shadow metric set — all 4 RAGAS metrics + per-step cross-validation.
SHADOW_METRIC_KEYS = ("faithfulness", "answer_relevance", "context_relevance", "completeness")

# Total wall-clock budget for both shadow metrics combined. Sized to fit
# under the agent_manager outer cap (200 s) with headroom.
SHADOW_TOTAL_BUDGET_S = 180.0
# Per-attempt HTTP timeout cap. context_relevance feeds the full retrieved
# context to Flash and can legitimately run 45–80 s, so the cap sits at 90 s.
# A genuinely hung call still returns within the shadow deadline budget.
PER_CALL_TIMEOUT_CAP_S = 90.0
# Exponential backoff base + cap. Wait = min(BASE * 3**attempt, CAP) + jitter.
# attempt 0 → ~15 s, attempt 1 → ~45 s. Jitter (0–3 s) prevents two retries
# from synchronizing into the same Vertex per-minute bucket.
RETRY_BACKOFF_BASE_S = 15.0
RETRY_BACKOFF_CAP_S = 60.0
RETRY_BACKOFF_JITTER_S = 3.0
MAX_ATTEMPTS = 3  # 1 initial + 2 retries


def _is_429(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "resourceexhausted" in text or "quota" in text


def _make_error_result(schema_cls, reason: str):
    """Construct a safe error/empty result for any schema type.

    Handles schemas with different field signatures:
    - Async schemas (score, reason, evidence)
    - Sync metric schemas (score, reason)
    - PerStepFaithfulnessResult (steps only)
    """
    fields = set(schema_cls.model_fields.keys())
    kwargs: dict = {}
    if "score" in fields:
        kwargs["score"] = None
    if "reason" in fields:
        kwargs["reason"] = reason
    if "evidence" in fields:
        kwargs["evidence"] = {}
    if "steps" in fields and "score" not in fields:
        kwargs["steps"] = []
    return schema_cls(**kwargs)


def _generate(
    prompt: str,
    schema_cls,
    deadline_s: float,
    image_parts: list | None = None,
    *,
    metric_label: str = "",
):
    """Flash call with up to 2 retries on 429, bounded by `deadline_s`.

    Returns ``(parsed_result, telemetry_record)``.
    """
    started_at = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()

    contents: list = [prompt]
    if image_parts:
        contents.extend(image_parts)

    input_tokens: int | None = None
    output_tokens: int | None = None

    def _telemetry(error: str | None = None) -> dict:
        return make_record(
            node_name=metric_label or schema_cls.__name__,
            model=MODEL_FLASH,
            operation_type="llm_call",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            started_at=started_at,
            completed_at=dt.datetime.now(dt.timezone.utc),
            error=error,
        )

    for attempt in range(MAX_ATTEMPTS):
        time_left = deadline_s - time.time()
        if time_left < 5:
            logger.warning(
                f"{schema_cls.__name__} deadline reached before attempt {attempt + 1} "
                f"(time_left={time_left:.1f}s)"
            )
            return _make_error_result(schema_cls, "deadline_exceeded"), _telemetry("deadline_exceeded")

        sdk_timeout_ms = int(min(PER_CALL_TIMEOUT_CAP_S, time_left) * 1000)
        try:
            res = genai_client.models.generate_content(
                model=MODEL_FLASH,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_cls,
                    temperature=0.0,
                    http_options=genai_types.HttpOptions(timeout=sdk_timeout_ms),
                ),
            )
            input_tokens, output_tokens = _extract_genai_tokens(res)
            data = json.loads(res.text)
            return schema_cls(**data), _telemetry()
        except Exception as e:
            is_quota = _is_429(e)
            can_retry = attempt < MAX_ATTEMPTS - 1 and is_quota
            if can_retry:
                base_wait = min(
                    RETRY_BACKOFF_BASE_S * (3**attempt), RETRY_BACKOFF_CAP_S
                )
                wait = base_wait + random.uniform(0.0, RETRY_BACKOFF_JITTER_S)
                wait = min(wait, max(0.0, deadline_s - time.time() - 5.0))
                if wait <= 0:
                    logger.warning(
                        f"{schema_cls.__name__} 429 — no time left for retry "
                        f"(deadline in {deadline_s - time.time():.1f}s)"
                    )
                    return _make_error_result(schema_cls, "429 deadline_exceeded"), _telemetry(f"429 deadline_exceeded: {e}")
                logger.warning(
                    f"{schema_cls.__name__} 429 — backing off {wait:.0f}s before "
                    f"retry {attempt + 2}"
                )
                time.sleep(wait)
                continue
            logger.warning(f"{schema_cls.__name__} async failed: {e}")
            reason = "429 after retries" if is_quota else f"error: {e}"
            return _make_error_result(schema_cls, reason), _telemetry(str(e))
    return _make_error_result(schema_cls, "retries exhausted"), _telemetry("retries exhausted")


def _async_context_relevance(
    question: str, context: str, deadline_s: float, image_parts: list | None = None
) -> tuple[ContextRelevanceAsyncResult, dict]:
    return _generate(
        PROMPT_ASYNC_CONTEXT_RELEVANCE.format(question=question, context=context),
        ContextRelevanceAsyncResult,
        deadline_s,
        image_parts=image_parts,
        metric_label="ShadowEvaluator:context_relevance",
    )


def _async_completeness(
    question: str,
    context: str,
    plan: str,
    deadline_s: float,
    image_parts: list | None = None,
) -> tuple[CompletenessAsyncResult, dict]:
    return _generate(
        PROMPT_ASYNC_COMPLETENESS.format(question=question, context=context, plan=plan),
        CompletenessAsyncResult,
        deadline_s,
        image_parts=image_parts,
        metric_label="ShadowEvaluator:completeness",
    )


def _async_faithfulness(
    plan: str, context: str, deadline_s: float, image_parts: list | None = None
) -> tuple[FaithfulnessSyncResult, dict]:
    return _generate(
        PROMPT_SYNC_FAITHFULNESS.format(plan=plan, context=context),
        FaithfulnessSyncResult,
        deadline_s,
        image_parts=image_parts,
        metric_label="ShadowEvaluator:faithfulness",
    )


def _async_answer_relevance(
    question: str, plan: str, deadline_s: float
) -> tuple[AnswerRelevanceSyncResult, dict]:
    return _generate(
        PROMPT_SYNC_ANSWER_RELEVANCE.format(question=question, plan=plan),
        AnswerRelevanceSyncResult,
        deadline_s,
        metric_label="ShadowEvaluator:answer_relevance",
    )


def _async_per_step_faithfulness(
    steps: list[dict], context: str, deadline_s: float
) -> tuple[PerStepFaithfulnessResult | None, dict]:
    """Per-step faithfulness for cross-validation (text-only, no image parts)."""
    if not steps:
        return None, make_record(
            node_name="ShadowEvaluator:per_step_faithfulness",
            model=MODEL_FLASH,
            operation_type="llm_call",
            input_tokens=None, output_tokens=None, latency_ms=0,
            started_at=dt.datetime.now(dt.timezone.utc),
            completed_at=dt.datetime.now(dt.timezone.utc),
            error="no steps",
        )

    step_lines = [f"[{s.get('step_id', '')}] {s.get('text', '')}" for s in steps]
    prompt = PROMPT_SYNC_PER_STEP_FAITHFULNESS.format(
        steps="\n".join(step_lines),
        context=context,
    )
    result, telem = _generate(
        prompt,
        PerStepFaithfulnessResult,
        deadline_s,
        metric_label="ShadowEvaluator:per_step_faithfulness",
    )
    return result, telem


def run_shadow_evaluation(state: dict) -> dict:
    """Run the full 5-metric shadow evaluation on the final state.

    Metrics: faithfulness, answer_relevance, context_relevance, completeness,
    per-step cross-validation. All use gemini-2.5-flash.

    Returns dict with keys per metric ({score, reason}), evidence, step_verdicts,
    duration_ms, response_id, skipped, completed_at, telemetry.
    """
    start = time.time()
    response_id = state.get("response_id") or "unknown"

    question = extract_question(state)
    final_response = state.get("final_response") or {}
    context = extract_context(state)
    image_parts = extract_image_parts(state)

    artifact_type = classify_artifact(final_response, bool(context))

    empty_result = {k: {"score": None, "reason": "not_applicable"} for k in SHADOW_METRIC_KEYS}

    if artifact_type == ArtifactType.NON_RETRIEVAL:
        return {
            **empty_result,
            "evidence": {k: None for k in SHADOW_METRIC_KEYS},
            "step_verdicts": None,
            "duration_ms": int((time.time() - start) * 1000),
            "response_id": response_id,
            "skipped": "not_applicable: non-retrieval response",
            "completed_at": dt.datetime.now(dt.timezone.utc),
            "telemetry": [],
        }

    plan = extract_evaluable_text(final_response, artifact_type)

    if not (question and plan and context):
        skip_result = {k: {"score": None, "reason": "skipped: missing input"} for k in SHADOW_METRIC_KEYS}
        return {
            **skip_result,
            "evidence": {k: None for k in SHADOW_METRIC_KEYS},
            "step_verdicts": None,
            "duration_ms": int((time.time() - start) * 1000),
            "response_id": response_id,
            "skipped": f"missing: question={bool(question)} plan={bool(plan)} context={bool(context)}",
            "completed_at": dt.datetime.now(dt.timezone.utc),
            "telemetry": [],
        }

    deadline = start + SHADOW_TOTAL_BUDGET_S
    out: dict = {k: {"score": None, "reason": "skipped"} for k in SHADOW_METRIC_KEYS}
    evidence: dict = {k: None for k in SHADOW_METRIC_KEYS}
    shadow_telemetry: list[dict] = []
    step_verdicts: list[dict] | None = None

    # Build step data for per-step cross-validation
    eval_steps: list[dict] = []
    raw_steps = final_response.get("repair_steps") or []
    for idx, s in enumerate(raw_steps):
        if isinstance(s, dict):
            eval_steps.append({"step_id": s.get("step_id", f"s-{idx}"), "text": s.get("text", "")})
        else:
            eval_steps.append({"step_id": f"s-{idx}", "text": str(s)})

    # Phase 1: faithfulness (serial, needs context)
    try:
        faith_result, faith_telem = _async_faithfulness(plan, context, deadline, image_parts)
        out["faithfulness"] = {"score": faith_result.score, "reason": faith_result.reason}
        shadow_telemetry.append(faith_telem)
    except Exception as e:
        logger.error(f"Shadow faithfulness crashed: {e}")
        out["faithfulness"] = {"score": None, "reason": f"error: {e}"}

    # Phase 2: answer_relevance (serial, text-only)
    try:
        ar_result, ar_telem = _async_answer_relevance(question, plan, deadline)
        out["answer_relevance"] = {"score": ar_result.score, "reason": ar_result.reason}
        shadow_telemetry.append(ar_telem)
    except Exception as e:
        logger.error(f"Shadow answer_relevance crashed: {e}")
        out["answer_relevance"] = {"score": None, "reason": f"error: {e}"}

    # Phase 3: context_relevance + completeness in parallel
    ex = ThreadPoolExecutor(max_workers=2)
    try:
        future_comp = ex.submit(
            _async_completeness, question, context, plan, deadline, image_parts,
        )
        future_ctx = ex.submit(
            _async_context_relevance, question, context, deadline, image_parts,
        )

        remaining = max(deadline - time.time(), 1.0)
        try:
            for fut in as_completed([future_comp, future_ctx], timeout=remaining):
                try:
                    result, telem = fut.result()
                    if fut is future_comp:
                        out["completeness"] = {"score": result.score, "reason": result.reason}
                        evidence["completeness"] = getattr(result, "evidence", {}) or {}
                    else:
                        out["context_relevance"] = {"score": result.score, "reason": result.reason}
                        evidence["context_relevance"] = getattr(result, "evidence", {}) or {}
                    shadow_telemetry.append(telem)
                except Exception as e:
                    metric = "completeness" if fut is future_comp else "context_relevance"
                    logger.error(f"{metric} call crashed: {e}")
                    out[metric] = {"score": None, "reason": f"error: {e}"}
        except TimeoutError:
            for k in ("context_relevance", "completeness"):
                if out[k]["reason"] == "skipped":
                    out[k] = {"score": None, "reason": "deadline_exceeded"}
    finally:
        ex.shutdown(wait=False)

    # Phase 4: per-step faithfulness cross-validation (text-only)
    if eval_steps:
        try:
            ps_result, ps_telem = _async_per_step_faithfulness(eval_steps, context, deadline)
            shadow_telemetry.append(ps_telem)
            if ps_result and ps_result.steps:
                step_verdicts = [v.model_dump() for v in ps_result.steps]
        except Exception as e:
            logger.error(f"Shadow per-step faithfulness crashed: {e}")

    # Phase 5 (L2-EA09): Safety coverage — fraction of safety protocols
    # that survived in the final plan text.
    safety_coverage: float | None = None
    try:
        safety_protocols = state.get("safety_protocols") or {}
        protocols = []
        if isinstance(safety_protocols, dict):
            protocols = safety_protocols.get("protocols", [])
            if not protocols:
                protocols = safety_protocols.get("rules", [])
        elif isinstance(safety_protocols, list):
            protocols = safety_protocols

        if protocols and plan:
            from pipeline.safety_judge_gate import protocol_survives

            survived = sum(1 for p in protocols if protocol_survives(p, plan))
            safety_coverage = survived / len(protocols)
            evidence["safety_coverage"] = {
                "total_protocols": len(protocols),
                "survived": survived,
                "rate": safety_coverage,
            }
    except Exception as e:
        logger.debug(f"Safety coverage computation failed: {e}")

    duration_ms = int((time.time() - start) * 1000)
    logger.info(
        f"[ShadowEvaluator] response_id={response_id} duration_ms={duration_ms} "
        f"faithfulness={out['faithfulness']['score']} "
        f"answer_relevance={out['answer_relevance']['score']} "
        f"context_relevance={out['context_relevance']['score']} "
        f"completeness={out['completeness']['score']} "
        f"safety_coverage={safety_coverage} "
        f"image_parts={len(image_parts)}"
    )

    return {
        "faithfulness": out["faithfulness"],
        "answer_relevance": out["answer_relevance"],
        "context_relevance": out["context_relevance"],
        "completeness": out["completeness"],
        "safety_coverage": safety_coverage,
        "evidence": evidence,
        "step_verdicts": step_verdicts,
        "duration_ms": duration_ms,
        "response_id": response_id,
        "skipped": None,
        "completed_at": dt.datetime.now(dt.timezone.utc),
        "telemetry": shadow_telemetry,
    }
