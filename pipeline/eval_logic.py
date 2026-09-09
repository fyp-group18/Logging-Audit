"""Dual-loop RAGAS evaluator — sync metric runners and badge computation.

The InlineEvaluator node calls these in parallel via ThreadPoolExecutor; total
node wall-time is bounded to ~5 s (max of 4 parallel Flash calls). Each runner
returns a typed Pydantic result; on LLM/JSON failure `score` is None and the
badge degrades to gray rather than blocking.
"""

from __future__ import annotations

import json
import logging
import random
import time
from datetime import datetime, timezone
from enum import Enum

from google.genai import types as genai_types

from core.config import MODEL_FLASH, MODEL_PRO, genai_client
from pipeline.telemetry import _extract_genai_tokens, make_record
from pipeline.eval_schemas import (
    AnswerRelevanceSyncResult,
    CompletenessSyncResult,
    ContextRelevanceSyncResult,
    FaithfulnessSyncResult,
    PerStepFaithfulnessResult,
    UnifiedInlineEvalResult,
)
from pipeline.utils import extract_gcs_uris
from prompts.system_prompts import (
    PROMPT_SYNC_ANSWER_RELEVANCE,
    PROMPT_SYNC_COMPLETENESS,
    PROMPT_SYNC_CONTEXT_RELEVANCE,
    PROMPT_SYNC_FAITHFULNESS,
    PROMPT_SYNC_PER_STEP_FAITHFULNESS,
    PROMPT_UNIFIED_INLINE_EVAL,
)

logger = logging.getLogger("eval_logic")

METRIC_KEYS = ("faithfulness", "answer_relevance", "context_relevance", "completeness")


class ArtifactType(str, Enum):
    """Response artifact classification for evaluation dispatch.

    Evaluation strategy — whether to evaluate, what text to evaluate,
    whether the badge gates retry — is determined by artifact type
    rather than inferred from schema key presence/absence.
    """

    REPAIR_PLAN = "repair_plan"  # RepairPlanner with non-empty steps
    INFORMATIONAL = "informational"  # RepairPlanner with root_cause, empty steps
    FOLLOW_UP = "follow_up"  # FollowUpResponder with retrieval context
    NON_RETRIEVAL = "non_retrieval"  # FollowUpResponder without retrieval context


# Cap images sent to the multimodal judge. Each image part means a separate
# GCS fetch + vision pass on Flash, which dominates latency.
#
# Shadow (background): up to 8 — fits comfortably in the 180 s shadow budget.
# Inline (gates the user): just 3 — empirically a single multimodal Flash call
# with 8 images runs ~55 s, busting the 50 s inline-node timeout. 3 keeps
# faithfulness in the ~15-25 s range so the inline tier stays under budget
# even when retried.
MAX_EVAL_IMAGE_PARTS = 8
MAX_INLINE_EVAL_IMAGE_PARTS = 3

# Two-tier evaluation: the inline judge (gates the user) computes only the
# safety-critical pair, so its latency budget can stay tight (~10 s). The full
# 4-metric set is computed by the shadow evaluator post-AGENT_DONE for the
# analytics dashboard. Keep DB columns for all 4 — inline simply leaves the
# other two as None until shadow fills them in via update_evaluation_metric_async.
INLINE_METRIC_KEYS = ("faithfulness", "answer_relevance")

# 429 retry tuning for the inline tier. The shadow tier's full 3-attempt /
# 15-60 s backoff doesn't fit in the 100 s inline node budget, so inline gets
# a single short retry. Empirically a Vertex per-minute bucket clears within
# ~5-10 s, so a 4-8 s sleep absorbs most transient quota blips without busting
# the budget. Skip the retry on multimodal calls — the 90 s SDK timeout
# already eats most of the budget, leaving no room for a second attempt.
INLINE_EVAL_MODEL = "gemini-2.5-flash-lite"

INLINE_RETRY_BACKOFF_BASE_S = 4.0
INLINE_RETRY_BACKOFF_JITTER_S = 4.0
INLINE_MAX_ATTEMPTS = 2  # 1 initial + 1 retry


def _is_429(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "429" in text or "resourceexhausted" in text or "quota" in text


# --- Per-metric runners ---


def _generate(
    prompt: str,
    schema_cls,
    image_parts: list | None = None,
    *,
    metric_label: str = "",
    model: str = MODEL_FLASH,
):
    """Single LLM call with structured-JSON enforcement + telemetry.

    When `image_parts` is non-empty the call becomes multimodal — the parts
    are appended after the text prompt so the judge can ground claims against
    diagrams as well as text. On 429 (Vertex per-minute quota), a text-only
    call is retried once with a short jittered backoff; multimodal calls
    don't retry because the 90 s SDK timeout already consumes most of the
    inline node's 100 s budget.

    Returns ``(parsed_result, telemetry_record)``.
    """
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    contents: list = [prompt]
    if image_parts:
        contents.extend(image_parts)
    is_multimodal = bool(image_parts)
    sdk_timeout_ms = 90_000 if is_multimodal else 60_000
    max_attempts = 1 if is_multimodal else INLINE_MAX_ATTEMPTS

    input_tokens: int | None = None
    output_tokens: int | None = None
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            res = genai_client.models.generate_content(
                model=model,
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
            latency_ms = int((time.perf_counter() - t0) * 1000)
            telemetry = make_record(
                node_name=metric_label or schema_cls.__name__,
                model=model,
                operation_type="llm_call",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
            return schema_cls(**data), telemetry
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1 and _is_429(e):
                wait = INLINE_RETRY_BACKOFF_BASE_S + random.uniform(
                    0.0, INLINE_RETRY_BACKOFF_JITTER_S
                )
                logger.warning(
                    f"{schema_cls.__name__} 429 — backing off {wait:.1f}s before retry"
                )
                time.sleep(wait)
                continue
            break

    logger.warning(f"{schema_cls.__name__} failed: {last_exc}")
    reason = (
        f"429 after retry: {last_exc}"
        if last_exc is not None and _is_429(last_exc)
        else f"error: {last_exc}"
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    telemetry = make_record(
        node_name=metric_label or schema_cls.__name__,
        model=model,
        operation_type="llm_call",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        error=str(last_exc),
    )
    return schema_cls(score=None, reason=reason), telemetry


def eval_faithfulness_sync(
    plan: str, context: str, image_parts: list | None = None
) -> tuple[FaithfulnessSyncResult, dict]:
    return _generate(
        PROMPT_SYNC_FAITHFULNESS.format(plan=plan, context=context),
        FaithfulnessSyncResult,
        image_parts=image_parts,
        metric_label="InlineEvaluator:faithfulness",
        model=MODEL_PRO,
    )


def eval_answer_relevance_sync(
    question: str, plan: str
) -> tuple[AnswerRelevanceSyncResult, dict]:
    return _generate(
        PROMPT_SYNC_ANSWER_RELEVANCE.format(question=question, plan=plan),
        AnswerRelevanceSyncResult,
        metric_label="InlineEvaluator:answer_relevance",
        model=MODEL_PRO,
    )


def eval_context_relevance_sync(
    question: str, context: str, image_parts: list | None = None
) -> tuple[ContextRelevanceSyncResult, dict]:
    return _generate(
        PROMPT_SYNC_CONTEXT_RELEVANCE.format(question=question, context=context),
        ContextRelevanceSyncResult,
        image_parts=image_parts,
        metric_label="InlineEvaluator:context_relevance",
    )


def eval_completeness_sync(
    question: str, context: str, plan: str, image_parts: list | None = None
) -> tuple[CompletenessSyncResult, dict]:
    return _generate(
        PROMPT_SYNC_COMPLETENESS.format(question=question, context=context, plan=plan),
        CompletenessSyncResult,
        image_parts=image_parts,
        metric_label="InlineEvaluator:completeness",
    )


# --- Per-step faithfulness (Group 2) ---


def evaluate_per_step_faithfulness(
    steps: list[dict],
    retrieved_chunks: list[dict],
    image_parts: list | None = None,
) -> tuple[PerStepFaithfulnessResult | None, dict | None]:
    """Single gemini-2.5-pro call returning per-step faithfulness verdicts.

    Returns ``(result, telemetry_record)`` on success.
    Returns ``(None, telemetry_record)`` on timeout/failure (graceful degradation).
    """
    if not steps:
        return None, None

    # Build step text with step_ids for the prompt
    step_lines: list[str] = []
    for s in steps:
        sid = s.get("step_id", "")
        text = s.get("text", "") if isinstance(s, dict) else str(s)
        step_lines.append(f"[{sid}] {text}")

    # Build chunk text with chunk_ids for the prompt
    chunk_lines: list[str] = []
    for c in retrieved_chunks:
        cid = c.get("chunk_id", c.get("id", ""))
        ctext = c.get("text", "")
        chunk_lines.append(f"[chunk_id={cid}] {ctext}")

    prompt = PROMPT_SYNC_PER_STEP_FAITHFULNESS.format(
        steps="\n".join(step_lines),
        context="\n".join(chunk_lines),
    )

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    contents: list = [prompt]
    if image_parts:
        contents.extend(image_parts)

    try:
        res = genai_client.models.generate_content(
            model=MODEL_PRO,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PerStepFaithfulnessResult,
                temperature=0.0,
                http_options=genai_types.HttpOptions(
                    timeout=90_000 if image_parts else 60_000
                ),
            ),
        )
        input_tokens, output_tokens = _extract_genai_tokens(res)
        data = json.loads(res.text)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        telemetry = make_record(
            node_name="InlineEvaluator:per_step_faithfulness",
            model=MODEL_PRO,
            operation_type="llm_call",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
        )
        return PerStepFaithfulnessResult(**data), telemetry
    except Exception as e:
        logger.warning(f"Per-step faithfulness failed: {e}")
        latency_ms = int((time.perf_counter() - t0) * 1000)
        telemetry = make_record(
            node_name="InlineEvaluator:per_step_faithfulness",
            model=MODEL_PRO,
            operation_type="llm_call",
            input_tokens=None,
            output_tokens=None,
            latency_ms=latency_ms,
            started_at=started_at,
            completed_at=datetime.now(timezone.utc),
            error=str(e),
        )
        return None, telemetry


# --- Unified single-call inline evaluation (Flash-Lite) ---


def evaluate_unified(
    plan: str,
    steps: list[dict],
    context: str,
    question: str,
    image_parts: list | None = None,
) -> tuple[UnifiedInlineEvalResult, dict]:
    """Single Flash-Lite call producing per-step faithfulness + answer relevance.

    Replaces the previous 3-call Pro-based inline evaluation. Derives whole-plan
    faithfulness score arithmetically from step verdicts rather than LLM claim
    decomposition.

    Returns ``(result, telemetry_record)``. On failure returns a result with
    empty steps and None score (badge degrades to gray).
    """
    # Build step text with step_ids
    step_lines: list[str] = []
    for s in steps:
        sid = s.get("step_id", "")
        text = s.get("text", "") if isinstance(s, dict) else str(s)
        step_lines.append(f"[{sid}] {text}")

    prompt = PROMPT_UNIFIED_INLINE_EVAL.format(
        question=question,
        steps="\n".join(step_lines) if step_lines else "(No steps to evaluate)",
        context=context,
    )

    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    contents: list = [prompt]
    if image_parts:
        contents.extend(image_parts)
    is_multimodal = bool(image_parts)
    sdk_timeout_ms = 45_000
    max_attempts = 1 if is_multimodal else INLINE_MAX_ATTEMPTS

    input_tokens: int | None = None
    output_tokens: int | None = None
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            res = genai_client.models.generate_content(
                model=INLINE_EVAL_MODEL,
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=UnifiedInlineEvalResult,
                    temperature=0.0,
                    http_options=genai_types.HttpOptions(timeout=sdk_timeout_ms),
                ),
            )
            input_tokens, output_tokens = _extract_genai_tokens(res)
            data = json.loads(res.text)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            telemetry = make_record(
                node_name="InlineEvaluator:unified",
                model=INLINE_EVAL_MODEL,
                operation_type="llm_call",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
            return UnifiedInlineEvalResult(**data), telemetry
        except Exception as e:
            last_exc = e
            if attempt < max_attempts - 1 and _is_429(e):
                wait = INLINE_RETRY_BACKOFF_BASE_S + random.uniform(
                    0.0, INLINE_RETRY_BACKOFF_JITTER_S
                )
                logger.warning(
                    f"UnifiedInlineEval 429 — backing off {wait:.1f}s before retry"
                )
                time.sleep(wait)
                continue
            break

    logger.warning(f"UnifiedInlineEval failed: {last_exc}")
    reason = (
        f"429 after retry: {last_exc}"
        if last_exc is not None and _is_429(last_exc)
        else f"error: {last_exc}"
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)
    telemetry = make_record(
        node_name="InlineEvaluator:unified",
        model=INLINE_EVAL_MODEL,
        operation_type="llm_call",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        started_at=started_at,
        completed_at=datetime.now(timezone.utc),
        error=str(last_exc),
    )
    return UnifiedInlineEvalResult(
        steps=[],
        answer_relevance_score=None,
        answer_relevance_reason=reason,
    ), telemetry


# --- Badge computation ---


def compute_quality_badge(
    scores: dict,
    thresholds: dict,
    keys: tuple = INLINE_METRIC_KEYS,
    step_verdicts: list[dict] | None = None,
) -> str:
    """Quality badge: green / yellow / red / gray over the given key subset.

    - gray: any metric in `keys` is None (LLM/JSON failure or skipped).
    - red:  any metric below its red threshold.
    - yellow: not red, but at least one metric below its green threshold.
    - green: all metrics ≥ green thresholds.

    When `step_verdicts` is provided, per-step unfaithful checks are applied:
    - ≥1 unfaithful step → yellow minimum (upgrades green to yellow).
    - ≥ `max_unfaithful_step_pct` (default 30%) → red.

    The default `keys=INLINE_METRIC_KEYS` matches the inline judge's 2-metric
    tier; pass `keys=METRIC_KEYS` to badge against the full RAGAS set.
    """
    if any(scores.get(k) is None for k in keys):
        return "gray"
    red = thresholds.get("red", {})
    green = thresholds.get("green", {})
    if any(scores[k] < red.get(k, 0.0) for k in keys):
        badge = "red"
    elif any(scores[k] < green.get(k, 0.0) for k in keys):
        badge = "yellow"
    else:
        badge = "green"

    # Per-step unfaithful check (Group 2)
    if step_verdicts:
        total_steps = len(step_verdicts)
        unfaithful_steps = sum(1 for v in step_verdicts if not v.get("faithful", True))
        if total_steps > 0 and unfaithful_steps > 0:
            unfaithful_pct = unfaithful_steps / total_steps
            max_pct = thresholds.get("max_unfaithful_step_pct", 0.3)
            if unfaithful_pct >= max_pct:
                badge = "red"
            elif badge == "green":
                badge = "yellow"

    return badge


# Backward-compatible alias — external callers (diagnostics router, shadow
# evaluator) that don't need the split can keep using the old name.
compute_badge = compute_quality_badge


_BADGE_RANK = _BADGE_ORDER = {"green": 3, "yellow": 2, "red": 1, "gray": 0}


def compute_safety_badge(
    safety_protocols: dict | list | None, thresholds: dict
) -> str | None:
    """Safety badge based on extraction confidence and rule presence.

    Returns None when safety evaluation is not applicable (no protocols, legacy
    format, or no basis chunks). Otherwise returns green/yellow/red based on
    confidence thresholds.
    """
    if safety_protocols is None:
        return None
    if isinstance(safety_protocols, list):
        return None
    if not isinstance(safety_protocols, dict):
        return None

    basis = safety_protocols.get("basis", [])
    if not basis:
        return None

    rules = safety_protocols.get("rules", [])
    confidence = safety_protocols.get("confidence", 0.0)

    # Has context chunks but extracted zero rules → safety concern
    if not rules:
        return "red"

    # Check for malformed rules (missing severity)
    has_malformed = any(
        not r.get("severity") for r in rules if isinstance(r, dict)
    )

    safety_thresholds = thresholds.get("safety", {})
    green_conf = safety_thresholds.get("green", {}).get("confidence", 0.5)
    red_conf = safety_thresholds.get("red", {}).get("confidence", 0.2)

    if confidence >= green_conf and not has_malformed:
        return "green"
    elif confidence < red_conf:
        return "red"
    else:
        return "yellow"


def composite_badge(
    quality_badge: str | None, safety_badge: str | None
) -> str:
    """Composite badge: worst of quality and safety badges.

    - If both are None → gray.
    - If one is None → return the other (or gray if unrecognized).
    - Unrecognized badge values are treated as None (not applicable).
    """
    valid_badges = {"green", "yellow", "red", "gray"}

    q = quality_badge if quality_badge in valid_badges else None
    s = safety_badge if safety_badge in valid_badges else None

    if q is None and s is None:
        return "gray"
    if q is None:
        return s  # type: ignore[return-value]
    if s is None:
        return q

    # Return the worse of the two (lower rank)
    q_rank = _BADGE_RANK.get(q, 0)
    s_rank = _BADGE_RANK.get(s, 0)
    return q if q_rank <= s_rank else s


# --- State extractors ---


def extract_question(state: dict) -> str:
    """First user message in the messages list, falling back to _seed_question."""
    seed = state.get("_seed_question")
    if seed:
        return seed
    for m in reversed(state.get("messages") or []):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    str(item.get("data", ""))
                    for item in content
                    if item.get("type") == "text"
                )
    return ""


def extract_plan_text(final_response: dict) -> str:
    """Concatenate the plan's textual content for evaluation."""
    if not final_response:
        return ""
    steps = final_response.get("repair_steps") or []
    if steps and isinstance(steps[0], dict):
        return "\n".join(s.get("text", "") for s in steps if s)
    if steps:
        return "\n".join(str(s) for s in steps)
    if "answer" in final_response:
        return str(final_response["answer"])
    if "checklist_items" in final_response:
        return "\n".join(str(s) for s in final_response["checklist_items"])
    if "report_summary" in final_response:
        return str(final_response["report_summary"])
    return ""


# Sentinel substrings produced by upstream-node fallbacks (RepairPlanner /
# RootCauseAnalyzer / FollowUpResponder time out → return a single error
# string as the "plan"). Scoring these wastes a full LLM round-trip per
# metric and produces meaningless numbers — short-circuit instead.
_PLAN_FAILURE_MARKERS = (
    "Error: Repair plan generation timed out",
    "Error: Failed to generate repair plan",
    "Error generating response.",
    "Error generating inspection checklist.",
    "Error generating report.",
    "Root cause analysis timed out",
)


def is_plan_failure(plan_text: str) -> bool:
    if not plan_text:
        return True
    head = plan_text.strip()[:200]
    return any(marker in head for marker in _PLAN_FAILURE_MARKERS)


def extract_context(state: dict) -> str:
    manuals = state.get("retrieved_manuals") or []
    safety = state.get("retrieved_safety_docs") or []
    return "\n".join(str(c) for c in manuals + safety if c)


def extract_manual_context(state: dict) -> str:
    """Context for faithfulness — manuals only, no safety docs.

    Faithfulness measures "is the plan grounded in the technical manuals",
    not "does the plan repeat safety warnings".  Excluding safety docs
    reduces the context window by ~30% and avoids false-positive
    faithfulness failures when the plan omits a safety rule that appears
    only in the safety corpus.
    """
    manuals = state.get("retrieved_manuals") or []
    return "\n".join(str(c) for c in manuals if c)


def extract_image_parts(state: dict, max_images: int = MAX_EVAL_IMAGE_PARTS) -> list:
    """Pull up to `max_images` GCS image Parts from the retrieved context.

    Reuses `extract_gcs_uris` so the parsing rules (markdown image regex,
    URL→gs:// rewrite, dedup) match what the planner and judge already use.
    Returns an empty list when no images are present, so callers can pass the
    result through unconditionally without branching.
    """
    text_context = extract_context(state)
    if not text_context:
        return []
    parts = extract_gcs_uris(text_context)
    return parts[:max_images] if len(parts) > max_images else parts


# --- Artifact classification ---


def classify_artifact(final_response: dict | None, has_context: bool) -> ArtifactType:
    """Classify the response to determine evaluation strategy.

    Replaces the binary is-repair-plan check with multi-type dispatch.
    Evaluation mode, evaluable text extraction, and routing decisions
    all key off this classification.
    """
    if not final_response:
        return ArtifactType.NON_RETRIEVAL

    # RepairPlanner with actionable steps → full step-level evaluation
    if final_response.get("repair_steps"):
        return ArtifactType.REPAIR_PLAN

    # RepairPlanner with root_cause but empty/absent steps → informational
    if "root_cause" in final_response:
        return ArtifactType.INFORMATIONAL

    # FollowUpResponder produces "answer"
    if "answer" in final_response:
        return ArtifactType.FOLLOW_UP if has_context else ArtifactType.NON_RETRIEVAL

    # Unknown structure — evaluate if context exists, skip otherwise
    return ArtifactType.FOLLOW_UP if has_context else ArtifactType.NON_RETRIEVAL


def extract_evaluable_text(final_response: dict, artifact_type: ArtifactType) -> str:
    """Extract the text appropriate for evaluation given the artifact type.

    Unlike extract_plan_text (which guesses by key presence), this function
    knows which field to read because the artifact type is already resolved.
    """
    if not final_response:
        return ""

    if artifact_type == ArtifactType.REPAIR_PLAN:
        steps = final_response.get("repair_steps") or []
        if steps and isinstance(steps[0], dict):
            return "\n".join(s.get("text", "") for s in steps if s)
        if steps:
            return "\n".join(str(s) for s in steps)
        return ""

    if artifact_type == ArtifactType.INFORMATIONAL:
        return str(final_response.get("root_cause", ""))

    if artifact_type == ArtifactType.FOLLOW_UP:
        return str(final_response.get("answer", ""))

    return ""
