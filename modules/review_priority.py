"""Auto-priority computation for async review tasks.

Priority is set at task creation, never manually by admin. Inputs come from
the trigger that fired the task: safety risk severity, RAGAS metric scores,
or the kind of feedback the expert submitted.

The mapping is intentionally conservative — bumps are easier than demotions
because experts can always snooze a P0/P1 task if the queue is overwhelmed.
"""

from __future__ import annotations

from typing import Iterable, Optional

from core.models import ReviewTaskPriority, ReviewTaskSource

# Risk categories that always escalate to P0 regardless of other signals.
# Mirrors the SAFETY_CRITICAL severity bucket used by AntiExample / SafetyExtractor.
SAFETY_CRITICAL_CATEGORIES = {
    "LOTO",
    "HIGH_VOLTAGE",
    "CONFINED_SPACE",
    "PRESSURE_SYSTEM",
    "ARC_FLASH",
}


def _any_metric_below(metrics: dict, threshold: float) -> int:
    """Count how many of {faithfulness, answer_relevance, context_relevance,
    completeness} are present and below ``threshold``."""
    keys = ("faithfulness", "answer_relevance", "context_relevance", "completeness")
    return sum(1 for k in keys if metrics.get(k) is not None and metrics[k] < threshold)


def compute_priority(
    source: ReviewTaskSource,
    *,
    risk_categories: Optional[Iterable[str]] = None,
    metrics: Optional[dict] = None,
    has_correction: bool = False,
    badge: Optional[str] = None,
    quality_badge: Optional[str] = None,
    safety_badge: Optional[str] = None,
) -> ReviewTaskPriority:
    """Return the auto-priority for a new review task.

    Args:
        source: which trigger fired the task.
        risk_categories: list of risk codes from SafetyExtractor (AUTO_SAFETY).
        metrics: dict of RAGAS scores at the time of flag (any source).
            Keys: faithfulness, answer_relevance, context_relevance, completeness.
            Lower = worse.
        has_correction: True if expert submitted a thumbs-down with correction_text
            (treats it as actionable, not a vague complaint).
        badge: composite badge (backward compat).
        quality_badge: quality-only badge from faithfulness + answer_relevance.
        safety_badge: safety-only badge from SafetyExtractor extraction.

    Returns:
        One of P0/P1/P2/P3.
    """
    metrics = metrics or {}
    risks = set(risk_categories or [])
    # Use quality_badge if provided, else fall back to composite badge
    effective_quality_badge = quality_badge or badge

    # ── P0: safety-critical, immediate ─────────────────────────────────
    if risks & SAFETY_CRITICAL_CATEGORIES:
        return ReviewTaskPriority.P0
    if metrics.get("faithfulness") is not None and metrics["faithfulness"] < 0.3:
        return ReviewTaskPriority.P0
    if source == ReviewTaskSource.AUTO_EVAL and effective_quality_badge == "red":
        # red-after-retry indicates the model failed twice — treat as P0
        return ReviewTaskPriority.P0

    # ── P1: high — multiple metric failures, non-critical risk, safety badge red,
    #         or actionable correction ───
    if safety_badge == "red":
        return ReviewTaskPriority.P1
    failing_count = _any_metric_below(metrics, threshold=0.6)
    if failing_count >= 2:
        return ReviewTaskPriority.P1
    if risks:  # any non-critical risk category
        return ReviewTaskPriority.P1
    if has_correction and source == ReviewTaskSource.MANUAL_FLAG:
        return ReviewTaskPriority.P1

    # ── P2: medium — single metric failure or junior manual flag on red badge ──
    if failing_count == 1:
        return ReviewTaskPriority.P2
    if source == ReviewTaskSource.MANUAL_FLAG and effective_quality_badge in (
        "red",
        "gray",
    ):
        return ReviewTaskPriority.P2

    # ── P3: best-effort ────────────────────────────────────────────────
    return ReviewTaskPriority.P3


def priority_sla_hours(priority: ReviewTaskPriority) -> int:
    """SLA window in hours for a given priority. Used by the reminder job."""
    return {
        ReviewTaskPriority.P0: 4,
        ReviewTaskPriority.P1: 24,
        ReviewTaskPriority.P2: 72,
        ReviewTaskPriority.P3: 168,
    }[priority]
