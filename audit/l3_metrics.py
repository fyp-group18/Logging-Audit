"""Layer 3 computed metrics for operator feedback quality audit.

Covers:
  - Schema coverage: fraction of corrections expressible without free-text
  - Feedback entropy: Shannon entropy across correction_type distribution
"""

import logging
import math

from sqlalchemy import text

from core.database import SessionLocal, with_db_retry

logger = logging.getLogger(__name__)

# correction_type values that map to a structured schema field.
# Free-text-only feedback (correction_type IS NULL or 'OTHER') is unstructured.
_STRUCTURED_CORRECTION_TYPES = {
    "WRONG_VALUE",
    "WRONG_SEQUENCE",
    "MISSING_STEP",
    "WRONG_TOOL",
    "SAFETY_VIOLATION",
    "OUTDATED_PROCEDURE",
}


@with_db_retry
def compute_schema_coverage() -> dict:
    """Fraction of step-level corrections expressible without free-text.

    A correction is considered schema-expressible when its correction_type maps
    to one of the known structured categories. Null or 'OTHER' correction_type
    values require free-text and are counted as unstructured.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    correction_type,
                    COUNT(*) AS cnt
                FROM step_feedback
                WHERE action IN ('MODIFY', 'INCORRECT')
                  AND retracted = FALSE
                GROUP BY correction_type
                """
            )
        ).fetchall()

    total = 0
    structured = 0
    distribution: dict[str, int] = {}

    for correction_type, cnt in rows:
        key = correction_type if correction_type else "NULL"
        distribution[key] = cnt
        total += cnt
        if correction_type in _STRUCTURED_CORRECTION_TYPES:
            structured += cnt

    coverage = structured / total if total > 0 else 0.0

    return {
        "metric_name": "schema_coverage",
        "value": round(coverage, 4),
        "metadata": {
            "total_corrections": total,
            "structured_count": structured,
            "unstructured_count": total - structured,
            "correction_type_distribution": distribution,
        },
    }


@with_db_retry
def compute_feedback_entropy() -> dict:
    """Shannon entropy (bits) of the correction_type distribution in step_feedback.

    Higher entropy means corrections are spread evenly across types (diverse
    failure modes). Low entropy indicates concentration on a single type, which
    may warrant targeted model improvement.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    COALESCE(correction_type, 'NULL') AS correction_type,
                    COUNT(*) AS cnt
                FROM step_feedback
                WHERE retracted = FALSE
                GROUP BY correction_type
                """
            )
        ).fetchall()

    counts = {r[0]: r[1] for r in rows}
    total = sum(counts.values())

    if total == 0:
        return {
            "metric_name": "feedback_entropy",
            "value": 0.0,
            "metadata": {"total_records": 0, "distribution": {}},
        }

    entropy = 0.0
    proportions: dict[str, float] = {}
    for label, cnt in counts.items():
        p = cnt / total
        proportions[label] = round(p, 4)
        if p > 0:
            entropy -= p * math.log2(p)

    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "metric_name": "feedback_entropy",
        "value": round(entropy, 4),
        "metadata": {
            "total_records": total,
            "num_categories": len(counts),
            "entropy_bits": round(entropy, 4),
            "max_entropy_bits": round(max_entropy, 4),
            "normalized_entropy": round(normalized, 4),
            "proportions": proportions,
        },
    }
