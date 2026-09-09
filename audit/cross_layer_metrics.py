"""Cross-layer metrics that span L1 (corpus), L2 (execution), and L3 (feedback).

Covers:
  - Provenance score: fraction of agent steps traceable to a source chunk
  - Safety provenance completeness: safety protocol -> source chunk -> L1 doc chain
  - L2-to-L1 FK integrity: validate all execution-layer chunk references
  - L3-to-L2 FK integrity: validate step_feedback step_index resolves to plan steps
"""

import logging

from sqlalchemy import text

from core.database import SessionLocal, with_db_retry

logger = logging.getLogger(__name__)


@with_db_retry
def compute_provenance_score(thread_id: str) -> dict:
    """Scalar representing the fraction of agent steps traceable to a source chunk.

    A step is considered traceable when its response_id appears in
    response_chunk_link with at least one chunk_id. Steps are derived from
    distinct (response_id, step_index) pairs in step_feedback; when step_feedback
    is absent, response_chunk_link rows for the thread serve as the step proxy.
    """
    with SessionLocal() as db:
        # Distinct steps attempted within this thread
        step_rows = db.execute(
            text(
                """
                SELECT DISTINCT response_id, step_index
                FROM step_feedback
                WHERE thread_id = :tid
                """
            ),
            {"tid": thread_id},
        ).fetchall()

        if not step_rows:
            # Fall back: count distinct response_ids from response_chunk_link
            # joined to evaluation_metrics for the thread.
            response_rows = db.execute(
                text(
                    """
                    SELECT DISTINCT rcl.response_id
                    FROM response_chunk_link rcl
                    JOIN evaluation_metrics em ON em.response_id = rcl.response_id
                    WHERE em.thread_id = :tid
                    """
                ),
                {"tid": thread_id},
            ).fetchall()
            total_steps = len(response_rows)
            response_ids = {r[0] for r in response_rows}
            traceable = len(response_ids)  # all have at least one chunk link
            fallback_used = True
        else:
            total_steps = len(step_rows)
            response_ids = {r[0] for r in step_rows}
            # Check which response_ids have at least one chunk link
            if response_ids:
                linked = db.execute(
                    text(
                        "SELECT DISTINCT response_id FROM response_chunk_link "
                        "WHERE response_id = ANY(:rids)"
                    ),
                    {"rids": list(response_ids)},
                ).fetchall()
                linked_ids = {r[0] for r in linked}
                # A step is traceable when its response_id has chunk links
                traceable = sum(1 for r in step_rows if r[0] in linked_ids)
            else:
                traceable = 0
            fallback_used = False

    score = traceable / total_steps if total_steps > 0 else 0.0

    return {
        "metric_name": "provenance_score",
        "value": round(score, 4),
        "metadata": {
            "thread_id": thread_id,
            "total_steps": total_steps,
            "traceable_steps": traceable,
            "fallback_mode": fallback_used,
        },
    }


@with_db_retry
def compute_safety_provenance_completeness(thread_id: str) -> dict:
    """Completeness of the safety protocol -> source chunk -> L1 document chain.

    For each safety-tagged chunk that was retrieved during this thread, verifies
    that the chunk resolves back to an L1 document record. Returns the fraction
    of safety-relevant chunk references with a complete provenance chain.
    """
    with SessionLocal() as db:
        # Safety chunks retrieved during this thread (via response_chunk_link)
        rows = db.execute(
            text(
                """
                SELECT
                    rcl.chunk_id,
                    c.has_safety_content,
                    c.document_id,
                    d.id AS doc_exists
                FROM response_chunk_link rcl
                JOIN evaluation_metrics em ON em.response_id = rcl.response_id
                LEFT JOIN document_chunks_multimodal c ON c.id = rcl.chunk_id
                LEFT JOIN documents_multimodal d ON d.id = c.document_id
                WHERE em.thread_id = :tid
                """
            ),
            {"tid": thread_id},
        ).fetchall()

    if not rows:
        return {
            "metric_name": "safety_provenance_completeness",
            "value": None,
            "metadata": {"thread_id": thread_id, "safety_chunks_total": 0},
        }

    safety_rows = [r for r in rows if r[1] is True]  # has_safety_content = TRUE
    total_safety = len(safety_rows)

    if total_safety == 0:
        return {
            "metric_name": "safety_provenance_completeness",
            "value": 1.0,
            "metadata": {
                "thread_id": thread_id,
                "safety_chunks_total": 0,
                "note": "no safety-tagged chunks retrieved in this thread",
            },
        }

    # Chain is complete when: chunk exists AND document exists
    complete = sum(1 for r in safety_rows if r[2] is not None and r[3] is not None)
    completeness = complete / total_safety

    return {
        "metric_name": "safety_provenance_completeness",
        "value": round(completeness, 4),
        "metadata": {
            "thread_id": thread_id,
            "safety_chunks_total": total_safety,
            "complete_chain_count": complete,
            "broken_chain_count": total_safety - complete,
        },
    }


@with_db_retry
def validate_l2_to_l1_fk_integrity() -> dict:
    """Global FK validation: every chunk_id in response_chunk_link must exist in L1.

    response_chunk_link is the canonical L2 record of what the agent retrieved.
    This metric surfaces any dangling references caused by document deletion or
    ingestion errors that bypass the ON DELETE CASCADE constraint.
    """
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total_links,
                    COUNT(*) FILTER (
                        WHERE NOT EXISTS (
                            SELECT 1 FROM document_chunks_multimodal c
                            WHERE c.id = rcl.chunk_id
                        )
                    ) AS broken_links
                FROM response_chunk_link rcl
                """
            )
        ).first()

    total = result[0] if result else 0
    broken = result[1] if result else 0
    integrity_rate = (total - broken) / total if total > 0 else 1.0

    return {
        "metric_name": "l2_to_l1_fk_integrity",
        "value": round(integrity_rate, 4),
        "metadata": {
            "total_links": total,
            "valid_links": total - broken,
            "broken_links": broken,
        },
    }


@with_db_retry
def validate_l3_to_l2_fk_integrity() -> dict:
    """Validate that step_feedback step_index values resolve to valid plan steps.

    For each step_feedback row, checks that the step_index falls within the
    range of step_verdicts in the corresponding evaluation_metrics record
    (joined via response_id). A broken FK indicates the feedback references a
    plan step that does not exist in the evaluation trace.
    """
    with SessionLocal() as db:
        # Count feedback rows that have a matching evaluation record
        result = db.execute(
            text(
                """
                WITH matched AS (
                    SELECT
                        sf.step_index,
                        em.step_verdicts
                    FROM step_feedback sf
                    JOIN evaluation_metrics em
                        ON em.response_id = sf.response_id
                    WHERE sf.retracted = FALSE
                      AND em.step_verdicts IS NOT NULL
                      AND jsonb_typeof(em.step_verdicts) = 'array'
                )
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (
                        WHERE step_index < 0
                           OR step_index >= jsonb_array_length(step_verdicts)
                    ) AS out_of_range
                FROM matched
                """
            )
        ).first()

        # Count feedback rows with no evaluation record at all
        unmatched_row = db.execute(
            text(
                """
                SELECT COUNT(*)
                FROM step_feedback sf
                WHERE sf.retracted = FALSE
                  AND NOT EXISTS (
                      SELECT 1 FROM evaluation_metrics em
                      WHERE em.response_id = sf.response_id
                  )
                """
            )
        ).first()

    matched_total = result[0] if result else 0
    out_of_range = result[1] if result else 0
    unmatched = unmatched_row[0] if unmatched_row else 0
    total = matched_total + unmatched
    broken = out_of_range + unmatched
    integrity_rate = (total - broken) / total if total > 0 else 1.0

    return {
        "metric_name": "l3_to_l2_fk_integrity",
        "value": round(integrity_rate, 4),
        "metadata": {
            "total_feedback_steps": total,
            "valid_references": total - broken,
            "broken_references": broken,
            "out_of_range": out_of_range,
            "no_eval_record": unmatched,
        },
    }


def _main() -> int:
    """CLI entry point: the corpus-wide cross-layer FK validations.

    Provenance Completeness and Safety Provenance are reported per corpus
    rather than per thread, and are computed by
    ``eval.scripts.compute_metrics``; this entry point covers the two
    validations that this module evaluates across the whole database.
    """
    import json
    import os

    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL is not set — export it before running this module.")
        return 1

    print(json.dumps({
        "l2_to_l1_fk_integrity": validate_l2_to_l1_fk_integrity(),
        "l3_to_l2_fk_integrity": validate_l3_to_l2_fk_integrity(),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
