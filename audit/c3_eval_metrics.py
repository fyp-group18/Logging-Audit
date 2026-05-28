"""C3 evaluation metrics: safety gate effectiveness and judge badge impact.

Covers:
  - Safety survival rate: protocols that survive into the final approved plan
  - Safety judge gate impact: badge grade improvement attributable to the gate
"""

import logging

from sqlalchemy import text

from core.database import SessionLocal, with_db_retry

logger = logging.getLogger(__name__)

# Badge ordering used to compute gate impact (higher index = better grade).
_BADGE_ORDER = ["red", "yellow", "green"]


def _badge_rank(badge: str | None) -> int:
    """Return ordinal rank for a badge string; unknown badges rank as -1."""
    if badge is None:
        return -1
    return _BADGE_ORDER.index(badge.lower()) if badge.lower() in _BADGE_ORDER else -1


@with_db_retry
def compute_safety_survival_rate(thread_id: str) -> dict:
    """Fraction of safety protocols that survived into the final approved plan.

    Safety protocols are extracted from the gate_payload JSONB of interrupt_gates
    rows for the thread. A protocol is considered to have survived when the
    gate's resolution_payload confirms the protocol was accepted (status APPROVED
    or EDITED rather than REJECTED).
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    status,
                    gate_payload,
                    resolution_payload
                FROM interrupt_gates
                WHERE thread_id = :tid
                  AND gate_type = 'RISK'
                ORDER BY created_at ASC
                """
            ),
            {"tid": thread_id},
        ).fetchall()

    if not rows:
        return {
            "metric_name": "safety_survival_rate",
            "value": None,
            "metadata": {
                "thread_id": thread_id,
                "total_protocols": 0,
                "note": "no RISK gates found for this thread",
            },
        }

    total_protocols = 0
    survived_protocols = 0

    for status, gate_payload, resolution_payload in rows:
        # Count protocols listed in gate_payload (key: safety_protocols or similar)
        protocols = []
        if isinstance(gate_payload, dict):
            protocols = gate_payload.get("safety_protocols") or gate_payload.get(
                "protocols", []
            )
        if not isinstance(protocols, list):
            protocols = []

        gate_count = (
            len(protocols) if protocols else 1
        )  # treat the gate itself as 1 unit
        total_protocols += gate_count

        gate_survived = status in ("APPROVED", "EDITED")
        if gate_survived:
            survived_protocols += gate_count

    survival_rate = survived_protocols / total_protocols if total_protocols > 0 else 0.0

    return {
        "metric_name": "safety_survival_rate",
        "value": round(survival_rate, 4),
        "metadata": {
            "thread_id": thread_id,
            "total_protocols": total_protocols,
            "survived_protocols": survived_protocols,
            "rejected_protocols": total_protocols - survived_protocols,
            "gate_count": len(rows),
        },
    }


@with_db_retry
def compute_safety_judge_gate_impact() -> dict:
    """Mean badge improvement attributable to the safety gate across all sessions.

    Compares the badge on the first evaluation attempt (attempt_number=1) to the
    badge on the retry attempt (attempt_number=2) for the same response_id. A
    positive delta indicates the gate and retry improved quality.

    Returns the mean badge rank delta and the fraction of response_ids where
    the badge improved after retry.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    a1.response_id,
                    a1.badge AS badge_before,
                    a2.badge AS badge_after
                FROM evaluation_metrics a1
                JOIN evaluation_metrics a2
                    ON a2.response_id = a1.response_id
                   AND a2.attempt_number = 2
                WHERE a1.attempt_number = 1
                """
            )
        ).fetchall()

    if not rows:
        return {
            "metric_name": "safety_judge_gate_impact",
            "value": None,
            "metadata": {
                "sample_size": 0,
                "note": "no retry attempts found",
            },
        }

    deltas: list[int] = []
    improved = 0
    unchanged = 0
    degraded = 0

    for response_id, badge_before, badge_after in rows:
        rank_before = _badge_rank(badge_before)
        rank_after = _badge_rank(badge_after)
        delta = rank_after - rank_before
        deltas.append(delta)
        if delta > 0:
            improved += 1
        elif delta == 0:
            unchanged += 1
        else:
            degraded += 1

    n = len(deltas)
    mean_delta = sum(deltas) / n
    improvement_rate = improved / n

    return {
        "metric_name": "safety_judge_gate_impact",
        "value": round(mean_delta, 4),
        "metadata": {
            "sample_size": n,
            "mean_badge_rank_delta": round(mean_delta, 4),
            "improvement_rate": round(improvement_rate, 4),
            "improved_count": improved,
            "unchanged_count": unchanged,
            "degraded_count": degraded,
        },
    }


# --- EV-M01/EV-M02: Gold-Standard Safety Extraction Recall/Precision ---


@with_db_retry
def compute_safety_extraction_recall(
    thread_id: str, gold_protocols: list[str]
) -> dict:
    """Safety extraction recall against a gold-standard annotation set.

    For each gold-standard protocol text, checks whether a matching extracted
    protocol exists in the thread's safety_extraction_metadata. Matching uses
    word overlap (≥50% of gold protocol key words appear in an extracted protocol).

    Args:
        thread_id: The diagnostic thread to evaluate.
        gold_protocols: List of gold-standard safety protocol instruction texts.

    Returns:
        Recall = |matched_gold_protocols| / |gold_protocols|
    """
    if not gold_protocols:
        return {
            "metric_name": "safety_extraction_recall",
            "value": None,
            "metadata": {"thread_id": thread_id, "error": "empty gold_protocols list"},
        }

    # Retrieve extracted protocols from the thread's execution logs
    with SessionLocal() as db:
        row = db.execute(
            text(
                """
                SELECT metadata -> 'safety_protocols'
                FROM agent_execution_logs
                WHERE thread_id = :tid
                  AND node_name = 'SafetyExtractor'
                  AND operation_type IN ('llm', 'node')
                  AND metadata ? 'safety_protocols'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ),
            {"tid": thread_id},
        ).first()

    if not row or not row[0]:
        return {
            "metric_name": "safety_extraction_recall",
            "value": 0.0,
            "metadata": {
                "thread_id": thread_id,
                "gold_total": len(gold_protocols),
                "extracted_total": 0,
                "matched": 0,
            },
        }

    extracted_protocols = row[0] if isinstance(row[0], list) else []
    extracted_texts = [
        p.get("instruction", "") for p in extracted_protocols if isinstance(p, dict)
    ]

    matched = 0
    for gold in gold_protocols:
        gold_words = set(w.lower() for w in gold.split() if len(w) > 3)
        if not gold_words:
            matched += 1  # Trivially short protocols considered matched
            continue
        for ext in extracted_texts:
            ext_words = set(w.lower() for w in ext.split() if len(w) > 3)
            overlap = len(gold_words & ext_words)
            if overlap / len(gold_words) >= 0.5:
                matched += 1
                break

    recall = matched / len(gold_protocols)

    return {
        "metric_name": "safety_extraction_recall",
        "value": round(recall, 4),
        "metadata": {
            "thread_id": thread_id,
            "gold_total": len(gold_protocols),
            "extracted_total": len(extracted_texts),
            "matched": matched,
        },
    }


@with_db_retry
def compute_safety_extraction_precision(
    thread_id: str, gold_protocols: list[str]
) -> dict:
    """Safety extraction precision against a gold-standard annotation set.

    For each extracted protocol, checks whether it matches any gold-standard
    protocol via word overlap. Precision = matched_extracted / total_extracted.

    Args:
        thread_id: The diagnostic thread to evaluate.
        gold_protocols: List of gold-standard safety protocol instruction texts.
    """
    with SessionLocal() as db:
        row = db.execute(
            text(
                """
                SELECT metadata -> 'safety_protocols'
                FROM agent_execution_logs
                WHERE thread_id = :tid
                  AND node_name = 'SafetyExtractor'
                  AND operation_type IN ('llm', 'node')
                  AND metadata ? 'safety_protocols'
                ORDER BY started_at DESC
                LIMIT 1
                """
            ),
            {"tid": thread_id},
        ).first()

    if not row or not row[0]:
        return {
            "metric_name": "safety_extraction_precision",
            "value": None,
            "metadata": {
                "thread_id": thread_id,
                "error": "no extracted protocols found",
            },
        }

    extracted_protocols = row[0] if isinstance(row[0], list) else []
    extracted_texts = [
        p.get("instruction", "") for p in extracted_protocols if isinstance(p, dict)
    ]

    if not extracted_texts:
        return {
            "metric_name": "safety_extraction_precision",
            "value": None,
            "metadata": {"thread_id": thread_id, "extracted_total": 0},
        }

    gold_word_sets = [
        set(w.lower() for w in g.split() if len(w) > 3) for g in gold_protocols
    ]

    matched = 0
    for ext in extracted_texts:
        ext_words = set(w.lower() for w in ext.split() if len(w) > 3)
        if not ext_words:
            continue
        for gold_ws in gold_word_sets:
            if not gold_ws:
                continue
            overlap = len(ext_words & gold_ws)
            if overlap / len(ext_words) >= 0.4:
                matched += 1
                break

    precision = matched / len(extracted_texts)

    return {
        "metric_name": "safety_extraction_precision",
        "value": round(precision, 4),
        "metadata": {
            "thread_id": thread_id,
            "gold_total": len(gold_protocols),
            "extracted_total": len(extracted_texts),
            "matched": matched,
        },
    }


@with_db_retry
def compute_safety_hallucination_rate(thread_id: str) -> dict:
    """Fraction of safety claims in the plan not grounded in retrieved context.

    For each safety-related claim in the repair plan, checks whether its key words
    can be found in the retrieved chunks. Claims with <40% word overlap with any
    chunk are considered hallucinated.
    """
    with SessionLocal() as db:
        # Get the plan text
        plan_row = db.execute(
            text(
                """
                SELECT metadata ->> 'plan_text'
                FROM agent_execution_logs
                WHERE thread_id = :tid
                  AND node_name = 'RepairPlanner'
                  AND operation_type IN ('llm', 'node')
                ORDER BY started_at DESC
                LIMIT 1
                """
            ),
            {"tid": thread_id},
        ).first()

        # Get retrieved chunks
        chunk_rows = db.execute(
            text(
                """
                SELECT c.text
                FROM response_chunk_link rcl
                JOIN evaluation_metrics em ON em.response_id = rcl.response_id
                JOIN document_chunks_multimodal c ON c.id = rcl.chunk_id
                WHERE em.thread_id = :tid
                """
            ),
            {"tid": thread_id},
        ).fetchall()

    # Extract safety-related lines from the plan
    plan_text = plan_row[0] if plan_row and plan_row[0] else ""
    if not plan_text:
        # Fall back: try to get plan from the response itself
        return {
            "metric_name": "safety_hallucination_rate",
            "value": None,
            "metadata": {"thread_id": thread_id, "error": "no plan text found in logs"},
        }

    safety_keywords = {"warning", "caution", "danger", "hazard", "safety", "ppe", "lockout"}
    plan_lines = plan_text.split("\n")
    safety_claims = [
        line for line in plan_lines
        if any(kw in line.lower() for kw in safety_keywords) and len(line.strip()) > 20
    ]

    if not safety_claims:
        return {
            "metric_name": "safety_hallucination_rate",
            "value": 0.0,
            "metadata": {
                "thread_id": thread_id,
                "total_safety_claims": 0,
                "note": "no safety claims found in plan",
            },
        }

    # Build context word set from all retrieved chunks
    context_text = " ".join(r[0] for r in chunk_rows if r[0])
    context_words = set(w.lower() for w in context_text.split() if len(w) > 3)

    hallucinated = 0
    for claim in safety_claims:
        claim_words = set(w.lower() for w in claim.split() if len(w) > 3)
        if not claim_words:
            continue
        overlap = len(claim_words & context_words)
        if overlap / len(claim_words) < 0.4:
            hallucinated += 1

    total = len(safety_claims)
    rate = hallucinated / total if total > 0 else 0.0

    return {
        "metric_name": "safety_hallucination_rate",
        "value": round(rate, 4),
        "metadata": {
            "thread_id": thread_id,
            "total_safety_claims": total,
            "hallucinated_claims": hallucinated,
            "grounded_claims": total - hallucinated,
            "context_chunks_available": len(chunk_rows),
        },
    }
