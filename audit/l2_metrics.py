"""Layer 2 computed metrics for agent execution and evaluation audit.

Covers:
  - FK integrity of retrieved_chunk_ids in agent_execution_logs
  - Safety event completeness per thread
  - Badge escalation rate across sessions
  - Badge escalation latency (AGENT_DONE to async completion)
  - Calibration drift between sync and async evaluation scores
  - Cohen's kappa for inline vs shadow step verdict agreement
  - Conditional routing correctness for route_from_knowledge
  - Retrieval complement rate from SafetyExtractor Path B
"""

import logging

from sqlalchemy import text

from core.database import SessionLocal, with_db_retry

logger = logging.getLogger(__name__)

# Intents that require safety event logging. Used by safety_event_completeness.
_SAFETY_APPLICABLE_INTENTS = {"DIAGNOSTIC", "REPAIR", "SAFETY_PROTOCOL"}

# Expected safety event types within a qualifying session.
_REQUIRED_SAFETY_EVENT_GROUPS = {
    "safety_check_initiated",
    "safety_protocol_presented",
    "safety_confirmation_received",
}


@with_db_retry
def compute_fk_integrity() -> dict:
    """Validate that all chunk IDs logged in agent_execution_logs exist in L1.

    Extracts chunk IDs from the metadata JSONB field (key: retrieved_chunk_ids)
    and cross-references them against document_chunks_multimodal. Returns the
    fraction of logged chunk IDs that resolve successfully.
    """
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                WITH logged_chunks AS (
                    SELECT DISTINCT jsonb_array_elements_text(
                        metadata -> 'retrieved_chunk_ids'
                    ) AS chunk_id
                    FROM agent_execution_logs
                    WHERE metadata ? 'retrieved_chunk_ids'
                ),
                resolved AS (
                    SELECT lc.chunk_id, (c.id IS NOT NULL) AS exists
                    FROM logged_chunks lc
                    LEFT JOIN document_chunks_multimodal c ON c.id = lc.chunk_id
                )
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE exists = TRUE) AS resolved_count
                FROM resolved
                """
            )
        ).first()

    total = result[0] if result else 0
    resolved = result[1] if result else 0
    rate = resolved / total if total > 0 else 1.0  # vacuously valid when no logs

    return {
        "metric_name": "fk_integrity",
        "value": round(rate, 4),
        "metadata": {
            "total_logged_chunk_ids": total,
            "resolved_count": resolved,
            "broken_count": total - resolved,
        },
    }


@with_db_retry
def compute_safety_event_completeness(thread_id: str) -> dict:
    """For applicable intents, verify all required safety event groups were logged.

    Checks agent_execution_logs for the thread and determines whether every
    required safety event group (node_name) appears at least once. Returns the
    fraction of required groups that are present.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT DISTINCT node_name FROM agent_execution_logs "
                "WHERE thread_id = :tid"
            ),
            {"tid": thread_id},
        ).fetchall()

    observed: set[str] = {r[0] for r in rows}
    present = _REQUIRED_SAFETY_EVENT_GROUPS & observed
    completeness = len(present) / len(_REQUIRED_SAFETY_EVENT_GROUPS)

    return {
        "metric_name": "safety_event_completeness",
        "value": round(completeness, 4),
        "metadata": {
            "thread_id": thread_id,
            "required_groups": sorted(_REQUIRED_SAFETY_EVENT_GROUPS),
            "present_groups": sorted(present),
            "missing_groups": sorted(_REQUIRED_SAFETY_EVENT_GROUPS - present),
        },
    }


@with_db_retry
def compute_badge_escalation_rate() -> dict:
    """Fraction of diagnostic sessions in which badge escalation was triggered.

    NOTE (L2-SSE07): BadgeEscalation is delivered via frontend polling at
    GET /evaluation/shadow/{response_id}, not as a real-time SSE event. This is
    by design — shadow evaluation completes asynchronously after AGENT_DONE.

    A session is considered escalated when diagnostic_threads.is_escalated = TRUE
    or at least one EscalationFlag row with source AUTO_SAFETY or AUTO_EVAL exists.
    """
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total_sessions,
                    COUNT(*) FILTER (
                        WHERE is_escalated = TRUE
                           OR EXISTS (
                               SELECT 1 FROM escalation_flags ef
                               WHERE ef.thread_id = dt.id
                                 AND ef.source IN ('AUTO_SAFETY', 'AUTO_EVAL')
                           )
                    ) AS escalated_sessions
                FROM diagnostic_threads dt
                """
            )
        ).first()

    total = result[0] if result else 0
    escalated = result[1] if result else 0
    rate = escalated / total if total > 0 else 0.0

    return {
        "metric_name": "badge_escalation_rate",
        "value": round(rate, 4),
        "metadata": {
            "total_sessions": total,
            "escalated_sessions": escalated,
        },
    }


@with_db_retry
def compute_badge_escalation_latency() -> dict:
    """Mean and p95 latency in seconds from AGENT_DONE log entry to async_completed_at.

    Pairs the AGENT_DONE node_name timestamp in agent_execution_logs with the
    async_completed_at timestamp in evaluation_metrics for the same response_id.
    Only rows where both timestamps are populated are included.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    EXTRACT(EPOCH FROM (em.async_completed_at - ael.completed_at))
                        AS latency_seconds
                FROM evaluation_metrics em
                JOIN agent_execution_logs ael
                    ON ael.response_id = em.response_id
                   AND ael.node_name = 'AGENT_DONE'
                WHERE em.async_completed_at IS NOT NULL
                  AND ael.completed_at IS NOT NULL
                ORDER BY latency_seconds
                """
            )
        ).fetchall()

    latencies = [r[0] for r in rows if r[0] is not None and r[0] >= 0]
    n = len(latencies)

    if n == 0:
        return {
            "metric_name": "badge_escalation_latency",
            "value": None,
            "metadata": {"sample_size": 0},
        }

    mean_latency = sum(latencies) / n
    p95_index = max(0, int(n * 0.95) - 1)
    p95_latency = latencies[p95_index]

    return {
        "metric_name": "badge_escalation_latency",
        "value": round(mean_latency, 2),
        "metadata": {
            "sample_size": n,
            "mean_seconds": round(mean_latency, 2),
            "p95_seconds": round(p95_latency, 2),
            "min_seconds": round(latencies[0], 2),
            "max_seconds": round(latencies[-1], 2),
        },
    }


@with_db_retry
def compute_calibration_drift() -> dict:
    """Mean signed drift between sync and async scores per evaluation metric.

    Drift = sync_score - async_score, computed per metric dimension. A positive
    drift means sync over-estimates relative to the shadow evaluator; negative
    means it under-estimates. Only rows where both values are non-null are included.
    """
    metric_pairs = [
        ("sync_faithfulness", "async_faithfulness"),
        ("sync_answer_relevance", "async_answer_relevance"),
        ("sync_context_relevance", "async_context_relevance"),
        ("sync_completeness", "async_completeness"),
    ]

    drift_by_metric: dict[str, float | None] = {}
    sample_counts: dict[str, int] = {}

    with SessionLocal() as db:
        for sync_col, async_col in metric_pairs:
            row = db.execute(
                text(
                    f"SELECT AVG({sync_col} - {async_col}), COUNT(*) "
                    f"FROM evaluation_metrics "
                    f"WHERE {sync_col} IS NOT NULL AND {async_col} IS NOT NULL"
                )
            ).first()
            metric_key = sync_col.replace("sync_", "")
            drift_by_metric[metric_key] = (
                round(row[0], 4) if row[0] is not None else None
            )
            sample_counts[metric_key] = row[1] if row else 0

    # Overall mean drift across all metrics with data
    populated = [v for v in drift_by_metric.values() if v is not None]
    overall = round(sum(populated) / len(populated), 4) if populated else None

    return {
        "metric_name": "calibration_drift",
        "value": overall,
        "metadata": {
            "drift_by_metric": drift_by_metric,
            "sample_counts": sample_counts,
        },
    }


@with_db_retry
def compute_step_verdict_kappa() -> dict:
    """Cohen's kappa for inline (Flash-Lite) vs shadow (Flash) step verdicts.

    Extracts per-step binary verdicts from both evaluation runs and computes
    inter-rater reliability. Values: <0 = worse than chance, 0 = chance,
    0.6–0.8 = substantial agreement, >0.8 = near-perfect.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT step_verdicts, async_evidence
                FROM evaluation_metrics
                WHERE step_verdicts IS NOT NULL
                  AND async_evidence IS NOT NULL
                  AND async_evidence ? 'step_cross_validation'
                """
            )
        ).fetchall()

    if not rows:
        return {
            "metric_name": "step_verdict_kappa",
            "value": None,
            "metadata": {"sample_size": 0},
        }

    # Aggregate binary verdicts across all sessions
    agree_true = 0  # both say faithful
    agree_false = 0  # both say unfaithful
    inline_only = 0  # inline=faithful, shadow=unfaithful
    shadow_only = 0  # inline=unfaithful, shadow=faithful
    total_steps = 0

    for step_verdicts_json, async_evidence_json in rows:
        inline_verdicts = (
            step_verdicts_json if isinstance(step_verdicts_json, list) else []
        )
        cross_val = (
            async_evidence_json.get("step_cross_validation", {})
            if isinstance(async_evidence_json, dict)
            else {}
        )
        shadow_verdicts_list = cross_val.get("verdicts", [])

        shadow_by_step: dict[int, bool] = {}
        for sv in shadow_verdicts_list:
            if isinstance(sv, dict):
                shadow_by_step[sv.get("step_index", -1)] = sv.get("faithful", True)

        for i, iv in enumerate(inline_verdicts):
            if not isinstance(iv, dict):
                continue
            inline_f = iv.get("faithful", True)
            shadow_f = shadow_by_step.get(
                i, shadow_by_step.get(iv.get("step_index", i), None)
            )
            if shadow_f is None:
                continue
            total_steps += 1
            if inline_f and shadow_f:
                agree_true += 1
            elif not inline_f and not shadow_f:
                agree_false += 1
            elif inline_f and not shadow_f:
                inline_only += 1
            else:
                shadow_only += 1

    if total_steps == 0:
        return {
            "metric_name": "step_verdict_kappa",
            "value": None,
            "metadata": {"sample_size": 0, "total_steps": 0},
        }

    n = total_steps
    p_o = (agree_true + agree_false) / n
    p_inline_yes = (agree_true + inline_only) / n
    p_shadow_yes = (agree_true + shadow_only) / n
    p_e = p_inline_yes * p_shadow_yes + (1 - p_inline_yes) * (1 - p_shadow_yes)

    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1.0 else 1.0

    return {
        "metric_name": "step_verdict_kappa",
        "value": round(kappa, 4),
        "metadata": {
            "sample_size": len(rows),
            "total_steps": total_steps,
            "observed_agreement": round(p_o, 4),
            "expected_agreement": round(p_e, 4),
            "agree_true": agree_true,
            "agree_false": agree_false,
            "disagree_inline_only": inline_only,
            "disagree_shadow_only": shadow_only,
        },
    }


# Intents that should route to safety evaluation
_SAFETY_EVAL_INTENTS = {"troubleshoot", "replace_part", "procedural"}


@with_db_retry
def compute_routing_correctness() -> dict:
    """Fraction of sessions where route_from_knowledge made the correct decision.

    Correct = replace_part routes to SafetyExtractor (skips RCA);
    all other intents route to RootCauseAnalyzer. Cross-references
    route_from_knowledge edge decisions in agent_execution_logs.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    edge.thread_id,
                    edge.metadata ->> 'route' AS intent,
                    edge.metadata ->> 'target' AS edge_target
                FROM agent_execution_logs edge
                WHERE edge.node_name = 'route_from_knowledge'
                  AND edge.metadata IS NOT NULL
                """
            )
        ).fetchall()

    if not rows:
        # Fall back: check legacy route_to_safety_eval or infer from node presence
        with SessionLocal() as db:
            edge_rows = db.execute(
                text(
                    """
                    WITH extractor_threads AS (
                        SELECT DISTINCT thread_id
                        FROM agent_execution_logs
                        WHERE node_name IN ('SafetyExtractor', 'SafetyEvaluator')
                    )
                    SELECT
                        ael.thread_id,
                        ael.metadata ->> 'classified_intent' AS intent,
                        CASE WHEN st.thread_id IS NOT NULL
                             THEN 'executed' ELSE 'skipped'
                        END AS safety_eval_status
                    FROM agent_execution_logs ael
                    LEFT JOIN extractor_threads st ON st.thread_id = ael.thread_id
                    WHERE ael.node_name = 'IntentRouter'
                      AND ael.metadata IS NOT NULL
                    """
                )
            ).fetchall()

        if not edge_rows:
            return {
                "metric_name": "routing_correctness",
                "value": None,
                "metadata": {"sample_size": 0},
            }

        correct = 0
        total = len(edge_rows)
        for _, intent, safety_status in edge_rows:
            should_eval = (intent or "").lower() in _SAFETY_EVAL_INTENTS
            did_eval = safety_status == "executed"
            if should_eval == did_eval:
                correct += 1

        return {
            "metric_name": "routing_correctness",
            "value": round(correct / total, 4) if total > 0 else None,
            "metadata": {
                "sample_size": total,
                "correct_routes": correct,
                "method": "inferred_from_node_presence",
            },
        }

    correct = 0
    total = len(rows)
    for _, intent, edge_target in rows:
        should_eval = (intent or "").lower() in _SAFETY_EVAL_INTENTS
        did_eval = edge_target != "skip_safety_eval"
        if should_eval == did_eval:
            correct += 1

    return {
        "metric_name": "routing_correctness",
        "value": round(correct / total, 4) if total > 0 else None,
        "metadata": {
            "sample_size": total,
            "correct_routes": correct,
            "method": "edge_decision_logs",
        },
    }


@with_db_retry
def compute_complement_rate() -> dict:
    """Mean ratio of complement chunks (Path B minus Path A) to total Path B chunks.

    A high complement rate means Path B's targeted safety search adds unique
    chunks not found in the general KnowledgeRetriever results, indicating the
    dual-path approach provides value.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    (metadata ->> 'complement_count')::int AS complement,
                    (metadata ->> 'path_b_count')::int AS path_b
                FROM agent_execution_logs
                WHERE node_name = 'SafetyExtractor'
                  AND operation_type = 'node'
                  AND metadata ? 'complement_count'
                  AND metadata ? 'path_b_count'
                """
            )
        ).fetchall()

    if not rows:
        return {
            "metric_name": "complement_rate",
            "value": None,
            "metadata": {"sample_size": 0},
        }

    rates: list[float] = []
    for complement, path_b in rows:
        if path_b and path_b > 0:
            rates.append(complement / path_b)

    if not rates:
        return {
            "metric_name": "complement_rate",
            "value": 0.0,
            "metadata": {
                "sample_size": len(rows),
                "sessions_with_path_b": 0,
            },
        }

    mean_rate = sum(rates) / len(rates)

    return {
        "metric_name": "complement_rate",
        "value": round(mean_rate, 4),
        "metadata": {
            "sample_size": len(rows),
            "sessions_with_path_b": len(rates),
            "mean_complement_rate": round(mean_rate, 4),
        },
    }


# --- P3 New Aggregate Metrics (L2-M-NEW01 through L2-M-NEW05) ---


@with_db_retry
def compute_deterministic_shortcircuit_rate() -> dict:
    """Fraction of sessions resolved by DeterministicRuleChecker without RAG pipeline.

    Queries edge decisions from route_from_deterministic_check where the target
    was END (meaning a deterministic rule match short-circuited the pipeline).
    """
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total_sessions,
                    COUNT(*) FILTER (
                        WHERE metadata ->> 'has_override' = 'true'
                    ) AS shortcircuited
                FROM agent_execution_logs
                WHERE node_name = 'edge:route_from_deterministic_check'
                  AND operation_type = 'edge_decision'
                """
            )
        ).first()

    total = result[0] if result else 0
    shortcircuited = result[1] if result else 0
    rate = shortcircuited / total if total > 0 else 0.0

    return {
        "metric_name": "deterministic_shortcircuit_rate",
        "value": round(rate, 4),
        "metadata": {
            "total_sessions": total,
            "shortcircuited_sessions": shortcircuited,
        },
    }


@with_db_retry
def compute_retrieval_path_distribution() -> dict:
    """Distribution of retrieval paths (full_walkthrough/scoped_procedural/semantic).

    Queries KnowledgeRetriever telemetry records and extracts the retrieval_path
    metadata field to compute distribution percentages.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    metadata ->> 'retrieval_path' AS path,
                    COUNT(*) AS cnt
                FROM agent_execution_logs
                WHERE node_name = 'KnowledgeRetriever'
                  AND operation_type = 'node'
                  AND metadata ? 'retrieval_path'
                GROUP BY metadata ->> 'retrieval_path'
                """
            )
        ).fetchall()

    distribution: dict[str, int] = {r[0]: r[1] for r in rows if r[0]}
    total = sum(distribution.values())
    percentages: dict[str, float] = {
        k: round(v / total, 4) for k, v in distribution.items()
    } if total > 0 else {}

    return {
        "metric_name": "retrieval_path_distribution",
        "value": total,
        "metadata": {
            "total_sessions": total,
            "distribution": distribution,
            "percentages": percentages,
        },
    }


@with_db_retry
def compute_followup_invocation_rate() -> dict:
    """Fraction of sessions routed to FollowUpResponder.

    Queries edge decisions from route_from_intent where the target was
    FollowUpResponder to determine the follow-up invocation rate.
    """
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total_sessions,
                    COUNT(*) FILTER (
                        WHERE metadata ->> 'decision' = 'FollowUpResponder'
                    ) AS followup_sessions
                FROM agent_execution_logs
                WHERE node_name = 'edge:route_from_intent'
                  AND operation_type = 'edge_decision'
                """
            )
        ).first()

    total = result[0] if result else 0
    followup = result[1] if result else 0
    rate = followup / total if total > 0 else 0.0

    return {
        "metric_name": "followup_invocation_rate",
        "value": round(rate, 4),
        "metadata": {
            "total_sessions": total,
            "followup_sessions": followup,
        },
    }


@with_db_retry
def compute_retry_rate() -> dict:
    """Fraction of sessions where RepairPlanner was invoked more than once.

    A session with is_retry=true or attempt_number > 1 in RepairPlanner telemetry
    indicates that either SafetyJudgeGate or InlineEvaluator triggered a retry.
    """
    with SessionLocal() as db:
        result = db.execute(
            text(
                """
                WITH session_retries AS (
                    SELECT DISTINCT thread_id
                    FROM agent_execution_logs
                    WHERE node_name = 'RepairPlanner'
                      AND operation_type IN ('llm', 'node')
                      AND (
                          (metadata ->> 'is_retry')::boolean = TRUE
                          OR (metadata ->> 'attempt_number')::int > 1
                      )
                ),
                total_sessions AS (
                    SELECT COUNT(DISTINCT thread_id) AS cnt
                    FROM agent_execution_logs
                    WHERE node_name = 'RepairPlanner'
                      AND operation_type IN ('llm', 'node')
                )
                SELECT
                    ts.cnt AS total,
                    COUNT(sr.thread_id) AS retried
                FROM total_sessions ts
                LEFT JOIN session_retries sr ON TRUE
                GROUP BY ts.cnt
                """
            )
        ).first()

    total = result[0] if result else 0
    retried = result[1] if result else 0
    rate = retried / total if total > 0 else 0.0

    return {
        "metric_name": "retry_rate",
        "value": round(rate, 4),
        "metadata": {
            "total_sessions_with_planner": total,
            "sessions_with_retry": retried,
        },
    }


@with_db_retry
def compute_safety_judge_retry_split() -> dict:
    """For sessions that retried, split between SafetyJudgeGate and InlineEvaluator triggers.

    Determines whether the retry was triggered by the safety judge gate
    (route_from_safety_judge → RepairPlanner) or the inline evaluator
    (route_from_inline_evaluator → RepairPlanner).
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    node_name,
                    COUNT(*) AS cnt
                FROM agent_execution_logs
                WHERE operation_type = 'edge_decision'
                  AND node_name IN ('edge:route_from_safety_judge', 'edge:route_from_inline_evaluator')
                  AND metadata ->> 'decision' = 'RepairPlanner'
                GROUP BY node_name
                """
            )
        ).fetchall()

    counts: dict[str, int] = {r[0]: r[1] for r in rows}
    safety_judge_retries = counts.get("edge:route_from_safety_judge", 0)
    inline_eval_retries = counts.get("edge:route_from_inline_evaluator", 0)
    total_retries = safety_judge_retries + inline_eval_retries

    return {
        "metric_name": "safety_judge_retry_split",
        "value": total_retries,
        "metadata": {
            "total_retries": total_retries,
            "safety_judge_triggered": safety_judge_retries,
            "inline_eval_triggered": inline_eval_retries,
            "safety_judge_fraction": (
                round(safety_judge_retries / total_retries, 4)
                if total_retries > 0
                else None
            ),
            "inline_eval_fraction": (
                round(inline_eval_retries / total_retries, 4)
                if total_retries > 0
                else None
            ),
        },
    }
