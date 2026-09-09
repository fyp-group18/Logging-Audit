"""Process-mining metrics for workflow conformance analysis.

Covers:
  - Token-replay fitness: fraction of session traces conforming to the 12-node graph
  - Precision: fraction of model-allowed transitions that actually occur
  - Decision-mining: chi-squared correlations between state variables and routing decisions
"""

import logging
from collections import defaultdict

from scipy.stats import chi2 as chi2_dist
from sqlalchemy import text

from core.database import SessionLocal, with_db_retry

logger = logging.getLogger(__name__)

# Reference model: allowed transitions in the 12-node graph.
# Each key is a node name, values are the set of valid successor nodes.
# "__END__" represents the terminal state.
_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "__START__": {"IntentRouter"},
    "IntentRouter": {"UnsafeMethodGate", "DeterministicRuleChecker", "FollowUpResponder"},
    "UnsafeMethodGate": {"DeterministicRuleChecker", "FollowUpResponder"},
    "DeterministicRuleChecker": {"SymptomAnalyzer", "DirectReplacementAnalyzer", "__END__"},
    "SymptomAnalyzer": {"KnowledgeRetriever"},
    "DirectReplacementAnalyzer": {"KnowledgeRetriever"},
    "KnowledgeRetriever": {"RootCauseAnalyzer", "SafetyExtractor"},
    "RootCauseAnalyzer": {"SafetyExtractor", "__END__"},
    "SafetyExtractor": {"RepairPlanner"},
    "RepairPlanner": {"SafetyJudgeGate"},
    "SafetyJudgeGate": {"InlineEvaluator", "RepairPlanner", "__END__"},
    "FollowUpResponder": {"InlineEvaluator"},
    "InlineEvaluator": {"__END__"},
}

# All node names in the model
_ALL_NODES = set(_ALLOWED_TRANSITIONS.keys()) | {"__END__"}


def _extract_trace(rows: list) -> list[str]:
    """Extract ordered node sequence from execution log rows (sorted by started_at)."""
    return [r[0] for r in rows if r[0] and r[0] in _ALL_NODES - {"__START__", "__END__"}]


# Variant fingerprints: map (key_nodes_frozenset, terminal_node) → label.
# Checked in order — first match wins.
_VARIANT_SIGNATURES: list[tuple[str, set[str], str]] = [
    # 7: followup — FollowUpResponder present (always ends at InlineEvaluator)
    ("followup", {"FollowUpResponder", "InlineEvaluator"}, "InlineEvaluator"),
    # 1: deterministic exit — DeterministicRuleChecker is the terminal node
    ("troubleshoot_deterministic_exit", {"DeterministicRuleChecker"}, "DeterministicRuleChecker"),
    # 6: replace_part — DirectReplacementAnalyzer present
    ("replace_part", {"DirectReplacementAnalyzer", "InlineEvaluator"}, "InlineEvaluator"),
    # 2/8: RootCauseAnalyzer → END (no-docs or mismatch; indistinguishable by node sequence)
    ("troubleshoot_short_circuit", {"SymptomAnalyzer", "RootCauseAnalyzer"}, "RootCauseAnalyzer"),
]


def match_trace_variant(node_sequence: list[str]) -> dict:
    """Classify a single session's node sequence against known trace variants.

    Replays the sequence against _ALLOWED_TRANSITIONS to check conformance,
    then identifies which of the ~9 known execution paths it matches.

    Returns:
        {
            "conforming": bool,
            "variant_label": str | None,
            "deviating_edge": (str, str) | None,
        }
    """
    if not node_sequence:
        return {"conforming": True, "variant_label": None, "deviating_edge": None}

    # Deduplicate consecutive repeats (same node logged multiple times)
    deduped = [node_sequence[0]]
    for node in node_sequence[1:]:
        if node != deduped[-1]:
            deduped.append(node)

    # Replay against model
    deviating_edge: tuple[str, str] | None = None

    # Check start
    if deduped[0] not in _ALLOWED_TRANSITIONS.get("__START__", set()):
        deviating_edge = ("__START__", deduped[0])

    # Check transitions
    if deviating_edge is None:
        for i in range(len(deduped) - 1):
            current = deduped[i]
            next_node = deduped[i + 1]
            if next_node not in _ALLOWED_TRANSITIONS.get(current, set()):
                deviating_edge = (current, next_node)
                break

    # Check terminal
    if deviating_edge is None:
        last = deduped[-1]
        if "__END__" not in _ALLOWED_TRANSITIONS.get(last, set()):
            deviating_edge = (last, "__END__")

    conforming = deviating_edge is None
    node_set = set(deduped)
    terminal = deduped[-1] if deduped else None

    # Identify variant
    variant_label: str | None = None

    # Check signature-based variants first
    for label, required_nodes, expected_terminal in _VARIANT_SIGNATURES:
        if required_nodes <= node_set and terminal == expected_terminal:
            variant_label = label
            break

    # Fallback heuristics for troubleshoot paths that reach InlineEvaluator
    if variant_label is None and terminal == "InlineEvaluator":
        has_unsafe = "UnsafeMethodGate" in node_set
        # Count how many times SafetyJudgeGate → RepairPlanner occurs (retry loops)
        safety_retries = sum(
            1 for i in range(len(deduped) - 1)
            if deduped[i] == "SafetyJudgeGate" and deduped[i + 1] == "RepairPlanner"
        )
        eval_retries = sum(
            1 for i in range(len(deduped) - 1)
            if deduped[i] == "InlineEvaluator" and deduped[i + 1] == "RepairPlanner"
        )

        if has_unsafe:
            variant_label = "unsafe_method_redirect"
        elif eval_retries > 0:
            variant_label = "troubleshoot_eval_retry"
        elif safety_retries > 0:
            variant_label = "troubleshoot_safety_retry"
        else:
            variant_label = "troubleshoot_full"

    return {
        "conforming": conforming,
        "variant_label": variant_label,
        "deviating_edge": deviating_edge,
    }


@with_db_retry
def compute_process_fitness(min_sessions: int = 20) -> dict:
    """Token-replay fitness: fraction of observed transitions that are valid per the model.

    For each session trace, replays the node sequence against _ALLOWED_TRANSITIONS.
    A transition is "consumed" if it matches an allowed transition; otherwise it's
    a deviation. Fitness = consumed / (consumed + deviations).

    Returns None if fewer than min_sessions traces are available.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT trace_id, node_name, started_at
                FROM agent_execution_logs
                WHERE operation_type IN ('llm_call', 'node')
                  AND node_name IS NOT NULL
                  AND trace_id IS NOT NULL
                ORDER BY trace_id, started_at ASC
                """
            )
        ).fetchall()

    # Group by trace_id, normalizing suffixed node names (e.g. "InlineEvaluator:unified")
    _GRAPH_NODES = _ALL_NODES - {"__START__", "__END__"}
    traces: dict[str, list[str]] = defaultdict(list)
    for trace_id, node_name, _ in rows:
        base = node_name.split(":")[0] if ":" in node_name else node_name
        if base in _GRAPH_NODES:
            traces[trace_id].append(base)

    # Deduplicate consecutive repeated nodes (same node logged multiple times)
    deduped_traces: dict[str, list[str]] = {}
    for tid, trace in traces.items():
        deduped = [trace[0]] if trace else []
        for node in trace[1:]:
            if node != deduped[-1]:
                deduped.append(node)
        deduped_traces[tid] = deduped

    if len(deduped_traces) < min_sessions:
        return {
            "metric_name": "process_fitness",
            "value": None,
            "metadata": {
                "sample_size": len(deduped_traces),
                "min_required": min_sessions,
                "error": "insufficient session traces",
            },
        }

    total_consumed = 0
    total_deviations = 0
    conforming_traces = 0
    # trace_id -> the edges that could not be replayed, so a non-conforming
    # trace can be traced back to its session rather than only counted.
    non_conforming: dict[str, list[str]] = {}

    for tid, trace in deduped_traces.items():
        consumed = 0
        deviations = 0
        bad_edges: list[str] = []

        # Check start: first node should be reachable from __START__
        if trace and trace[0] in _ALLOWED_TRANSITIONS.get("__START__", set()):
            consumed += 1
        elif trace:
            deviations += 1
            bad_edges.append(f"__START__ -> {trace[0]}")

        # Check each transition
        for i in range(len(trace) - 1):
            current = trace[i]
            next_node = trace[i + 1]
            allowed = _ALLOWED_TRANSITIONS.get(current, set())
            if next_node in allowed:
                consumed += 1
            else:
                deviations += 1
                bad_edges.append(f"{current} -> {next_node}")

        # Check end: last node should be allowed to reach __END__
        if trace and "__END__" in _ALLOWED_TRANSITIONS.get(trace[-1], set()):
            consumed += 1
        elif trace:
            deviations += 1
            bad_edges.append(f"{trace[-1]} -> __END__")

        total_consumed += consumed
        total_deviations += deviations
        if deviations == 0:
            conforming_traces += 1
        else:
            non_conforming[str(tid)] = bad_edges

    total = total_consumed + total_deviations
    fitness = total_consumed / total if total > 0 else 1.0

    return {
        "metric_name": "process_fitness",
        "value": round(fitness, 4),
        "metadata": {
            "sample_size": len(deduped_traces),
            "consumed_transitions": total_consumed,
            "deviant_transitions": total_deviations,
            "fully_conforming_traces": conforming_traces,
            "conformance_rate": round(
                conforming_traces / len(deduped_traces), 4
            ),
            "non_conforming_traces": non_conforming,
        },
    }


@with_db_retry
def compute_process_precision(min_sessions: int = 20) -> dict:
    """Precision: fraction of model-allowed transitions that actually occurred in logs.

    A high precision means the model doesn't over-generalize — it only allows
    transitions that are actually taken. Low precision means many model paths
    are never exercised (the model is too permissive).
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT trace_id, node_name, started_at
                FROM agent_execution_logs
                WHERE operation_type IN ('llm_call', 'node')
                  AND node_name IS NOT NULL
                  AND trace_id IS NOT NULL
                ORDER BY trace_id, started_at ASC
                """
            )
        ).fetchall()

    # Group and deduplicate, normalizing suffixed node names
    _GRAPH_NODES = _ALL_NODES - {"__START__", "__END__"}
    traces: dict[str, list[str]] = defaultdict(list)
    for trace_id, node_name, _ in rows:
        base = node_name.split(":")[0] if ":" in node_name else node_name
        if base in _GRAPH_NODES:
            traces[trace_id].append(base)

    if len(traces) < min_sessions:
        return {
            "metric_name": "process_precision",
            "value": None,
            "metadata": {"sample_size": len(traces), "min_required": min_sessions},
        }

    # Collect observed transitions
    observed_transitions: set[tuple[str, str]] = set()
    for trace in traces.values():
        deduped = [trace[0]] if trace else []
        for node in trace[1:]:
            if node != deduped[-1]:
                deduped.append(node)
        if deduped and deduped[0] in _ALLOWED_TRANSITIONS.get("__START__", set()):
            observed_transitions.add(("__START__", deduped[0]))
        for i in range(len(deduped) - 1):
            observed_transitions.add((deduped[i], deduped[i + 1]))
        if deduped and "__END__" in _ALLOWED_TRANSITIONS.get(deduped[-1], set()):
            observed_transitions.add((deduped[-1], "__END__"))

    # Count total allowed transitions in model
    total_model_transitions: set[tuple[str, str]] = set()
    for source, targets in _ALLOWED_TRANSITIONS.items():
        for target in targets:
            total_model_transitions.add((source, target))

    # Precision = observed ∩ model / model
    valid_observed = observed_transitions & total_model_transitions
    precision = len(valid_observed) / len(total_model_transitions) if total_model_transitions else 1.0

    return {
        "metric_name": "process_precision",
        "value": round(precision, 4),
        "metadata": {
            "sample_size": len(traces),
            "model_transitions_total": len(total_model_transitions),
            "observed_transitions_total": len(observed_transitions),
            "model_transitions_exercised": len(valid_observed),
            "unexercised_transitions": sorted(
                [f"{s}->{t}" for s, t in (total_model_transitions - valid_observed)]
            ),
        },
    }


@with_db_retry
def compute_decision_mining() -> dict:
    """Chi-squared correlation between state variables and routing decisions.

    For each conditional edge, tests whether the distribution of routing targets
    is independent of key state variables (intent_type, equipment_category,
    is_procedural). Returns significant correlations with chi-squared statistic
    and p-values.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                """
                SELECT
                    node_name,
                    metadata ->> 'decision' AS target,
                    metadata ->> 'route' AS route,
                    metadata ->> 'is_procedural' AS is_procedural,
                    metadata ->> 'has_override' AS has_override
                FROM agent_execution_logs
                WHERE operation_type = 'edge_decision'
                  AND metadata IS NOT NULL
                  AND node_name IN (
                      'edge:route_from_intent',
                      'edge:route_from_deterministic_check',
                      'edge:route_to_safety_eval',
                      'edge:route_from_safety_eval',
                      'edge:route_from_root_cause',
                      'edge:route_from_safety_judge',
                      'edge:route_from_inline_evaluator'
                  )
                """
            )
        ).fetchall()

    if not rows:
        return {
            "metric_name": "decision_mining",
            "value": None,
            "metadata": {"sample_size": 0},
        }

    # Group by edge name
    edge_data: dict[str, list[dict]] = defaultdict(list)
    for node_name, target, route, is_proc, has_override in rows:
        edge_data[node_name].append({
            "target": target,
            "route": route,
            "is_procedural": is_proc,
            "has_override": has_override,
        })

    # Compute chi-squared for each edge × variable combination
    correlations: list[dict] = []

    for edge_name, decisions in edge_data.items():
        targets = [d["target"] for d in decisions if d["target"]]
        if len(set(targets)) < 2:
            continue  # No variation in targets — nothing to correlate

        # Test correlation with route variable
        routes = [d["route"] for d in decisions if d["route"]]
        if routes and len(set(routes)) >= 2:
            chi2, p_value = _chi_squared_test(
                [d["route"] for d in decisions],
                [d["target"] for d in decisions],
            )
            if chi2 is not None:
                correlations.append({
                    "edge": edge_name,
                    "variable": "route",
                    "chi_squared": round(chi2, 4),
                    "p_value": round(p_value, 6),
                    "significant": p_value < 0.05,
                    "sample_size": len(decisions),
                })

        # Test correlation with is_procedural
        proc_vals = [d["is_procedural"] for d in decisions if d["is_procedural"] is not None]
        if proc_vals and len(set(proc_vals)) >= 2:
            chi2, p_value = _chi_squared_test(
                [d["is_procedural"] for d in decisions],
                [d["target"] for d in decisions],
            )
            if chi2 is not None:
                correlations.append({
                    "edge": edge_name,
                    "variable": "is_procedural",
                    "chi_squared": round(chi2, 4),
                    "p_value": round(p_value, 6),
                    "significant": p_value < 0.05,
                    "sample_size": len(decisions),
                })

    significant_count = sum(1 for c in correlations if c["significant"])

    return {
        "metric_name": "decision_mining",
        "value": significant_count,
        "metadata": {
            "total_edges_analyzed": len(edge_data),
            "total_correlations_tested": len(correlations),
            "significant_correlations": significant_count,
            "correlations": correlations,
        },
    }


def _chi_squared_test(
    variable: list[str | None], target: list[str | None]
) -> tuple[float | None, float]:
    """Compute chi-squared statistic for independence between variable and target.

    Returns (chi2, p_value) or (None, 1.0) if the test cannot be computed.
    """
    # Build contingency table
    var_vals = sorted(set(v for v in variable if v is not None))
    tgt_vals = sorted(set(t for t in target if t is not None))

    if len(var_vals) < 2 or len(tgt_vals) < 2:
        return None, 1.0

    # Count occurrences
    table: dict[tuple[str, str], int] = defaultdict(int)
    for v, t in zip(variable, target):
        if v is not None and t is not None:
            table[(v, t)] += 1

    n = sum(table.values())
    if n == 0:
        return None, 1.0

    # Row and column totals
    row_totals = {v: sum(table.get((v, t), 0) for t in tgt_vals) for v in var_vals}
    col_totals = {t: sum(table.get((v, t), 0) for v in var_vals) for t in tgt_vals}

    # Chi-squared statistic
    chi2 = 0.0
    for v in var_vals:
        for t in tgt_vals:
            observed = table.get((v, t), 0)
            expected = (row_totals[v] * col_totals[t]) / n if n > 0 else 0
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected

    # Approximate p-value using chi-squared distribution
    # df = (rows - 1) * (cols - 1)
    df = (len(var_vals) - 1) * (len(tgt_vals) - 1)
    if df == 0:
        return None, 1.0

    p_value = float(chi2_dist.sf(chi2, df))

    return chi2, max(0.0, min(1.0, p_value))


def _main() -> int:
    """CLI entry point: token-replay fitness and precision over all logged traces."""
    import json
    import os

    if not os.getenv("DATABASE_URL"):
        print("DATABASE_URL is not set — export it before running this module.")
        return 1

    print(json.dumps({
        "process_fitness": compute_process_fitness(),
        "process_precision": compute_process_precision(),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
