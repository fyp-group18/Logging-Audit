"""
Run the 4-step diagnostic procedure against the live DB for all 12 failure-
injection sessions and write diagnostic_trace_log.json with observations
pre-populated.  Judgment fields are left empty for manual review.

Usage:
    cd backend && uv run python -m eval.scripts.run_diagnostics \
        --config-dir eval/results/failure_injection/experiment_run_001 \
        --db-url "$DATABASE_URL" \
        --output eval/results/failure_injection/experiment_run_001/diagnostic_trace_log.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, text


# ---------------------------------------------------------------------------
# Trace variant classification (ported from audit/process_mining.py)
# ---------------------------------------------------------------------------

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

_ALL_NODES = set(_ALLOWED_TRANSITIONS.keys()) | {"__END__"}

_VARIANT_SIGNATURES: list[tuple[str, set[str], str]] = [
    ("followup", {"FollowUpResponder", "InlineEvaluator"}, "InlineEvaluator"),
    ("troubleshoot_deterministic_exit", {"DeterministicRuleChecker"}, "DeterministicRuleChecker"),
    ("replace_part", {"DirectReplacementAnalyzer", "InlineEvaluator"}, "InlineEvaluator"),
    ("troubleshoot_short_circuit", {"SymptomAnalyzer", "RootCauseAnalyzer"}, "RootCauseAnalyzer"),
]

# Maps expected_trace_variant number (from eval_queries.json) to allowed labels
_VARIANT_NUM_TO_LABELS: dict[int, set[str]] = {
    1: {"troubleshoot_deterministic_exit"},
    2: {"followup"},
    3: {"troubleshoot_short_circuit", "troubleshoot_full"},
    4: {"troubleshoot_full", "troubleshoot_safety_retry", "troubleshoot_eval_retry"},
    5: {"replace_part"},
    6: {"unsafe_method_redirect"},
    7: {"unsafe_method_redirect"},
}


def _classify_trace_variant(node_sequence: list[str]) -> dict:
    """Classify a node sequence against known trace variants."""
    if not node_sequence:
        return {"conforming": True, "variant_label": None, "deviating_edge": None}

    deduped = [node_sequence[0]]
    for node in node_sequence[1:]:
        if node != deduped[-1]:
            deduped.append(node)

    deviating_edge: tuple[str, str] | None = None
    if deduped[0] not in _ALLOWED_TRANSITIONS.get("__START__", set()):
        deviating_edge = ("__START__", deduped[0])

    if deviating_edge is None:
        for i in range(len(deduped) - 1):
            current, next_node = deduped[i], deduped[i + 1]
            if next_node not in _ALLOWED_TRANSITIONS.get(current, set()):
                deviating_edge = (current, next_node)
                break

    if deviating_edge is None:
        last = deduped[-1]
        if "__END__" not in _ALLOWED_TRANSITIONS.get(last, set()):
            deviating_edge = (last, "__END__")

    node_set = set(deduped)
    terminal = deduped[-1] if deduped else None
    variant_label: str | None = None

    for label, required_nodes, expected_terminal in _VARIANT_SIGNATURES:
        if required_nodes <= node_set and terminal == expected_terminal:
            variant_label = label
            break

    if variant_label is None and terminal == "InlineEvaluator":
        has_unsafe = "UnsafeMethodGate" in node_set
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
        "conforming": deviating_edge is None,
        "variant_label": variant_label,
        "deviating_edge": deviating_edge,
    }


# ---------------------------------------------------------------------------
# Manifest integrity check (ported from audit/manifest.py)
# ---------------------------------------------------------------------------

def _verify_manifest_for_document(conn, document_id: int) -> dict:
    """Verify hash-chain integrity for a single document's manifest entries."""
    rows = conn.execute(
        text(
            "SELECT entry_sequence, entry_hash, previous_hash, entry_data "
            "FROM ingestion_manifest "
            "WHERE document_id = :doc_id "
            "ORDER BY entry_sequence ASC"
        ),
        {"doc_id": document_id},
    ).fetchall()

    if not rows:
        return {"valid": True, "entries_checked": 0, "first_broken_at": None}

    for i, (seq, stored_hash, stored_prev, entry_data) in enumerate(rows):
        data_dict = entry_data if isinstance(entry_data, dict) else json.loads(entry_data)
        data_json = json.dumps(data_dict, default=str, sort_keys=True)
        expected_hash = hashlib.sha256((stored_prev + data_json).encode()).hexdigest()

        if expected_hash != stored_hash:
            return {
                "valid": False,
                "entries_checked": i + 1,
                "first_broken_at": seq,
            }

    return {"valid": True, "entries_checked": len(rows), "first_broken_at": None}


# ---------------------------------------------------------------------------
# Diagnostic steps
# ---------------------------------------------------------------------------

def _step1_overview(conn, session: dict, expected_variant: int | None) -> tuple[dict, list[dict]]:
    """Step 1: Overview — node sequence, count, duration, variant."""
    tid = session["thread_id"]
    rid = session["response_id"]

    rows = conn.execute(
        text(
            "SELECT node_name, operation_type, started_at, completed_at, "
            "latency_ms, metadata "
            "FROM agent_execution_logs "
            "WHERE thread_id = :tid AND response_id = :rid "
            "ORDER BY started_at ASC"
        ),
        {"tid": tid, "rid": rid},
    ).mappings().fetchall()

    # Also fetch rows without response_id filter (some logs may have different response_id)
    all_rows = conn.execute(
        text(
            "SELECT node_name, operation_type, started_at, completed_at, "
            "latency_ms, metadata "
            "FROM agent_execution_logs "
            "WHERE thread_id = :tid "
            "ORDER BY started_at ASC"
        ),
        {"tid": tid},
    ).mappings().fetchall()

    # Use whichever has more rows (the response_id filter may be too narrow)
    log_rows = list(all_rows) if len(all_rows) > len(rows) else list(rows)

    # Extract node names (excluding edge decisions for variant classification)
    node_names_raw = [
        r["node_name"] for r in log_rows
        if r["node_name"] and not r["node_name"].startswith("edge:")
    ]
    # Normalize: strip ":unified" suffix from InlineEvaluator
    node_names = [n.replace(":unified", "") for n in node_names_raw]

    node_count = len(node_names)

    # Duration
    timestamps = [r["started_at"] for r in log_rows if r["started_at"]]
    end_timestamps = [r["completed_at"] for r in log_rows if r["completed_at"]]
    all_times = timestamps + end_timestamps
    if all_times:
        start = min(timestamps) if timestamps else min(all_times)
        end = max(end_timestamps) if end_timestamps else max(all_times)
        duration_s = (end - start).total_seconds()
    else:
        duration_s = 0.0

    # Variant classification
    variant_info = _classify_trace_variant(node_names)
    variant_label = variant_info["variant_label"] or "unknown"

    # Short node sequence for display
    abbrev = {
        "IntentRouter": "IR", "DeterministicRuleChecker": "DRC",
        "SymptomAnalyzer": "SA", "KnowledgeRetriever": "KR",
        "RootCauseAnalyzer": "RCA", "SafetyExtractor": "SE",
        "RepairPlanner": "RP", "SafetyJudgeGate": "SJG",
        "InlineEvaluator": "IE", "FollowUpResponder": "FUR",
        "UnsafeMethodGate": "UMG", "DirectReplacementAnalyzer": "DRA",
    }
    deduped_display = []
    for n in node_names:
        short = abbrev.get(n, n)
        if not deduped_display or deduped_display[-1] != short:
            deduped_display.append(short)
    seq_str = "→".join(deduped_display)

    # Variant match check
    if expected_variant is not None:
        allowed = _VARIANT_NUM_TO_LABELS.get(expected_variant, set())
        match_str = "✓" if variant_label in allowed else f"✗ (expected variant {expected_variant})"
    else:
        match_str = "N/A"

    observation = (
        f"{node_count} nodes: {seq_str}. "
        f"Duration: {duration_s:.1f}s. "
        f"Variant: {variant_label} ({match_str})"
    )
    if not variant_info["conforming"]:
        observation += f". Non-conforming edge: {variant_info['deviating_edge']}"

    trace_entry = {
        "step": 1,
        "action": f"Queried node sequence and timing overview",
        "observation": observation,
        "inference": "",
    }

    return trace_entry, log_rows


def _step2_cross_layer(conn, session: dict, node_names: list[str]) -> list[dict]:
    """Step 2: Cross-layer metrics — Provenance Completeness, Safety Provenance, Sync/Async Scores."""
    rid = session["response_id"]
    traces = []
    metrics_found = []

    # Provenance Completeness
    total = conn.execute(
        text("SELECT COUNT(*) FROM response_chunk_link WHERE response_id = :rid"),
        {"rid": rid},
    ).scalar()

    resolved = conn.execute(
        text(
            "SELECT COUNT(*) "
            "FROM response_chunk_link rcl "
            "JOIN document_chunks_multimodal dcm ON dcm.id = rcl.chunk_id "
            "WHERE rcl.response_id = :rid AND dcm.document_id IS NOT NULL"
        ),
        {"rid": rid},
    ).scalar()

    prov_pct = (resolved / total * 100) if total else 0
    traces.append({
        "step": 2,
        "action": "Computed provenance completeness",
        "observation": f"{resolved}/{total} chunks resolve to documents ({prov_pct:.0f}%)",
        "inference": "",
    })
    metrics_found.append("provenance_completeness")

    # Safety Provenance (only if SafetyExtractor fired)
    has_safety_extractor = "SafetyExtractor" in node_names
    if has_safety_extractor:
        safety_rows = conn.execute(
            text(
                "SELECT rcl.chunk_id, dcm.has_safety_content, dcm.document_id, "
                "  (d.id IS NOT NULL) as doc_exists "
                "FROM response_chunk_link rcl "
                "JOIN document_chunks_multimodal dcm ON dcm.id = rcl.chunk_id "
                "LEFT JOIN documents_multimodal d ON d.id = dcm.document_id "
                "WHERE rcl.response_id = :rid AND dcm.has_safety_content = TRUE"
            ),
            {"rid": rid},
        ).mappings().fetchall()

        total_safety = len(safety_rows)
        complete_safety = sum(1 for r in safety_rows if r["doc_exists"])
        safety_pct = (complete_safety / total_safety * 100) if total_safety else 0
        traces.append({
            "step": 2,
            "action": "Computed safety provenance completeness",
            "observation": (
                f"{complete_safety}/{total_safety} safety chunks have complete provenance chain "
                f"({safety_pct:.0f}%)"
            ),
            "inference": "",
        })
        metrics_found.append("safety_provenance")

    # Sync vs Async Scores
    eval_row = conn.execute(
        text(
            "SELECT sync_faithfulness, async_faithfulness, "
            "sync_answer_relevance, async_answer_relevance, "
            "sync_context_relevance, async_context_relevance, "
            "sync_completeness, async_completeness, "
            "badge, async_completed_at "
            "FROM evaluation_metrics "
            "WHERE response_id = :rid "
            "ORDER BY id LIMIT 1"
        ),
        {"rid": rid},
    ).mappings().fetchone()

    if eval_row:
        dims = [
            ("faithfulness", eval_row["sync_faithfulness"], eval_row["async_faithfulness"]),
            ("answer_relevance", eval_row["sync_answer_relevance"], eval_row["async_answer_relevance"]),
            ("context_relevance", eval_row["sync_context_relevance"], eval_row["async_context_relevance"]),
            ("completeness", eval_row["sync_completeness"], eval_row["async_completeness"]),
        ]
        parts = []
        for name, sync_val, async_val in dims:
            if sync_val is not None and async_val is not None:
                delta = abs(sync_val - async_val)
                parts.append(f"{name}: sync={sync_val:.2f} async={async_val:.2f} Δ={delta:.2f}")
            elif sync_val is not None:
                parts.append(f"{name}: sync={sync_val:.2f} async=NULL")
            else:
                parts.append(f"{name}: sync=NULL async={'%.2f' % async_val if async_val is not None else 'NULL'}")

        shadow_done = eval_row["async_completed_at"] is not None
        badge = eval_row["badge"]
        obs = "; ".join(parts) + f". Badge={badge}. Shadow={'done' if shadow_done else 'pending'}"
        traces.append({
            "step": 2,
            "action": "Compared sync vs async evaluation scores",
            "observation": obs,
            "inference": "",
        })
        metrics_found.append("sync_async_scores")
    else:
        traces.append({
            "step": 2,
            "action": "Looked up evaluation_metrics for sync vs async scores",
            "observation": "No evaluation_metrics row found for this response_id",
            "inference": "",
        })

    return traces, metrics_found


def _step3_trace_walk(log_rows: list[dict], variant_label: str) -> tuple[dict, list[str]]:
    """Step 3: Per-node trace walk — flag anomalies."""
    anomalies = []
    node_details = []

    for r in log_rows:
        name = r["node_name"]
        if not name or name.startswith("edge:"):
            continue
        name = name.replace(":unified", "")

        issues = []
        if r["completed_at"] is None:
            issues.append("NULL completed_at")
        if r["latency_ms"] is None:
            issues.append("NULL latency_ms")
        elif r["latency_ms"] == 0:
            issues.append("latency_ms=0")
        elif r["latency_ms"] > 120000:
            issues.append(f"latency_ms={r['latency_ms']} (>120s)")
        if r["metadata"] is None:
            issues.append("NULL metadata")

        status = f" [{', '.join(issues)}]" if issues else ""
        latency_str = f"{r['latency_ms']}ms" if r["latency_ms"] is not None else "?ms"
        node_details.append(f"{name}({latency_str}){status}")

        if issues:
            for issue in issues:
                anomalies.append(f"{name}: {issue}")

    observation = "; ".join(node_details)

    trace_entry = {
        "step": 3,
        "action": "Per-node trace walk — checked timing, metadata, completion",
        "observation": observation,
        "inference": "",
    }

    return trace_entry, anomalies


def _step4_content_check(conn, session: dict) -> tuple[list[dict], list[str]]:
    """Step 4: Content spot-check — 2 random chunks, manifest integrity, eval scores."""
    rid = session["response_id"]
    traces = []
    anomalies = []

    # 2 random chunks
    chunk_rows = conn.execute(
        text(
            "SELECT rcl.id as link_id, rcl.chunk_id, "
            "  dcm.id as chunk_exists, "
            "  LEFT(dcm.text, 200) as chunk_preview, "
            "  dcm.document_id, dcm.has_safety_content, "
            "  d.original_filename as source_document "
            "FROM response_chunk_link rcl "
            "LEFT JOIN document_chunks_multimodal dcm ON dcm.id = rcl.chunk_id "
            "LEFT JOIN documents_multimodal d ON d.id = dcm.document_id "
            "WHERE rcl.response_id = :rid "
            "ORDER BY RANDOM() LIMIT 2"
        ),
        {"rid": rid},
    ).mappings().fetchall()

    if not chunk_rows:
        traces.append({
            "step": 4,
            "action": "Content spot-check — sampled 2 random chunks",
            "observation": "No response_chunk_link rows found for this response_id",
            "inference": "",
        })
        anomalies.append("No chunk links found")
        return traces, anomalies

    chunk_parts = []
    doc_ids_seen = set()
    for cr in chunk_rows:
        if cr["chunk_exists"] is None:
            chunk_parts.append(
                f"chunk_id={cr['chunk_id'][:8]}… — MISSING (link_id={cr['link_id']} "
                f"exists but chunk row deleted)"
            )
            anomalies.append(f"Chunk {cr['chunk_id'][:12]}… missing from document_chunks_multimodal")
        else:
            preview = (cr["chunk_preview"] or "").replace("\n", " ")[:120]
            safety_tag = " [SAFETY]" if cr["has_safety_content"] else ""
            chunk_parts.append(
                f"chunk_id={cr['chunk_id'][:8]}… — present{safety_tag}. "
                f"Source: \"{cr['source_document'] or 'unknown'}\". "
                f"Preview: \"{preview}…\""
            )
            if cr["document_id"] is not None:
                doc_ids_seen.add(cr["document_id"])

    traces.append({
        "step": 4,
        "action": "Content spot-check — sampled 2 random chunks",
        "observation": " | ".join(chunk_parts),
        "inference": "",
    })

    # Manifest integrity per referenced document
    for doc_id in doc_ids_seen:
        result = _verify_manifest_for_document(conn, doc_id)
        if result["valid"]:
            obs = f"document_id={doc_id}: manifest valid ({result['entries_checked']} entries checked)"
        else:
            obs = (
                f"document_id={doc_id}: MANIFEST INVALID — "
                f"first break at entry_sequence={result['first_broken_at']} "
                f"(checked {result['entries_checked']} entries)"
            )
            anomalies.append(f"Manifest integrity violation for document_id={doc_id}")

        traces.append({
            "step": 4,
            "action": f"Verified manifest integrity for document_id={doc_id}",
            "observation": obs,
            "inference": "",
        })

    # Evaluation metrics anomaly check
    eval_row = conn.execute(
        text(
            "SELECT sync_faithfulness, sync_answer_relevance, "
            "sync_context_relevance, sync_completeness "
            "FROM evaluation_metrics "
            "WHERE response_id = :rid "
            "ORDER BY id LIMIT 1"
        ),
        {"rid": rid},
    ).mappings().fetchone()

    if eval_row:
        score_issues = []
        for col in ("sync_faithfulness", "sync_answer_relevance",
                     "sync_context_relevance", "sync_completeness"):
            val = eval_row[col]
            if val is not None and val == 0.0:
                score_issues.append(f"{col}=0.0")
        if score_issues:
            obs = f"Anomalous scores: {', '.join(score_issues)}"
            anomalies.append(obs)
        else:
            obs = (
                f"Sync scores: faithfulness={eval_row['sync_faithfulness']}, "
                f"relevance={eval_row['sync_answer_relevance']}, "
                f"context={eval_row['sync_context_relevance']}, "
                f"completeness={eval_row['sync_completeness']}"
            )
        traces.append({
            "step": 4,
            "action": "Checked evaluation_metrics for score anomalies",
            "observation": obs,
            "inference": "",
        })

    return traces, anomalies


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_diagnostics(db_url: str, config_dir: Path, output_path: Path) -> None:
    """Run 4-step diagnostics for all 12 sessions and write results."""
    # Load session order (blinded)
    session_order_path = config_dir / "session_order.json"
    with open(session_order_path) as f:
        session_order = json.load(f)

    # Load failure injection sessions for expected_trace_variant
    fi_sessions_path = config_dir.parent / "failure_injection_sessions.json"
    with open(fi_sessions_path) as f:
        fi_sessions = {s["session_id"]: s for s in json.load(f)}

    # Load template for manifest_hash
    template_path = config_dir / "diagnostic_trace_log_template.json"
    with open(template_path) as f:
        template = json.load(f)

    engine = create_engine(db_url)
    sessions_out = []

    print(f"Running diagnostics for {len(session_order)} sessions...")
    print(f"{'='*60}\n")

    with engine.connect() as conn:
        for idx, session in enumerate(session_order, 1):
            sid = session["session_id"]
            fi_info = fi_sessions.get(sid, {})
            expected_variant = fi_info.get("expected_trace_variant")

            print(f"  [{idx:2d}/12] {sid}...", end=" ", flush=True)
            ts_start = datetime.now(timezone.utc).isoformat()

            # Step 1: Overview
            trace1, log_rows = _step1_overview(conn, session, expected_variant)

            # Extract node names for step 2
            node_names = [
                r["node_name"].replace(":unified", "")
                for r in log_rows
                if r["node_name"] and not r["node_name"].startswith("edge:")
            ]

            # Step 2: Cross-layer metrics
            trace2_list, metrics_examined = _step2_cross_layer(conn, session, node_names)

            # Step 3: Per-node trace walk
            trace3, walk_anomalies = _step3_trace_walk(log_rows, "")

            # Step 4: Content spot-check
            trace4_list, content_anomalies = _step4_content_check(conn, session)

            ts_end = datetime.now(timezone.utc).isoformat()
            all_anomalies = walk_anomalies + content_anomalies

            # Assemble session record
            reasoning_trace = [trace1] + trace2_list + [trace3] + trace4_list

            sessions_out.append({
                "session_id": sid,
                "response_id": session["response_id"],
                "thread_id": session["thread_id"],
                "time_started": ts_start,
                "time_completed": ts_end,
                "time_to_diagnosis_seconds": None,
                "failure_detected": None,
                "confidence": "",
                "failure_type_guess": "",
                "root_cause_description": "",
                "metrics_examined": metrics_examined,
                "anomalies_found": all_anomalies,
                "reasoning_trace": reasoning_trace,
            })

            anomaly_tag = f" ({len(all_anomalies)} anomalies)" if all_anomalies else ""
            print(f"OK — {len(reasoning_trace)} observations{anomaly_tag}")

    # Write output
    output = {
        "evaluator": "blinded evaluator",
        "manifest_hash": template.get("manifest_hash", ""),
        "evaluation_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sessions": sessions_out,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"\n{'='*60}")
    print(f"Diagnostic trace log written to {output_path}")
    print(f"  Sessions: {len(sessions_out)}")
    print(f"  Total observations: {sum(len(s['reasoning_trace']) for s in sessions_out)}")
    total_anomalies = sum(len(s["anomalies_found"]) for s in sessions_out)
    print(f"  Total anomalies flagged: {total_anomalies}")
    print(f"\n  Next: review each session and fill in failure_detected, ")
    print(f"  failure_type_guess, confidence, root_cause_description, inference fields.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 4-step diagnostic procedure for failure injection sessions"
    )
    parser.add_argument(
        "--config-dir", type=Path, required=True,
        help="Experiment directory (e.g. experiment_run_001)",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="DATABASE_URL (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for diagnostic_trace_log.json",
    )
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: --db-url or DATABASE_URL required")
        sys.exit(1)

    run_diagnostics(db_url, args.config_dir, args.output)


if __name__ == "__main__":
    main()
