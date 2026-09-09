"""
Randomize failure injection assignments for the double-blind protocol.

The injection designer runs this script to:
1. Split 12 sessions into 6 injection + 6 control (non-reproducible seed)
2. Assign failure types with constraints (all 4 types used, safety chunk_deletion)
3. Query DB for concrete injection targets per session
4. Output sealed experiment artifacts

Outputs:
    assignments.json         — injection designer only (ground truth)
    manifest_sha256.txt      — Both authors (tamper evidence)
    session_order.json       — blinded evaluator only (randomized session list)
    injection_configs/       — injection designer only (per-session configs)
    diagnostic_trace_log_template.json — blinded evaluator (blank diagnosis form)

Usage:
    cd backend && uv run python -m eval.scripts.randomize_failures \
        --sessions eval/results/failure_injection/failure_injection_sessions.json \
        --out-dir eval/results/failure_injection/experiment_run_001 \
        --db-url "$DATABASE_URL"

    Add --dry-run to skip DB queries (generates assignments with placeholder targets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

FAILURE_TYPES = ["chunk_deletion", "score_perturbation", "log_gap", "manifest_tamper"]


def shuffle_and_split(sessions: list[dict], seed_hex: str) -> tuple[list[dict], list[dict]]:
    """Non-reproducible shuffle, split into injection (first 6) and control (last 6)."""
    import random

    rng = random.Random(seed_hex)
    shuffled = list(sessions)
    rng.shuffle(shuffled)
    return shuffled[:6], shuffled[6:]


def assign_failure_types(
    injection_group: list[dict], seed_hex: str
) -> list[tuple[dict, str]]:
    """Assign failure types ensuring all 4 used, with safety constraint on chunk_deletion."""
    import random

    rng = random.Random(seed_hex + "_types")

    safety_sessions = [s for s in injection_group if s.get("safety_expected")]

    # Build assignment list: mandatory 4 types + 2 random
    mandatory = list(FAILURE_TYPES)
    extra = [rng.choice(FAILURE_TYPES) for _ in range(2)]
    type_pool = mandatory + extra
    rng.shuffle(type_pool)

    # Ensure at least one chunk_deletion targets a safety session
    chunk_del_indices = [i for i, t in enumerate(type_pool) if t == "chunk_deletion"]

    # Try to place a chunk_deletion on a safety session
    assignments: list[tuple[dict, str]] = []
    safety_chunk_placed = False
    used_sessions: set[str] = set()

    # First pass: assign chunk_deletion to a safety session if possible
    if safety_sessions and chunk_del_indices:
        safety_target = safety_sessions[0]
        assignments.append((safety_target, "chunk_deletion"))
        used_sessions.add(safety_target["session_id"])
        type_pool.pop(chunk_del_indices[0])
        safety_chunk_placed = True

    # Fill remaining
    remaining_sessions = [s for s in injection_group if s["session_id"] not in used_sessions]
    rng.shuffle(remaining_sessions)

    for session, ftype in zip(remaining_sessions, type_pool):
        assignments.append((session, ftype))

    if not safety_chunk_placed:
        print(
            "WARNING: Could not assign chunk_deletion to a safety_expected session. "
            "No safety sessions in injection group."
        )

    return assignments


def query_chunk_deletion_target(conn, response_id: str, safety_required: bool) -> dict | None:
    """Find a response_chunk_link row to delete."""
    from sqlalchemy import text

    if safety_required:
        rows = conn.execute(
            text(
                "SELECT rcl.id, rcl.response_id, rcl.chunk_id, rcl.step_index, "
                "rcl.grounding_label, rcl.similarity_score, rcl.reranker_score, "
                "dcm.has_safety_content "
                "FROM response_chunk_link rcl "
                "JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id "
                "WHERE rcl.response_id = :rid AND dcm.has_safety_content = true "
                "ORDER BY rcl.step_index LIMIT 1"
            ),
            {"rid": response_id},
        ).mappings().fetchone()
    else:
        rows = conn.execute(
            text(
                "SELECT id, response_id, chunk_id, step_index, grounding_label, "
                "similarity_score, reranker_score "
                "FROM response_chunk_link "
                "WHERE response_id = :rid ORDER BY step_index LIMIT 1"
            ),
            {"rid": response_id},
        ).mappings().fetchone()

    if rows is None:
        return None
    return {"target_id": rows["id"], "response_id": response_id}


def query_score_perturbation_target(conn, response_id: str) -> dict | None:
    """Find the evaluation_metrics row for the response."""
    from sqlalchemy import text

    row = conn.execute(
        text(
            "SELECT id, sync_faithfulness FROM evaluation_metrics "
            "WHERE response_id = :rid ORDER BY id LIMIT 1"
        ),
        {"rid": response_id},
    ).mappings().fetchone()

    if row is None:
        return None
    return {"response_id": response_id, "eval_metrics_id": row["id"]}


def query_log_gap_target(conn, thread_id: str) -> dict | None:
    """Find a middle node in agent_execution_logs for the thread."""
    from sqlalchemy import text

    rows = conn.execute(
        text(
            "SELECT id, node_name FROM agent_execution_logs "
            "WHERE thread_id = :tid ORDER BY started_at"
        ),
        {"tid": thread_id},
    ).mappings().fetchall()

    if len(rows) < 3:
        return None

    # Pick from the middle (not first or last)
    mid_idx = len(rows) // 2
    target = rows[mid_idx]
    return {"target_id": target["id"], "node_name": target["node_name"]}


def query_manifest_tamper_target(conn, response_id: str) -> dict | None:
    """Find an ingestion_manifest entry for a document linked to retrieved chunks."""
    from sqlalchemy import text

    # Get a document_id from the chunks linked to this response
    doc_row = conn.execute(
        text(
            "SELECT DISTINCT dcm.document_id "
            "FROM response_chunk_link rcl "
            "JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id "
            "WHERE rcl.response_id = :rid AND dcm.document_id IS NOT NULL "
            "LIMIT 1"
        ),
        {"rid": response_id},
    ).mappings().fetchone()

    if doc_row is None:
        return None

    document_id = doc_row["document_id"]

    manifest_row = conn.execute(
        text(
            "SELECT id, entry_hash FROM ingestion_manifest "
            "WHERE document_id = :did ORDER BY entry_sequence LIMIT 1"
        ),
        {"did": document_id},
    ).mappings().fetchone()

    if manifest_row is None:
        return None

    return {
        "target_id": manifest_row["id"],
        "document_id": document_id,
        "original_hash_prefix": manifest_row["entry_hash"][:8],
    }


TARGET_QUERY_FNS = {
    "chunk_deletion": lambda conn, s, safety: query_chunk_deletion_target(
        conn, s["response_id"], safety
    ),
    "score_perturbation": lambda conn, s, _: query_score_perturbation_target(
        conn, s["response_id"]
    ),
    "log_gap": lambda conn, s, _: query_log_gap_target(conn, s["thread_id"]),
    "manifest_tamper": lambda conn, s, _: query_manifest_tamper_target(
        conn, s["response_id"]
    ),
}


def build_assignments(
    sessions: list[dict],
    db_url: str | None,
    dry_run: bool,
) -> dict:
    """Build the full experiment assignment structure."""
    seed_hex = secrets.token_hex(16)
    injection_group, control_group = shuffle_and_split(sessions, seed_hex)
    typed_assignments = assign_failure_types(injection_group, seed_hex)

    conn = None
    engine = None
    if not dry_run and db_url:
        from sqlalchemy import create_engine

        engine = create_engine(db_url)
        conn = engine.connect()

    injection_entries = []
    for session, failure_type in typed_assignments:
        is_safety_chunk = (
            failure_type == "chunk_deletion" and session.get("safety_expected", False)
        )

        target_config: dict | None = None
        if conn is not None:
            target_config = TARGET_QUERY_FNS[failure_type](conn, session, is_safety_chunk)

        if target_config is None and not dry_run:
            print(
                f"WARNING: No DB target found for {session['session_id']} "
                f"({failure_type}). Using placeholder."
            )

        injection_entries.append({
            "session_id": session["session_id"],
            "source_query_id": session["source_query_id"],
            "response_id": session["response_id"],
            "thread_id": session["thread_id"],
            "group": "injection",
            "failure_type": failure_type,
            "safety_expected": session.get("safety_expected", False),
            "target_config": target_config or {"placeholder": True},
        })

    control_entries = [
        {
            "session_id": s["session_id"],
            "source_query_id": s["source_query_id"],
            "response_id": s["response_id"],
            "thread_id": s["thread_id"],
            "group": "control",
            "failure_type": None,
            "safety_expected": s.get("safety_expected", False),
            "target_config": None,
        }
        for s in control_group
    ]

    if conn is not None:
        conn.close()
    if engine is not None:
        engine.dispose()

    return {
        "protocol_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed_hex": seed_hex,
        "dry_run": dry_run,
        "sessions": injection_entries + control_entries,
    }


def write_outputs(assignments: dict, out_dir: Path) -> None:
    """Write all experiment artifacts to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = out_dir / "injection_configs"
    configs_dir.mkdir(exist_ok=True)

    # 1. assignments.json — injection designer only
    assignments_path = out_dir / "assignments.json"
    assignments_json = json.dumps(assignments, indent=2, default=str)
    assignments_path.write_text(assignments_json)
    print(f"  Written: {assignments_path}")

    # 2. manifest_sha256.txt — tamper evidence
    sha256 = hashlib.sha256(assignments_json.encode()).hexdigest()
    manifest_path = out_dir / "manifest_sha256.txt"
    manifest_path.write_text(sha256 + "\n")
    print(f"  Written: {manifest_path}")
    print(f"  SHA-256: {sha256}")

    # 3. session_order.json — blinded evaluator only (no group labels)
    all_sessions = assignments["sessions"]
    # Shuffle session order for the blinded evaluator (use a fresh random to avoid correlation)
    import random

    session_order = [
        {"session_id": s["session_id"], "response_id": s["response_id"], "thread_id": s["thread_id"]}
        for s in all_sessions
    ]
    random.shuffle(session_order)
    session_order_path = out_dir / "session_order.json"
    session_order_path.write_text(json.dumps(session_order, indent=2))
    print(f"  Written: {session_order_path}")

    # 4. Per-session injection configs
    for entry in all_sessions:
        if entry["group"] != "injection":
            continue
        config_path = configs_dir / f"{entry['session_id']}.json"
        config_data = {
            "session_id": entry["session_id"],
            "failure_type": entry["failure_type"],
            "response_id": entry["response_id"],
            "thread_id": entry["thread_id"],
            "target_config": entry["target_config"],
        }
        config_path.write_text(json.dumps(config_data, indent=2, default=str))
    print(f"  Written: {len([e for e in all_sessions if e['group'] == 'injection'])} injection configs")

    # 5. diagnostic_trace_log_template.json — blinded evaluator
    template = {
        "evaluator": "blinded evaluator",
        "manifest_hash": sha256,
        "evaluation_date": "",
        "sessions": [
            {
                "session_id": s["session_id"],
                "response_id": s["response_id"],
                "thread_id": s["thread_id"],
                "time_started": "",
                "time_completed": "",
                "time_to_diagnosis_seconds": None,
                "failure_detected": None,
                "confidence": "",
                "failure_type_guess": "",
                "root_cause_description": "",
                "metrics_examined": [],
                "anomalies_found": [],
                "reasoning_trace": [
                    {
                        "step": 1,
                        "metric": "",
                        "expected": "",
                        "observed": "",
                        "inference": "",
                    }
                ],
            }
            for s in session_order
        ],
    }
    template_path = out_dir / "diagnostic_trace_log_template.json"
    template_path.write_text(json.dumps(template, indent=2))
    print(f"  Written: {template_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomize failure injection assignments (injection designer)"
    )
    parser.add_argument(
        "--sessions", type=Path, required=True,
        help="Path to failure_injection_sessions.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory for experiment artifacts",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="DATABASE_URL for target lookup (or set DATABASE_URL env var)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip DB queries; use placeholder targets",
    )
    args = parser.parse_args()

    import os

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url and not args.dry_run:
        print("ERROR: --db-url or DATABASE_URL required (or use --dry-run)")
        sys.exit(1)

    with open(args.sessions) as f:
        sessions = json.load(f)
    print(f"Loaded {len(sessions)} sessions")

    if len(sessions) != 12:
        print(f"ERROR: Expected 12 sessions, got {len(sessions)}")
        sys.exit(1)

    print("Building assignments...")
    assignments = build_assignments(sessions, db_url, args.dry_run)

    injection_count = sum(1 for s in assignments["sessions"] if s["group"] == "injection")
    control_count = sum(1 for s in assignments["sessions"] if s["group"] == "control")
    types_used = {s["failure_type"] for s in assignments["sessions"] if s["failure_type"]}
    print(f"  Injection: {injection_count}, Control: {control_count}")
    print(f"  Failure types used: {sorted(types_used)}")

    safety_chunk = any(
        s["failure_type"] == "chunk_deletion" and s["safety_expected"]
        for s in assignments["sessions"]
    )
    print(f"  Safety chunk_deletion: {'YES' if safety_chunk else 'NO (constraint not met)'}")

    print("\nWriting artifacts...")
    write_outputs(assignments, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
