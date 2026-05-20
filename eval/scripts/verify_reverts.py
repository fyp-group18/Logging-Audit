"""
Verify that all failure injection mutations have been reverted in the DB.

Reads backups.json and checks that each backed-up row/value has been restored.

Usage:
    cd backend && uv run python -m eval.scripts.verify_reverts \
        --backups eval/results/failure_injection/experiment_run_001/backups.json \
        --db-url "$DATABASE_URL"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text


def check_chunk_deletion(conn, backup: dict) -> tuple[bool, str]:
    """Verify deleted response_chunk_link row was re-inserted."""
    row_id = backup["backup_row"]["id"]
    result = conn.execute(
        text("SELECT id, response_id, chunk_id FROM response_chunk_link WHERE id = :id"),
        {"id": row_id},
    ).mappings().fetchone()

    if result is None:
        return False, f"response_chunk_link id={row_id} NOT FOUND — row still deleted"

    expected_chunk = backup["backup_row"]["chunk_id"]
    if result["chunk_id"] != expected_chunk:
        return False, (
            f"response_chunk_link id={row_id} exists but chunk_id mismatch: "
            f"got {result['chunk_id']}, expected {expected_chunk}"
        )

    return True, f"response_chunk_link id={row_id} restored (chunk_id={expected_chunk[:12]}…)"


def check_score_perturbation(conn, backup: dict) -> tuple[bool, str]:
    """Verify sync_faithfulness was restored to original value."""
    eid = backup["eval_metrics_id"]
    original = backup["original_sync_faithfulness"]

    result = conn.execute(
        text("SELECT sync_faithfulness FROM evaluation_metrics WHERE id = :id"),
        {"id": eid},
    ).mappings().fetchone()

    if result is None:
        return False, f"evaluation_metrics id={eid} NOT FOUND"

    actual = result["sync_faithfulness"]
    if actual != original:
        return False, (
            f"evaluation_metrics id={eid}: sync_faithfulness={actual}, "
            f"expected={original}"
        )

    return True, f"evaluation_metrics id={eid}: sync_faithfulness={actual} (correct)"


def check_log_gap(conn, backup: dict) -> tuple[bool, str]:
    """Verify deleted agent_execution_logs row was re-inserted."""
    row_id = backup["backup_row"]["id"]
    result = conn.execute(
        text("SELECT id, node_name, record_hash FROM agent_execution_logs WHERE id = :id"),
        {"id": row_id},
    ).mappings().fetchone()

    if result is None:
        return False, f"agent_execution_logs id={row_id} NOT FOUND — row still deleted"

    expected_node = backup["backup_row"]["node_name"]
    if result["node_name"] != expected_node:
        return False, (
            f"agent_execution_logs id={row_id}: node_name={result['node_name']}, "
            f"expected={expected_node}"
        )

    return True, f"agent_execution_logs id={row_id} restored (node={expected_node})"


def check_manifest_tamper(conn, backup: dict) -> tuple[bool, str]:
    """Verify entry_hash was restored to original (non-tampered) value."""
    mid = backup["manifest_id"]
    original_hash = backup["original_hash"]

    result = conn.execute(
        text("SELECT entry_hash FROM ingestion_manifest WHERE id = :id"),
        {"id": mid},
    ).mappings().fetchone()

    if result is None:
        return False, f"ingestion_manifest id={mid} NOT FOUND"

    actual = result["entry_hash"]
    if actual.startswith("TAMPERED_"):
        return False, f"ingestion_manifest id={mid}: hash still tampered ({actual[:20]}…)"

    if actual != original_hash:
        return False, (
            f"ingestion_manifest id={mid}: hash={actual[:16]}…, "
            f"expected={original_hash[:16]}…"
        )

    return True, f"ingestion_manifest id={mid}: hash restored ({actual[:16]}…)"


CHECKERS = {
    "chunk_deletion": check_chunk_deletion,
    "score_perturbation": check_score_perturbation,
    "log_gap": check_log_gap,
    "manifest_tamper": check_manifest_tamper,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify failure injection reverts against the live DB"
    )
    parser.add_argument(
        "--backups", type=Path, required=True,
        help="Path to backups.json",
    )
    parser.add_argument(
        "--db-url", type=str, default=None,
        help="DATABASE_URL (or set DATABASE_URL env var)",
    )
    args = parser.parse_args()

    db_url = args.db_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: --db-url or DATABASE_URL required")
        sys.exit(1)

    with open(args.backups) as f:
        backups = json.load(f)

    engine = create_engine(db_url)
    print(f"Verifying {len(backups)} reverts...\n")

    passed = 0
    failed = 0

    with engine.connect() as conn:
        for backup in backups:
            sid = backup["session_id"]
            ftype = backup["failure_type"]
            checker = CHECKERS.get(ftype)

            if checker is None:
                print(f"  SKIP {sid} ({ftype}): no checker for this type")
                continue

            ok, detail = checker(conn, backup)
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {sid} ({ftype}): {detail}")

            if ok:
                passed += 1
            else:
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(backups)}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
