"""
Failure injection and revert functions for the four mutation types.

Each type has an inject_* function (returns backup dict) and a revert_* function
(restores original state). All use raw SQL via sqlalchemy.text() to avoid
import dependency on the main project.

Usage:
    from eval.scripts.apply_injection import inject_chunk_deletion, revert_chunk_deletion
"""

from __future__ import annotations

from sqlalchemy import create_engine, text


def _engine(db_url: str):
    return create_engine(db_url)


# ---------------------------------------------------------------------------
# chunk_deletion
# ---------------------------------------------------------------------------

def inject_chunk_deletion(db_url: str, config: dict) -> dict:
    """Delete a row from response_chunk_link. Returns backup for revert."""
    target_id = config["target_id"]
    engine = _engine(db_url)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, response_id, chunk_id, step_index, grounding_label, "
                "similarity_score, reranker_score "
                "FROM response_chunk_link WHERE id = :id"
            ),
            {"id": target_id},
        ).mappings().one()

        backup = dict(row)

        conn.execute(
            text("DELETE FROM response_chunk_link WHERE id = :id"),
            {"id": target_id},
        )

    return {"type": "chunk_deletion", "backup_row": backup}


def revert_chunk_deletion(db_url: str, backup: dict) -> None:
    """Re-insert the deleted response_chunk_link row."""
    row = backup["backup_row"]
    engine = _engine(db_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO response_chunk_link "
                "(id, response_id, chunk_id, step_index, grounding_label, "
                "similarity_score, reranker_score) "
                "VALUES (:id, :response_id, :chunk_id, :step_index, "
                ":grounding_label, :similarity_score, :reranker_score)"
            ),
            row,
        )


# ---------------------------------------------------------------------------
# score_perturbation
# ---------------------------------------------------------------------------

def inject_score_perturbation(db_url: str, config: dict) -> dict:
    """Flip sync_faithfulness in evaluation_metrics (value → 1.0 - value)."""
    response_id = config["response_id"]
    engine = _engine(db_url)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, sync_faithfulness FROM evaluation_metrics "
                "WHERE response_id = :rid ORDER BY id LIMIT 1"
            ),
            {"rid": response_id},
        ).mappings().one()

        original = row["sync_faithfulness"]
        flipped = round(1.0 - (original or 0.0), 6)

        conn.execute(
            text(
                "UPDATE evaluation_metrics SET sync_faithfulness = :val "
                "WHERE id = :id"
            ),
            {"val": flipped, "id": row["id"]},
        )

    return {
        "type": "score_perturbation",
        "eval_metrics_id": row["id"],
        "original_sync_faithfulness": original,
    }


def revert_score_perturbation(db_url: str, backup: dict) -> None:
    """Restore original sync_faithfulness value."""
    engine = _engine(db_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE evaluation_metrics SET sync_faithfulness = :val "
                "WHERE id = :id"
            ),
            {"val": backup["original_sync_faithfulness"], "id": backup["eval_metrics_id"]},
        )


# ---------------------------------------------------------------------------
# log_gap
# ---------------------------------------------------------------------------

def inject_log_gap(db_url: str, config: dict) -> dict:
    """Delete a node from agent_execution_logs (middle of trace)."""
    target_id = config["target_id"]
    engine = _engine(db_url)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, trace_id, thread_id, document_id, response_id, "
                "operation_type, node_name, model, input_tokens, output_tokens, "
                "latency_ms, started_at, completed_at, error, metadata, "
                "created_at, record_hash, chain_id "
                "FROM agent_execution_logs WHERE id = :id"
            ),
            {"id": target_id},
        ).mappings().one()

        backup = {k: v for k, v in row.items()}
        # Serialize timestamps and jsonb for safe round-trip
        for key in ("started_at", "completed_at", "created_at"):
            if backup[key] is not None:
                backup[key] = backup[key].isoformat() if hasattr(backup[key], "isoformat") else str(backup[key])
        if backup["metadata"] is not None:
            import json
            backup["metadata"] = json.dumps(backup["metadata"]) if isinstance(backup["metadata"], dict) else backup["metadata"]

        conn.execute(
            text("DELETE FROM agent_execution_logs WHERE id = :id"),
            {"id": target_id},
        )

    return {"type": "log_gap", "backup_row": backup}


def revert_log_gap(db_url: str, backup: dict) -> None:
    """Re-insert the deleted agent_execution_logs row."""
    row = backup["backup_row"]
    engine = _engine(db_url)

    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO agent_execution_logs "
                "(id, trace_id, thread_id, document_id, response_id, "
                "operation_type, node_name, model, input_tokens, output_tokens, "
                "latency_ms, started_at, completed_at, error, metadata, "
                "created_at, record_hash, chain_id) "
                "VALUES (:id, :trace_id, :thread_id, :document_id, :response_id, "
                ":operation_type, :node_name, :model, :input_tokens, :output_tokens, "
                ":latency_ms, :started_at, :completed_at, :error, "
                "CAST(:metadata AS jsonb), :created_at, :record_hash, :chain_id)"
            ),
            row,
        )


# ---------------------------------------------------------------------------
# manifest_tamper
# ---------------------------------------------------------------------------

def inject_manifest_tamper(db_url: str, config: dict) -> dict:
    """Alter entry_hash in ingestion_manifest (bypassing immutability trigger)."""
    target_id = config["target_id"]
    engine = _engine(db_url)

    with engine.begin() as conn:
        row = conn.execute(
            text(
                "SELECT id, entry_hash FROM ingestion_manifest WHERE id = :id"
            ),
            {"id": target_id},
        ).mappings().one()

        original_hash = row["entry_hash"]
        tampered_hash = "TAMPERED_" + original_hash[:55]

        # Disable immutability trigger, mutate, re-enable
        conn.execute(text(
            "ALTER TABLE ingestion_manifest DISABLE TRIGGER trg_manifest_immutable"
        ))
        conn.execute(
            text(
                "UPDATE ingestion_manifest SET entry_hash = :hash WHERE id = :id"
            ),
            {"hash": tampered_hash, "id": target_id},
        )
        conn.execute(text(
            "ALTER TABLE ingestion_manifest ENABLE TRIGGER trg_manifest_immutable"
        ))

    return {
        "type": "manifest_tamper",
        "manifest_id": target_id,
        "original_hash": original_hash,
    }


def revert_manifest_tamper(db_url: str, backup: dict) -> None:
    """Restore original entry_hash in ingestion_manifest."""
    engine = _engine(db_url)

    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE ingestion_manifest DISABLE TRIGGER trg_manifest_immutable"
        ))
        conn.execute(
            text(
                "UPDATE ingestion_manifest SET entry_hash = :hash WHERE id = :id"
            ),
            {"hash": backup["original_hash"], "id": backup["manifest_id"]},
        )
        conn.execute(text(
            "ALTER TABLE ingestion_manifest ENABLE TRIGGER trg_manifest_immutable"
        ))


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------

INJECT_FNS = {
    "chunk_deletion": inject_chunk_deletion,
    "score_perturbation": inject_score_perturbation,
    "log_gap": inject_log_gap,
    "manifest_tamper": inject_manifest_tamper,
}

REVERT_FNS = {
    "chunk_deletion": revert_chunk_deletion,
    "score_perturbation": revert_score_perturbation,
    "log_gap": revert_log_gap,
    "manifest_tamper": revert_manifest_tamper,
}


def inject(db_url: str, failure_type: str, config: dict) -> dict:
    """Dispatch to the appropriate inject function."""
    return INJECT_FNS[failure_type](db_url, config)


def revert(db_url: str, failure_type: str, backup: dict) -> None:
    """Dispatch to the appropriate revert function."""
    REVERT_FNS[failure_type](db_url, backup)
