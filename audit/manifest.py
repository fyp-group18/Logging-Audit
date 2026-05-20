"""Hash-chain manifest for ingestion audit trail (L1-E08).

Each ingestion event (document registered, chunk created, embedding generated)
is appended to the ingestion_manifest table with a SHA-256 hash that chains it
to the previous entry. This makes the log tamper-evident — any deletion or
modification breaks the chain.

Usage:
    from audit.manifest import append_manifest_entry, verify_manifest_integrity

    append_manifest_entry("doc_registered", {"document_id": 42, ...}, document_id=42)
"""

import hashlib
import json
import logging

from sqlalchemy import text

from core.database import SessionLocal, with_db_retry

logger = logging.getLogger(__name__)

# Genesis hash — the "previous_hash" for the very first entry
_GENESIS_HASH = "0" * 64


@with_db_retry
def append_manifest_entry(
    entry_type: str,
    entry_data: dict,
    document_id: int | None = None,
) -> int:
    """Append a new entry to the ingestion manifest hash-chain.

    Returns the entry_sequence number of the new entry.
    """
    with SessionLocal() as db:
        # Fetch the last entry to chain from
        row = db.execute(
            text(
                "SELECT entry_sequence, entry_hash FROM ingestion_manifest "
                "ORDER BY entry_sequence DESC LIMIT 1 FOR UPDATE"
            )
        ).first()

        if row is None:
            prev_seq = 0
            prev_hash = _GENESIS_HASH
        else:
            prev_seq = row[0]
            prev_hash = row[1]

        new_seq = prev_seq + 1
        data_json = json.dumps(entry_data, default=str, sort_keys=True)
        # Hash = SHA-256(previous_hash + entry_data_json)
        entry_hash = hashlib.sha256((prev_hash + data_json).encode()).hexdigest()

        db.execute(
            text(
                "INSERT INTO ingestion_manifest "
                "(entry_sequence, entry_hash, previous_hash, entry_type, entry_data, document_id) "
                "VALUES (:seq, :hash, :prev, :etype, CAST(:edata AS jsonb), :doc_id)"
            ),
            {
                "seq": new_seq,
                "hash": entry_hash,
                "prev": prev_hash,
                "etype": entry_type,
                "edata": data_json,
                "doc_id": document_id,
            },
        )
        db.commit()
        return new_seq


@with_db_retry
def verify_manifest_integrity(document_id: int | None = None) -> dict:
    """Verify the hash-chain integrity of the ingestion manifest.

    Args:
        document_id: If provided, verify only entries for this document.
                     Otherwise verify the entire chain.

    Returns:
        {"valid": bool, "entries_checked": int, "first_broken_at": int|None}
    """
    with SessionLocal() as db:
        query = "SELECT entry_sequence, entry_hash, previous_hash, entry_data FROM ingestion_manifest"
        params: dict = {}
        if document_id is not None:
            query += " WHERE document_id = :doc_id"
            params["doc_id"] = document_id
        query += " ORDER BY entry_sequence ASC"

        rows = db.execute(text(query), params).fetchall()

    if not rows:
        return {"valid": True, "entries_checked": 0, "first_broken_at": None}

    # When filtering by document_id, entries are non-contiguous in the
    # global sequence (other documents' entries sit between them), so
    # cross-entry chain linkage is only valid for unfiltered queries.
    check_chain_linkage = document_id is None

    for i, row in enumerate(rows):
        seq, stored_hash, stored_prev, entry_data = row
        data_json = json.dumps(
            entry_data if isinstance(entry_data, dict) else json.loads(entry_data),
            default=str,
            sort_keys=True,
        )
        expected_hash = hashlib.sha256((stored_prev + data_json).encode()).hexdigest()

        if expected_hash != stored_hash:
            logger.warning(
                f"Manifest integrity violation at entry_sequence={seq}: "
                f"expected={expected_hash[:16]}..., stored={stored_hash[:16]}..."
            )
            return {
                "valid": False,
                "entries_checked": i + 1,
                "first_broken_at": seq,
            }

        # Verify chain linkage (prev_hash of entry N+1 == entry_hash of entry N)
        if check_chain_linkage and i > 0:
            prev_row_hash = rows[i - 1][1]
            if stored_prev != prev_row_hash:
                logger.warning(
                    f"Manifest chain break at entry_sequence={seq}: "
                    f"prev_hash doesn't match prior entry_hash"
                )
                return {
                    "valid": False,
                    "entries_checked": i + 1,
                    "first_broken_at": seq,
                }

    return {"valid": True, "entries_checked": len(rows), "first_broken_at": None}
