"""Backfill ingestion_manifest entries for documents that were ingested before
the manifest system was added.

Reconstructs the three entry types per document from the current DB state:
  1. doc_registered   — from documents_multimodal
  2. chunk_created    — from document_chunks_multimodal aggregate stats
  3. embedding_generated — from document_chunks_multimodal embedding counts

Usage:
    DATABASE_URL=<set-via-env> \
        python schemas/data/backfill_manifest.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from collections import Counter
from sqlalchemy import text
from core.database import SessionLocal, with_db_retry
from audit.manifest import append_manifest_entry

DOCUMENT_IDS = [87, 88]
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 3072
CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200


@with_db_retry
def get_document_metadata(doc_id: int) -> dict:
    with SessionLocal() as db:
        row = db.execute(
            text(
                "SELECT id, original_filename, file_hash, doc_format "
                "FROM documents_multimodal WHERE id = :doc_id"
            ),
            {"doc_id": doc_id},
        ).first()
        if not row:
            raise ValueError(f"Document {doc_id} not found")
        return {
            "id": row[0],
            "filename": row[1],
            "file_hash": row[2],
            "format": row[3],
        }


@with_db_retry
def get_chunk_stats(doc_id: int) -> dict:
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT level, count(*) FROM document_chunks_multimodal "
                "WHERE document_id = :doc_id GROUP BY level ORDER BY level"
            ),
            {"doc_id": doc_id},
        ).fetchall()
        per_level = {row[0]: row[1] for row in rows}
        total = sum(per_level.values())

        embed_count = db.execute(
            text(
                "SELECT count(*) FROM document_chunks_multimodal "
                "WHERE document_id = :doc_id AND embedding IS NOT NULL"
            ),
            {"doc_id": doc_id},
        ).scalar()

        return {
            "total_chunks": total,
            "per_level": per_level,
            "leaf_count": per_level.get(0, 0),
            "embed_count": embed_count,
        }


@with_db_retry
def manifest_already_populated(doc_id: int) -> bool:
    with SessionLocal() as db:
        count = db.execute(
            text(
                "SELECT count(*) FROM ingestion_manifest WHERE document_id = :doc_id"
            ),
            {"doc_id": doc_id},
        ).scalar()
        return count > 0


def backfill_document(doc_id: int) -> None:
    if manifest_already_populated(doc_id):
        print(f"  Document {doc_id}: manifest entries already exist — skipping")
        return

    doc = get_document_metadata(doc_id)
    stats = get_chunk_stats(doc_id)

    # 1. doc_registered
    seq1 = append_manifest_entry(
        "doc_registered",
        {
            "document_id": doc["id"],
            "filename": doc["filename"],
            "file_hash": doc["file_hash"],
            "format": doc["format"],
        },
        document_id=doc_id,
    )
    print(f"  [{doc_id}] doc_registered → entry_sequence={seq1}")

    # 2. chunk_created
    seq2 = append_manifest_entry(
        "chunk_created",
        {
            "document_id": doc["id"],
            "total_chunks": stats["total_chunks"],
            "per_level": stats["per_level"],
            "leaf_count": stats["leaf_count"],
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
        document_id=doc_id,
    )
    print(f"  [{doc_id}] chunk_created → entry_sequence={seq2}")

    # 3. embedding_generated
    seq3 = append_manifest_entry(
        "embedding_generated",
        {
            "document_id": doc["id"],
            "embedded_count": stats["embed_count"],
            "embedding_dimensions": EMBEDDING_DIM,
            "model": EMBEDDING_MODEL,
        },
        document_id=doc_id,
    )
    print(f"  [{doc_id}] embedding_generated → entry_sequence={seq3}")


def main():
    print("Backfilling ingestion manifest for documents:", DOCUMENT_IDS)
    for doc_id in DOCUMENT_IDS:
        backfill_document(doc_id)

    # Verify
    from audit.manifest import verify_manifest_integrity

    result = verify_manifest_integrity()
    print(f"\nManifest integrity check: {result}")


if __name__ == "__main__":
    main()
