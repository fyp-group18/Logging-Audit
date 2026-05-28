"""Layer 1 computed metrics for document chunk corpus audit.

Covers:
  - Leaf coverage by depth level
  - Chunk orphan rate (unreachable leaves via parent_id)
  - Summary drift index (child-parent embedding divergence)
  - Manifest hash-chain integrity
  - Safety tag statistics
"""

import logging

from sqlalchemy import text

from audit.manifest import verify_manifest_integrity
from core.database import SessionLocal, with_db_retry

logger = logging.getLogger(__name__)


@with_db_retry
def compute_leaf_coverage(document_id: int) -> dict:
    """Reachability-based leaf coverage at depth *d* (the document's max RAPTOR level).

    Leaf Coverage@d = |leaves reachable from depth-d summaries| / |total leaves|

    Starting from every node at the highest summary level, recursively follows
    ``source_chunk_ids`` downward until level-0 leaves are reached.  Leaves
    that are not reachable from *any* top-level summary are "orphaned" from the
    hierarchical tree and reduce coverage below 1.0.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT id, level, source_chunk_ids "
                "FROM document_chunks_multimodal "
                "WHERE document_id = :doc_id"
            ),
            {"doc_id": document_id},
        ).fetchall()

    if not rows:
        return {
            "metric_name": "leaf_coverage",
            "value": None,
            "metadata": {"document_id": document_id, "levels": {}, "total_chunks": 0},
        }

    chunks_by_id: dict[str, dict] = {}
    leaves: set[str] = set()
    max_level = 0
    level_counts: dict[int, int] = {}

    for cid, level, source_ids in rows:
        lvl = level or 0
        chunks_by_id[cid] = {"level": lvl, "source_chunk_ids": source_ids or []}
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
        if lvl == 0:
            leaves.add(cid)
        if lvl > max_level:
            max_level = lvl

    total = len(rows)
    leaf_count = len(leaves)

    if max_level == 0:
        # Flat corpus (no summaries) — every chunk is a leaf, coverage = 1.0
        return {
            "metric_name": "leaf_coverage",
            "value": 1.0,
            "metadata": {
                "document_id": document_id,
                "levels": level_counts,
                "leaf_count": leaf_count,
                "total_chunks": total,
                "max_depth": 0,
                "reachable_leaves": leaf_count,
            },
        }

    # DFS from top-level nodes following source_chunk_ids down to leaves
    top_nodes = [cid for cid, info in chunks_by_id.items() if info["level"] == max_level]

    reachable_leaves: set[str] = set()
    visited: set[str] = set()
    stack = list(top_nodes)

    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)

        info = chunks_by_id.get(node_id)
        if not info:
            continue

        if info["level"] == 0:
            reachable_leaves.add(node_id)
        else:
            for child_id in info["source_chunk_ids"]:
                if child_id not in visited:
                    stack.append(child_id)

    coverage = len(reachable_leaves) / leaf_count if leaf_count > 0 else 0.0

    return {
        "metric_name": "leaf_coverage",
        "value": round(coverage, 4),
        "metadata": {
            "document_id": document_id,
            "levels": level_counts,
            "leaf_count": leaf_count,
            "total_chunks": total,
            "max_depth": max_level,
            "reachable_leaves": len(reachable_leaves),
        },
    }


@with_db_retry
def compute_chunk_orphan_rate(document_id: int) -> dict:
    """Fraction of non-root chunks whose parent_id does not resolve to a valid chunk.

    Traverses one hop of the parent_id FK within the document. Chunks with a
    non-null parent_id that points to a chunk outside this document (or that no
    longer exists) are classified as orphans.
    """
    with SessionLocal() as db:
        # Count chunks that have a parent_id but whose parent does not exist
        # within the same document.
        result = db.execute(
            text(
                """
                SELECT
                    COUNT(*) FILTER (WHERE c.parent_id IS NOT NULL) AS non_root,
                    COUNT(*) FILTER (
                        WHERE c.parent_id IS NOT NULL
                          AND NOT EXISTS (
                              SELECT 1 FROM document_chunks_multimodal p
                              WHERE p.id = c.parent_id
                                AND p.document_id = c.document_id
                          )
                    ) AS orphan_count
                FROM document_chunks_multimodal c
                WHERE c.document_id = :doc_id
                """
            ),
            {"doc_id": document_id},
        ).first()

    non_root = result[0] if result else 0
    orphan_count = result[1] if result else 0
    rate = orphan_count / non_root if non_root > 0 else 0.0

    return {
        "metric_name": "chunk_orphan_rate",
        "value": round(rate, 4),
        "metadata": {
            "document_id": document_id,
            "non_root_chunks": non_root,
            "orphan_count": orphan_count,
        },
    }


@with_db_retry
def compute_summary_drift_index(document_id: int) -> dict:
    """Mean cosine distance between each child chunk and its parent summary.

    A high drift (> 0.3) indicates that RAPTOR summaries diverge significantly
    from their source chunks, which may degrade hierarchical retrieval quality.
    Uses pgvector's ``<=>`` cosine distance operator directly in SQL.
    """
    with SessionLocal() as db:
        row = db.execute(
            text(
                """
                SELECT
                    AVG(c.embedding <=> p.embedding) AS mean_distance,
                    COUNT(*) AS pair_count
                FROM document_chunks_multimodal c
                JOIN document_chunks_multimodal p
                    ON p.id = c.parent_id AND p.document_id = c.document_id
                WHERE c.document_id = :doc_id
                  AND c.parent_id IS NOT NULL
                  AND c.embedding IS NOT NULL
                  AND p.embedding IS NOT NULL
                """
            ),
            {"doc_id": document_id},
        ).first()

    mean_distance = float(row[0]) if row and row[0] is not None else None
    pair_count = row[1] if row else 0

    return {
        "metric_name": "summary_drift_index",
        "value": round(mean_distance, 4) if mean_distance is not None else None,
        "metadata": {
            "document_id": document_id,
            "parent_child_pairs": pair_count,
            "interpretation": (
                "cosine distance; 0=identical, 1=orthogonal; target < 0.3"
            ),
        },
    }


@with_db_retry
def compute_manifest_integrity(document_id: int | None = None) -> dict:
    """Verify hash-chain continuity of the ingestion manifest.

    Delegates to audit.manifest.verify_manifest_integrity and wraps the
    result in the standard metric envelope.
    """
    integrity = verify_manifest_integrity(document_id=document_id)

    return {
        "metric_name": "manifest_integrity",
        "value": 1.0 if integrity["valid"] else 0.0,
        "metadata": {
            "document_id": document_id,
            "valid": integrity["valid"],
            "entries_checked": integrity["entries_checked"],
            "first_broken_at": integrity["first_broken_at"],
        },
    }


@with_db_retry
def compute_kb_reconstructability(document_id: int, as_of_timestamp: str | None = None) -> dict:
    """Jaccard similarity between the current KB chunk set and the set reconstructable
    from the ingestion manifest at a given timestamp.

    If as_of_timestamp is None, uses the most recent manifest entry timestamp minus
    1 hour as the reconstruction target (testing whether near-current state can be
    reconstructed from the audit trail).

    KB Reconstructability@t = |KB_t ∩ KB_current| / |KB_t ∪ KB_current|
    """
    from datetime import timedelta

    with SessionLocal() as db:
        # Determine reconstruction timestamp
        if as_of_timestamp:
            target_ts = as_of_timestamp
        else:
            latest = db.execute(
                text(
                    "SELECT MAX(entry_timestamp) FROM ingestion_manifest "
                    "WHERE document_id = :doc_id"
                ),
                {"doc_id": document_id},
            ).scalar()
            if latest is None:
                return {
                    "metric_name": "kb_reconstructability",
                    "value": None,
                    "metadata": {
                        "document_id": document_id,
                        "error": "no manifest entries found",
                    },
                }
            # Use the latest chunk_created timestamp for this doc if available,
            # otherwise fall back to MAX - 1hr.  When all manifest entries share
            # the same second (bulk ingestion), the 1-hr offset pushes the target
            # before the chunk_created entry, causing a false miss.
            chunk_ts = db.execute(
                text(
                    "SELECT MAX(entry_timestamp) FROM ingestion_manifest "
                    "WHERE document_id = :doc_id AND entry_type = 'chunk_created'"
                ),
                {"doc_id": document_id},
            ).scalar()
            target_ts = (chunk_ts or (latest - timedelta(hours=1))).isoformat()

        # Get current chunk IDs for this document
        current_rows = db.execute(
            text(
                "SELECT id FROM document_chunks_multimodal "
                "WHERE document_id = :doc_id"
            ),
            {"doc_id": document_id},
        ).fetchall()
        current_ids: set[str] = {str(r[0]) for r in current_rows}

        # Reconstruct chunk set from manifest: all chunk_created entries up to target_ts
        # Each chunk_created manifest entry stores total chunks at that point.
        # We use the most recent chunk_created entry at or before target_ts.
        manifest_row = db.execute(
            text(
                """
                SELECT entry_data
                FROM ingestion_manifest
                WHERE document_id = :doc_id
                  AND entry_type = 'chunk_created'
                  AND entry_timestamp <= CAST(:target_ts AS timestamptz)
                ORDER BY entry_sequence DESC
                LIMIT 1
                """
            ),
            {"doc_id": document_id, "target_ts": target_ts},
        ).first()

        if not manifest_row or not manifest_row[0]:
            return {
                "metric_name": "kb_reconstructability",
                "value": None,
                "metadata": {
                    "document_id": document_id,
                    "target_timestamp": target_ts,
                    "error": "no chunk_created manifest entry found at target time",
                },
            }

        # The manifest entry_data contains chunk count info. Since individual
        # chunk IDs are not stored in manifest entries, we approximate by comparing
        # chunk counts. If counts match, Jaccard = 1.0. If not, we bound Jaccard.
        entry_data = manifest_row[0]
        manifest_chunk_count = entry_data.get("total_chunks", 0)
        current_count = len(current_ids)

        if manifest_chunk_count == 0 and current_count == 0:
            jaccard = 1.0
        elif manifest_chunk_count == 0 or current_count == 0:
            jaccard = 0.0
        else:
            # Approximate Jaccard: min/max of counts (upper bound when individual
            # IDs are not tracked in manifest)
            intersection_upper = min(manifest_chunk_count, current_count)
            union_lower = max(manifest_chunk_count, current_count)
            jaccard = intersection_upper / union_lower

    return {
        "metric_name": "kb_reconstructability",
        "value": round(jaccard, 4),
        "value_type": "upper_bound",
        "metadata": {
            "document_id": document_id,
            "target_timestamp": target_ts,
            "manifest_chunk_count": manifest_chunk_count,
            "current_chunk_count": current_count,
            "approximation": "count-based (individual chunk IDs not stored in manifest)",
        },
    }


@with_db_retry
def compute_kb_reconstructability_per_chunk(
    document_id: int, retrieved_chunk_ids: set[str]
) -> dict:
    """Per-chunk provenance check for retrieved chunks from a specific document.

    For each retrieved chunk belonging to this document, verifies that the chunk
    still exists in document_chunks_multimodal and that the ingestion manifest
    hash chain for this document is intact.  Returns the fraction of retrieved
    chunks with a complete provenance trail.
    """
    if not retrieved_chunk_ids:
        return {
            "metric_name": "kb_reconstructability_per_chunk",
            "value": None,
            "metadata": {
                "document_id": document_id,
                "chunks_with_provenance": 0,
                "chunks_total_retrieved": 0,
            },
        }

    # Verify manifest integrity once for the whole document
    integrity = verify_manifest_integrity(document_id=document_id)
    manifest_intact = integrity["valid"]

    with SessionLocal() as db:
        # Check which of the retrieved chunk IDs actually exist for this document
        existing = db.execute(
            text(
                "SELECT id FROM document_chunks_multimodal "
                "WHERE document_id = :doc_id AND id = ANY(:cids)"
            ),
            {"doc_id": document_id, "cids": list(retrieved_chunk_ids)},
        ).fetchall()
        existing_ids = {str(r[0]) for r in existing}

    total = len(retrieved_chunk_ids)

    if manifest_intact:
        # All existing chunks have valid provenance when chain is intact
        with_provenance = len(existing_ids)
    else:
        # Manifest broken — chunks that still exist get partial credit only if
        # the break point is after the chunk_created entry for this document.
        # Conservative: treat all as lacking provenance when chain is broken.
        with_provenance = 0

    score = with_provenance / total if total > 0 else 1.0

    return {
        "metric_name": "kb_reconstructability_per_chunk",
        "value": round(score, 4),
        "metadata": {
            "document_id": document_id,
            "chunks_with_provenance": with_provenance,
            "chunks_total_retrieved": total,
            "manifest_intact": manifest_intact,
            "first_broken_at": integrity.get("first_broken_at"),
        },
    }


@with_db_retry
def get_ingestion_metadata(document_id: int) -> dict:
    """Extract ingestion metadata from manifest entries for a document.

    Reads the chunk_created and embedding_generated manifest entries to return
    chunk_size, chunk_overlap, clustering_stats, embedding model, and timestamp.
    """
    with SessionLocal() as db:
        chunk_entry = db.execute(
            text(
                "SELECT entry_data, entry_timestamp "
                "FROM ingestion_manifest "
                "WHERE document_id = :doc_id AND entry_type = 'chunk_created' "
                "ORDER BY entry_sequence DESC LIMIT 1"
            ),
            {"doc_id": document_id},
        ).first()

        embed_entry = db.execute(
            text(
                "SELECT entry_data "
                "FROM ingestion_manifest "
                "WHERE document_id = :doc_id AND entry_type = 'embedding_generated' "
                "ORDER BY entry_sequence DESC LIMIT 1"
            ),
            {"doc_id": document_id},
        ).first()

    result: dict = {
        "document_id": document_id,
        "ingestion_timestamp": None,
        "embedding_model": None,
        "chunk_size": None,
        "chunk_overlap": None,
        "clusters_per_level": [],
    }

    if chunk_entry:
        data = chunk_entry[0] or {}
        ts = chunk_entry[1]
        result["ingestion_timestamp"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts) if ts else None
        result["chunk_size"] = data.get("chunk_size")
        result["chunk_overlap"] = data.get("chunk_overlap")
        clustering = data.get("clustering_stats", [])
        result["clusters_per_level"] = [
            {"level": s.get("level", i + 1), "count": s.get("actual_clusters", 0)}
            for i, s in enumerate(clustering)
        ]

    if embed_entry:
        data = embed_entry[0] or {}
        result["embedding_model"] = data.get("model")

    return result


@with_db_retry
def compute_safety_tag_stats(document_id: int) -> dict:
    """Distribution of has_safety_content tags and safety_signal_words density.

    Reports the fraction of chunks tagged with safety content and the mean
    number of signal words per tagged chunk.
    """
    with SessionLocal() as db:
        row = db.execute(
            text(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE has_safety_content = TRUE) AS safety_count,
                    SUM(
                        CASE
                            WHEN has_safety_content = TRUE
                                AND safety_signal_words IS NOT NULL
                            THEN jsonb_array_length(safety_signal_words)
                            ELSE 0
                        END
                    ) AS total_signal_words
                FROM document_chunks_multimodal
                WHERE document_id = :doc_id
                """
            ),
            {"doc_id": document_id},
        ).first()

    total = row[0] if row else 0
    safety_count = row[1] if row else 0
    total_signal_words = row[2] if row else 0

    tag_rate = safety_count / total if total > 0 else 0.0
    mean_signals = total_signal_words / safety_count if safety_count > 0 else 0.0

    return {
        "metric_name": "safety_tag_stats",
        "value": round(tag_rate, 4),
        "metadata": {
            "document_id": document_id,
            "total_chunks": total,
            "safety_tagged_count": safety_count,
            "safety_tag_rate": round(tag_rate, 4),
            "mean_signal_words_per_tagged_chunk": round(mean_signals, 2),
            "total_signal_words": total_signal_words,
        },
    }
