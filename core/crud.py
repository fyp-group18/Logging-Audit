# backend/core/crud.py
import logging
import time
from collections import Counter

import numpy as np
from sqlalchemy import and_, func, or_, select
from pgvector.sqlalchemy import HALFVEC

from modules.mmr import mmr_rerank

from core.config import (
    AUDIT_LOGGING_ENABLED,
    RETRIEVAL_MODE,
    SEARCH_MMR_CANDIDATES,
    SEARCH_MMR_ENABLED,
    SEARCH_MMR_LAMBDA,
    SEARCH_VECTOR_CANDIDATES,
)
from core.database import SessionLocal, with_db_retry
from core.models import (
    AgentExecutionLog,
    AntiExample,
    ChunkFeedbackCache,
    DeterministicRule,
    DiagnosticReport,
    DeviceModel,
    DocumentStatus,
    EquipmentCategory,
    ErrorCategory,
    EscalationFlag,
    EvaluationConfig,
    EvaluationConfigHistory,
    EvaluationMetric,
    FeedbackType,
    Notification,
    NotificationKind,
    PreferencePair,
    PromptAmendment,
    ReviewTaskPriority,
    ReviewTaskSource,
    ReviewTaskStatus,
    ResponseChunkLink,
    ResponseFeedback,
    ReviewStatus,
    ReviewVerdict,
    SourceReport,
    StepFeedback,
    ThreadTerminalState,
    UnlearningAuditLog,
    VerdictEnum,
    User,
    DiagnosticThread,
    OAuthRefreshToken,
    DocumentMultimodal,
    DocumentChunkMultimodal,
)
from core.storage import storage_client
from uuid import UUID
from sqlalchemy.orm import Session, selectinload
from core import models, security
from api import schemas
from typing import Optional, List

import time as _time
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


@with_db_retry
def get_all_documents(doc_type: str = None):
    with SessionLocal() as db:
        query = db.query(DocumentMultimodal).order_by(DocumentMultimodal.id.desc())
        if doc_type:
            query = query.filter(DocumentMultimodal.doc_type == doc_type)
        docs = query.all()
        return [
            (
                d.intuitive_name,
                d.original_filename,
                d.upload_date,
                d.filepath,
                d.doc_type,
                d.status.value,
                d.compatible_models or [],
                d.equipment_category.value if d.equipment_category else "GENERAL",
            )
            for d in docs
        ]


@with_db_retry
def update_document_name(filepath: str, new_name: str, doc_type: str):
    with SessionLocal() as db:
        doc = (
            db.query(DocumentMultimodal)
            .filter(DocumentMultimodal.filepath == filepath)
            .first()
        )
        if doc:
            doc.intuitive_name = new_name
            db.commit()


@with_db_retry
def delete_document(filepath: str, doc_type: str):
    with SessionLocal() as db:
        doc = (
            db.query(DocumentMultimodal)
            .filter(DocumentMultimodal.filepath == filepath)
            .first()
        )
        if doc:
            db.delete(doc)
            db.commit()
    try:
        storage_client.delete_file(filepath)
    except Exception as e:
        logger.warning(f"Could not delete physical file: {e}")


@with_db_retry
def get_reports_for_user(db: Session, user_id: Optional[UUID] = None) -> list:
    """Fetches PDF reports. If user_id is provided, filters by user. Otherwise returns all (for admins)."""

    # Use outerjoin for BOTH tables so we don't lose legacy reports
    query = (
        db.query(DiagnosticReport, User.email)
        .outerjoin(DiagnosticThread, DiagnosticReport.thread_id == DiagnosticThread.id)
        .outerjoin(User, DiagnosticThread.user_id == User.id)
        .order_by(DiagnosticReport.generated_at.desc())
    )

    if user_id:
        query = query.filter(DiagnosticThread.user_id == user_id)

    return query.all()


@with_db_retry
def semantic_search(
    query_emb,
    doc_type: str,
    device_id: str = None,
    equipment_category: str = None,
    limit: int = 10,
) -> tuple[list[dict], int]:
    with SessionLocal() as db:
        stmt = (
            select(DocumentChunkMultimodal, DocumentMultimodal)
            .join(DocumentMultimodal)
            .filter(DocumentMultimodal.doc_type == doc_type)
            .filter(DocumentMultimodal.status == DocumentStatus.READY)
        )

        if doc_type == "manual":
            if not device_id:
                raise ValueError("device_id required for manuals")
            # FIX: Properly handle empty arrays in PostgreSQL using cardinality
            stmt = stmt.filter(
                or_(
                    DocumentMultimodal.compatible_models.any(device_id),
                    func.cardinality(DocumentMultimodal.compatible_models) == 0,
                    DocumentMultimodal.compatible_models.is_(None),
                )
            )

        embedding_col = DocumentChunkMultimodal.embedding
        halfvec_col = embedding_col.cast(HALFVEC(3072))
        cosine_dist = halfvec_col.cosine_distance(query_emb).label("cosine_dist")
        stmt = stmt.filter(embedding_col.is_not(None))
        stmt = stmt.filter(DocumentChunkMultimodal.blacklisted.is_not(True))
        stmt = stmt.add_columns(cosine_dist)
        stmt = stmt.order_by(cosine_dist).limit(limit)

        t0 = time.time()
        results = db.execute(stmt).all()
        query_ms = int((time.time() - t0) * 1000)

        rows = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "intuitive_name": doc.intuitive_name,
                "doc_path": doc.filepath,
                "pages": chunk.pages,
                "level": str(chunk.level),
                "images": chunk.images,
                "score": 1.0 - float(dist),
                "page_start": chunk.page_start,
                "section_header": chunk.section_header,
            }
            for chunk, doc, dist in results
        ]
        return rows, query_ms


@with_db_retry
def check_duplicate_document(file_hash: str):
    with SessionLocal() as db:
        existing_doc = (
            db.query(DocumentMultimodal)
            .filter(
                DocumentMultimodal.file_hash == file_hash,
                DocumentMultimodal.status != DocumentStatus.FAILED,
            )
            .first()
        )
        return existing_doc.original_filename if existing_doc else None


@with_db_retry
def delete_failed_document_by_hash(file_hash: str) -> bool:
    """Delete any FAILED document with the given hash to allow re-upload."""
    with SessionLocal() as db:
        deleted = (
            db.query(DocumentMultimodal)
            .filter(
                DocumentMultimodal.file_hash == file_hash,
                DocumentMultimodal.status == DocumentStatus.FAILED,
            )
            .delete()
        )
        db.commit()
        return deleted > 0


@with_db_retry
def create_document_record(
    original_filename: str,
    intuitive_name: str,
    object_key: str,
    doc_type: str,
    file_hash: str,
) -> int:
    with SessionLocal() as db:
        new_doc = DocumentMultimodal(
            original_filename=original_filename,
            intuitive_name=intuitive_name,
            filepath=object_key,
            doc_type=doc_type,
            file_hash=file_hash,
            status=DocumentStatus.PENDING,
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        doc_id = new_doc.id

    # L1-E08: Append doc_registered manifest entry (separate transaction)
    try:
        from audit.manifest import append_manifest_entry

        append_manifest_entry(
            "doc_registered",
            {
                "document_id": doc_id,
                "filename": original_filename,
                "file_hash": file_hash,
                "format": doc_type,
            },
            document_id=doc_id,
        )
    except Exception:
        logger.warning(
            f"Failed to append doc_registered manifest entry for doc {doc_id}",
            exc_info=True,
        )

    return doc_id


@with_db_retry
def update_document_status(document_id: int, status: DocumentStatus):
    with SessionLocal() as db:
        doc = (
            db.query(DocumentMultimodal)
            .filter(DocumentMultimodal.id == document_id)
            .first()
        )
        if doc:
            doc.status = status
            db.commit()


@with_db_retry
def update_document_metadata(
    document_id: int,
    compatible_models: list[str] = None,
    equipment_category: str = None,
):
    with SessionLocal() as db:
        doc = (
            db.query(DocumentMultimodal)
            .filter(DocumentMultimodal.id == document_id)
            .first()
        )
        if not doc:
            raise ValueError(f"Document {document_id} not found")

        if compatible_models is not None:
            doc.compatible_models = compatible_models
            for model_name in compatible_models:
                if (
                    not db.query(DeviceModel)
                    .filter(DeviceModel.name == model_name)
                    .first()
                ):
                    db.add(DeviceModel(name=model_name))

        if equipment_category is not None:
            try:
                doc.equipment_category = EquipmentCategory[equipment_category.upper()]
            except KeyError:
                raise ValueError(f"Invalid category: {equipment_category}")
        db.commit()


@with_db_retry
def get_models_without_manuals():
    with SessionLocal() as db:
        all_models = {m[0] for m in db.query(DeviceModel.name).all()}
        docs_with_models = (
            db.query(DocumentMultimodal)
            .filter(
                DocumentMultimodal.doc_type == "manual",
                func.cardinality(DocumentMultimodal.compatible_models) > 0,
            )
            .all()
        )
        used_models = set()
        for doc in docs_with_models:
            if doc.compatible_models:
                used_models.update(doc.compatible_models)
        return sorted(list(all_models - used_models))


# --- Diagnostic Reports ---


@with_db_retry
def get_report_by_thread(thread_id: str) -> DiagnosticReport | None:
    with SessionLocal() as db:
        return (
            db.query(DiagnosticReport)
            .filter(DiagnosticReport.thread_id == thread_id)
            .first()
        )


@with_db_retry
def create_report_record(
    thread_id: str,
    device_id: str,
    object_key: str,
    file_size_bytes: int,
) -> DiagnosticReport:
    with SessionLocal() as db:
        report = DiagnosticReport(
            thread_id=thread_id,
            device_id=device_id,
            object_key=object_key,
            file_size_bytes=file_size_bytes,
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        return report


# Only truly generic words that never discriminate sections.
# Domain terms like "disassembly", "assembly", "part" are kept because
# they ARE actual section header content (e.g. "7-1. Disassembly",
# "8-3-7. Speaker Part").
_SECTION_FILTER_STOPWORDS = frozenset(
    {
        # English function words
        "this",
        "that",
        "with",
        "from",
        "have",
        "does",
        "been",
        "were",
        "will",
        "would",
        "could",
        "should",
        "they",
        "them",
        "their",
        "what",
        "when",
        "where",
        "which",
        "about",
        "into",
        "more",
        "some",
        "then",
        "than",
        "also",
        "just",
        "each",
        "only",
        # Meta terms that refer to the document itself, not content
        "manual",
        "section",
        "chapter",
        "page",
        "figure",
        "table",
        "note",
        "warning",
        "caution",
        "guide",
        "instructions",
        "procedure",
        "service",
        "maintenance",
    }
)

MAX_SECTION_CHUNKS = 100


@with_db_retry
def get_manual_walkthrough(
    device_id: str,
    search_terms: list[str] | None = None,
    limit: int = 80,
) -> list[dict]:
    """
    Return leaf chunks (level=0) from manuals compatible with the given
    device, ordered by reading sequence.

    When *search_terms* are provided the function first tries to narrow
    results to chunks whose ``section_header`` matches any significant
    word (>3 chars, not a stopword) from the terms.  If the filtered
    query returns nothing it falls back to the unfiltered walkthrough.
    """

    def _format(chunk, doc) -> dict:
        return {
            "id": str(chunk.id),
            "text": chunk.text,
            "intuitive_name": doc.intuitive_name,
            "doc_path": doc.filepath,
            "pages": chunk.pages,
            "images": chunk.images,
            "level": chunk.level,
            "sequence_index": chunk.sequence_index,
            "page_start": chunk.page_start,
            "section_header": chunk.section_header,
            "parent_id": chunk.parent_id,
            "section_code": chunk.section_code,
        }

    with SessionLocal() as db:
        # Base query without limit — limit is applied differently for
        # section-filtered vs. fallback paths (Issue 5).
        base = (
            select(DocumentChunkMultimodal, DocumentMultimodal)
            .join(DocumentMultimodal)
            .filter(DocumentMultimodal.doc_type == "manual")
            .filter(DocumentMultimodal.status == DocumentStatus.READY)
            .filter(DocumentChunkMultimodal.level == 0)
            .filter(DocumentChunkMultimodal.blacklisted.is_not(True))
            .filter(
                DocumentChunkMultimodal.parent_id.is_(None)
            )  # exclude table row children
            .filter(
                or_(
                    DocumentMultimodal.compatible_models.any(device_id),
                    func.cardinality(DocumentMultimodal.compatible_models) == 0,
                    DocumentMultimodal.compatible_models.is_(None),
                )
            )
            .order_by(
                DocumentMultimodal.id.asc(),
                DocumentChunkMultimodal.sequence_index.asc().nulls_last(),
                DocumentChunkMultimodal.page_start.asc().nulls_last(),
            )
        )

        # Try section-filtered query first
        if search_terms:
            words = {
                w.lower()
                for term in search_terms
                for w in term.split()
                if len(w) > 3 and w.lower() not in _SECTION_FILTER_STOPWORDS
            }
            if words:
                section_conditions = [
                    func.lower(DocumentChunkMultimodal.section_header).contains(w)
                    for w in words
                ]
                filtered = base.filter(or_(*section_conditions)).limit(200)
                filtered_results = db.execute(filtered).all()
                if filtered_results:
                    # Cross-document dedup: when one document contributes
                    # >70% of matched chunks, it's the actual procedure
                    # manual — drop interleaved chunks from other docs.
                    doc_counts = Counter(d.id for _, d in filtered_results)
                    if len(doc_counts) > 1:
                        top_doc_id, top_count = doc_counts.most_common(1)[0]
                        total_before_dedup = len(filtered_results)
                        if top_count / total_before_dedup > 0.7:
                            filtered_results = [
                                (c, d)
                                for c, d in filtered_results
                                if d.id == top_doc_id
                            ]
                            logger.info(
                                f"[Walkthrough] Deduped to dominant doc_id={top_doc_id} "
                                f"({top_count}/{total_before_dedup} chunks)"
                            )

                    logger.info(
                        f"[Walkthrough] Section filter matched {len(filtered_results)} "
                        f"chunks (terms={list(words)[:5]})"
                    )
                    if len(filtered_results) > MAX_SECTION_CHUNKS:
                        # Partition into image-bearing and text-only chunks so
                        # images survive the truncation cutoff.
                        with_images = [
                            (c, d) for c, d in filtered_results if c.images
                        ]
                        without_images = [
                            (c, d) for c, d in filtered_results if not c.images
                        ]

                        # Reserve up to 30% of budget for image chunks
                        image_budget = min(
                            len(with_images), MAX_SECTION_CHUNKS * 3 // 10
                        )
                        text_budget = MAX_SECTION_CHUNKS - image_budget

                        kept = without_images[:text_budget] + with_images[:image_budget]
                        # Back-fill if either pool was smaller than its budget
                        if len(kept) < MAX_SECTION_CHUNKS:
                            kept_set = set(kept)
                            remaining = [
                                p for p in filtered_results if p not in kept_set
                            ]
                            kept.extend(remaining[: MAX_SECTION_CHUNKS - len(kept)])
                        # Re-sort by original ordering (doc_id, sequence_index, page_start)
                        kept.sort(
                            key=lambda pair: (
                                pair[1].id,
                                pair[0].sequence_index
                                if pair[0].sequence_index is not None
                                else float("inf"),
                                pair[0].page_start
                                if pair[0].page_start is not None
                                else float("inf"),
                            )
                        )
                        logger.warning(
                            f"[Walkthrough] Section filter truncated: returning "
                            f"{len(kept)}/{len(filtered_results)} chunks "
                            f"({image_budget} image-bearing reserved) "
                            f"for device={device_id}."
                        )
                        filtered_results = kept
                    return [_format(c, d) for c, d in filtered_results]
                logger.info(
                    "[Walkthrough] Section filter matched 0 chunks, falling back to unfiltered"
                )

        # Fallback: return first `limit` chunks across all matching docs
        fallback = base.limit(limit)
        results = db.execute(fallback).all()

        # Warn once if the limit truncated available chunks.
        # The extra COUNT query only fires when results hit the exact
        # limit boundary, which is rare for most devices.
        if len(results) == limit:
            total = db.scalar(select(func.count()).select_from(base.subquery()))
            if total and total > limit:
                logger.warning(
                    f"[Walkthrough] Fallback truncated: returned {limit}/{total} "
                    f"chunks for device={device_id}. Consider raising the limit."
                )

        return [_format(c, d) for c, d in results]


@with_db_retry
def multimodal_semantic_search(
    query_text_emb: list,
    doc_type: str = "manual",
    device_id: str = None,
    document_id: int | None = None,
    retrieval_mode: str = RETRIEVAL_MODE,
    safety_only: bool = False,
) -> tuple[list[dict], int]:
    with SessionLocal() as db:
        base_query = (
            select(DocumentChunkMultimodal, DocumentMultimodal)
            .join(DocumentMultimodal)
            .filter(DocumentMultimodal.doc_type == doc_type)
            .filter(DocumentMultimodal.status == DocumentStatus.READY)
        )

        if doc_type == "manual" and device_id:
            base_query = base_query.filter(
                or_(
                    DocumentMultimodal.compatible_models.any(device_id),
                    func.cardinality(DocumentMultimodal.compatible_models) == 0,
                    DocumentMultimodal.compatible_models.is_(None),
                )
            )

        if document_id is not None:
            base_query = base_query.filter(
                DocumentChunkMultimodal.document_id == document_id
            )

        embedding_col = DocumentChunkMultimodal.embedding
        halfvec_col = embedding_col.cast(HALFVEC(3072))
        cosine_dist = halfvec_col.cosine_distance(query_text_emb).label("cosine_dist")
        filtered = base_query.filter(embedding_col.is_not(None)).filter(
            DocumentChunkMultimodal.blacklisted.is_not(True)
        )
        if safety_only:
            filtered = filtered.filter(
                DocumentChunkMultimodal.has_safety_content.is_(True)
            )
        if retrieval_mode == "flat":
            filtered = filtered.filter(DocumentChunkMultimodal.level == 0)

        use_mmr = SEARCH_MMR_ENABLED and SEARCH_MMR_CANDIDATES > SEARCH_VECTOR_CANDIDATES
        if use_mmr:
            stmt = (
                filtered
                .add_columns(cosine_dist, embedding_col)
                .order_by(cosine_dist)
                .limit(SEARCH_MMR_CANDIDATES)
            )
        else:
            stmt = (
                filtered
                .add_columns(cosine_dist)
                .order_by(cosine_dist)
                .limit(SEARCH_VECTOR_CANDIDATES)
            )

        t0 = time.time()
        results = db.execute(stmt).all()
        query_ms = int((time.time() - t0) * 1000)

        if use_mmr and len(results) > SEARCH_VECTOR_CANDIDATES:
            embeddings = np.array([row[-1] for row in results], dtype=np.float32)
            scores = [1.0 - float(row[-2]) for row in results]
            query_np = np.asarray(query_text_emb, dtype=np.float32)

            t1 = time.time()
            selected = mmr_rerank(
                query_np, embeddings, scores,
                k=SEARCH_VECTOR_CANDIDATES,
                lambda_=SEARCH_MMR_LAMBDA,
            )
            mmr_ms = int((time.time() - t1) * 1000)
            logger.debug("MMR rerank selected %d/%d in %dms", len(selected), len(results), mmr_ms)

            # Extract chunk, doc, dist (drop embedding column)
            results = [(results[i][0], results[i][1], results[i][2]) for i in selected]
        elif use_mmr:
            # Fewer candidates than SEARCH_VECTOR_CANDIDATES — strip embedding col
            results = [(row[0], row[1], row[2]) for row in results]

        rows = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "intuitive_name": doc.intuitive_name,
                "doc_path": doc.filepath,
                "pages": chunk.pages,
                "images": chunk.images,
                "level": chunk.level,
                "sequence_index": chunk.sequence_index,
                "page_start": chunk.page_start,
                "section_header": chunk.section_header,
                "parent_id": chunk.parent_id,
                "section_code": chunk.section_code,
                "has_safety_content": chunk.has_safety_content,
                "score": 1.0 - float(dist),
            }
            for chunk, doc, dist in results
        ]
        return rows, query_ms


@with_db_retry
def fetch_parent_chunks(parent_ids: set[str]) -> dict[str, dict]:
    """Fetch parent chunks by IDs for parent-child text swap.

    Returns {parent_id: {"text": ..., "pages": ..., ...}} for each found parent.
    """
    if not parent_ids:
        return {}
    with SessionLocal() as db:
        results = (
            db.query(DocumentChunkMultimodal)
            .filter(DocumentChunkMultimodal.id.in_(parent_ids))
            .filter(DocumentChunkMultimodal.blacklisted.is_not(True))
            .all()
        )
        return {
            chunk.id: {
                "text": chunk.text,
                "pages": chunk.pages,
                "images": chunk.images,
                "section_header": chunk.section_header,
                "page_start": chunk.page_start,
                "sequence_index": chunk.sequence_index,
            }
            for chunk in results
        }


@with_db_retry
def get_chunks_by_sequence_indices(
    document_id: int,
    sequence_indices: set[int],
) -> list[dict]:
    """Fetch level-0 chunks at specific sequence positions within a document.

    Used by scoped procedural retrieval to expand ±1 neighbor chunks around
    semantically retrieved results, providing prerequisite context."""
    if not sequence_indices:
        return []
    with SessionLocal() as db:
        stmt = (
            select(DocumentChunkMultimodal, DocumentMultimodal)
            .join(DocumentMultimodal)
            .filter(DocumentChunkMultimodal.document_id == document_id)
            .filter(DocumentChunkMultimodal.level == 0)
            .filter(DocumentChunkMultimodal.sequence_index.in_(sequence_indices))
            .order_by(
                DocumentChunkMultimodal.sequence_index.asc().nulls_last(),
                DocumentChunkMultimodal.page_start.asc().nulls_last(),
            )
        )
        results = db.execute(stmt).all()
        return [
            {
                "id": chunk.id,
                "text": chunk.text,
                "intuitive_name": doc.intuitive_name,
                "doc_path": doc.filepath,
                "pages": chunk.pages,
                "images": chunk.images,
                "level": chunk.level,
                "sequence_index": chunk.sequence_index,
                "page_start": chunk.page_start,
                "section_header": chunk.section_header,
                "parent_id": chunk.parent_id,
                "section_code": chunk.section_code,
            }
            for chunk, doc in results
        ]


@with_db_retry
def select_best_document(
    query_embedding: list[float],
    device_id: str,
) -> int | None:
    """Query level>=1 summary nodes to identify the most relevant document
    for the given query.  Returns the document_id with the best average
    cosine similarity, or None when no summary nodes exist (graceful
    fallback — callers must treat None as 'search all documents')."""
    with SessionLocal() as db:
        avg_dist = func.avg(
            DocumentChunkMultimodal.embedding.cast(HALFVEC(3072)).cosine_distance(
                query_embedding
            )
        ).label("avg_distance")
        stmt = (
            select(DocumentChunkMultimodal.document_id, avg_dist)
            .join(DocumentMultimodal)
            .filter(DocumentMultimodal.doc_type == "manual")
            .filter(DocumentMultimodal.status == DocumentStatus.READY)
            .filter(DocumentChunkMultimodal.level >= 1)
            .filter(DocumentChunkMultimodal.embedding.is_not(None))
            .filter(
                or_(
                    DocumentMultimodal.compatible_models.any(device_id),
                    func.cardinality(DocumentMultimodal.compatible_models) == 0,
                    DocumentMultimodal.compatible_models.is_(None),
                )
            )
            .group_by(DocumentChunkMultimodal.document_id)
            .order_by(avg_dist)
            .limit(1)
        )
        row = db.execute(stmt).first()
        if row is None:
            logger.info(f"[Document Selection] No summary nodes for device={device_id}")
            return None
        doc_id, avg_dist = row
        logger.info(
            f"[Document Selection] Selected doc_id={doc_id} "
            f"(avg_cosine_dist={avg_dist:.4f}) for device={device_id}"
        )
        return doc_id


@with_db_retry
def get_document_summaries(
    document_id: int,
    max_level: int = 2,
) -> list[dict]:
    """Fetch summary nodes (level 1..max_level) for a specific document,
    ordered broadest-first (level DESC) then by page position (page_start ASC).
    Returns an empty list when the document has no summaries."""
    with SessionLocal() as db:
        stmt = (
            select(DocumentChunkMultimodal)
            .filter(DocumentChunkMultimodal.document_id == document_id)
            .filter(DocumentChunkMultimodal.level >= 1)
            .filter(DocumentChunkMultimodal.level <= max_level)
            .order_by(
                DocumentChunkMultimodal.level.desc(),
                DocumentChunkMultimodal.page_start.asc().nulls_last(),
            )
        )
        chunks = db.execute(stmt).scalars().all()
        return [
            {
                "text": c.text,
                "section_header": c.section_header,
                "level": c.level,
                "pages": c.pages,
            }
            for c in chunks
        ]


# --- User CRUD ---
@with_db_retry
def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    """Fetches a user by their email address."""
    return db.query(models.User).filter(models.User.email == email).first()


@with_db_retry
def get_user_by_id(db: Session, user_id: UUID) -> Optional[models.User]:
    """Fetches a user by their ID."""
    return db.query(models.User).filter(models.User.id == user_id).first()


@with_db_retry
def create_db_user(db: Session, user: schemas.UserCreate) -> models.User:
    """Creates a new user in the database."""
    hashed_password = security.hash_password(user.password)

    # Per your request, default technician level to JUNIOR
    user_level = user.level
    if user.role == "TECHNICIAN" and not user_level:
        user_level = "JUNIOR"

    db_user = models.User(
        email=user.email,
        password_hash=hashed_password,
        role=user.role.upper(),
        level=user_level.upper() if user_level else None,
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user


@with_db_retry
def get_all_users(db: Session) -> list[type[User]]:
    return db.query(models.User).order_by(models.User.created_at.desc()).all()


@with_db_retry
def update_db_user(
    db: Session, user_id: UUID, user_data: schemas.UserUpdate
) -> Optional[models.User]:
    """Updates a user's email, role, and level."""
    user = get_user_by_id(db, user_id)
    if not user:
        return None

    user.email = user_data.email
    user.role = models.UserRole[user_data.role.upper()]

    if user.role == models.UserRole.TECHNICIAN:
        if user_data.level and user_data.level.upper() in ["JUNIOR", "SENIOR"]:
            user.level = models.TechnicianLevel[user_data.level.upper()]
        else:
            user.level = models.TechnicianLevel.JUNIOR  # Default fallback
    else:
        user.level = None

    db.commit()
    db.refresh(user)
    return user


@with_db_retry
def delete_db_user(db: Session, user_id: UUID) -> bool:
    """Deletes a user from the database."""
    user = get_user_by_id(db, user_id)
    if user:
        db.delete(user)
        db.commit()
        return True
    return False


@with_db_retry
def create_or_update_thread(db: Session, thread_id: str, user_id: Optional[UUID] = None, device_id: str = ""):
    """Registers a new diagnostic thread, or updates its timestamp."""
    thread = db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
    if thread:
        thread.device_id = device_id
        thread.updated_at = func.now()
    else:
        thread = DiagnosticThread(id=thread_id, user_id=user_id, device_id=device_id)
        db.add(thread)
    db.commit()


@with_db_retry
def get_threads_for_user(
    db: Session, user_id: Optional[UUID] = None
) -> List[DiagnosticThread]:
    """Fetches threads. If user_id is provided, filters by user. Otherwise returns all (for admins).

    DEPRECATED: Use get_threads() for filtered/paginated access with badge enrichment.
    Kept for backward compatibility with any callers not yet migrated.
    """
    query = (
        db.query(DiagnosticThread)
        .options(selectinload(DiagnosticThread.user))
        .order_by(DiagnosticThread.updated_at.desc())
    )
    if user_id:
        query = query.filter(DiagnosticThread.user_id == user_id)
    return query.all()


@with_db_retry
def get_threads(
    db: Session,
    user_id: Optional[UUID] = None,
    filters: Optional[dict] = None,
    page: int = 1,
    per_page: int = 20,
) -> tuple[list[tuple[DiagnosticThread, Optional[str]]], int]:
    """Fetch paginated thread list with latest badge from evaluation_metrics.

    Returns (rows, total) where each row is (DiagnosticThread, badge_or_None).
    """
    # Correlated subquery: latest badge per thread (reused for SELECT and WHERE)
    badge_subquery = (
        select(EvaluationMetric.badge)
        .where(EvaluationMetric.thread_id == DiagnosticThread.id)
        .order_by(EvaluationMetric.created_at.desc())
        .limit(1)
        .correlate(DiagnosticThread)
        .scalar_subquery()
    )

    # Base filter query (no badge column — used for counting)
    base_query = db.query(DiagnosticThread)

    # Role-based filter
    if user_id:
        base_query = base_query.filter(DiagnosticThread.user_id == user_id)

    # Apply filters
    if filters:
        if filters.get("badge"):
            base_query = base_query.filter(badge_subquery == filters["badge"])

        if filters.get("device_id"):
            base_query = base_query.filter(
                DiagnosticThread.device_id == filters["device_id"]
            )

        if filters.get("from_date"):
            base_query = base_query.filter(
                DiagnosticThread.created_at >= filters["from_date"]
            )

        if filters.get("to_date"):
            base_query = base_query.filter(
                DiagnosticThread.created_at <= filters["to_date"]
            )

        if filters.get("flagged") is not None:
            base_query = base_query.filter(
                DiagnosticThread.flagged_for_review == filters["flagged"]
            )

    # Sorting
    sort = (filters or {}).get("sort", "date_desc")
    if sort == "date_asc":
        base_query = base_query.order_by(DiagnosticThread.updated_at.asc())
    else:
        base_query = base_query.order_by(DiagnosticThread.updated_at.desc())

    # Count on the lightweight query (no badge column in SELECT)
    total = base_query.count()

    # Add badge column and eagerly load user for the paginated fetch only
    data_query = base_query.add_columns(badge_subquery.label("latest_badge")).options(
        selectinload(DiagnosticThread.user)
    )

    rows = data_query.offset((page - 1) * per_page).limit(per_page).all()

    return rows, total


@with_db_retry
def get_feedback_for_thread(db: Session, thread_id: str) -> dict[str, list]:
    """Get all response-level and step-level feedback for a thread.

    Both ResponseFeedback and StepFeedback have a direct thread_id FK.
    """
    response_feedbacks = (
        db.query(ResponseFeedback)
        .filter(
            ResponseFeedback.thread_id == thread_id,
            ResponseFeedback.retracted.is_(False),
        )
        .order_by(ResponseFeedback.created_at)
        .all()
    )
    step_feedbacks = (
        db.query(StepFeedback)
        .filter(
            StepFeedback.thread_id == thread_id,
            StepFeedback.retracted.is_(False),
        )
        .order_by(StepFeedback.created_at)
        .all()
    )
    return {
        "response_feedbacks": response_feedbacks,
        "step_feedbacks": step_feedbacks,
    }


@with_db_retry
def get_all_evaluation_rows(db: Session, thread_id: str) -> List[EvaluationMetric]:
    """All evaluation metrics rows for a thread (including retries), ordered by created_at."""
    return (
        db.query(EvaluationMetric)
        .filter(EvaluationMetric.thread_id == thread_id)
        .order_by(EvaluationMetric.created_at)
        .all()
    )


@with_db_retry
def get_latest_escalation_for_thread(
    db: Session, thread_id: str
) -> Optional[EscalationFlag]:
    """Latest EscalationFlag for a thread (by created_at DESC)."""
    return (
        db.query(EscalationFlag)
        .filter(EscalationFlag.thread_id == thread_id)
        .order_by(EscalationFlag.created_at.desc())
        .first()
    )


@with_db_retry
def get_thread_by_id(db: Session, thread_id: str) -> Optional[DiagnosticThread]:
    return db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()


@with_db_retry
def get_thread_device_id(thread_id: str) -> Optional[str]:
    """Return the device_id for a thread, or None if thread doesn't exist."""
    with SessionLocal() as db:
        row = (
            db.query(DiagnosticThread.device_id)
            .filter(DiagnosticThread.id == thread_id)
            .first()
        )
        return row[0] if row else None


@with_db_retry
def get_thread_terminal_state(thread_id: str) -> str:
    """Return the terminal_state for a thread; defaults to IN_PROGRESS when
    the row is absent or the enum is null (refresh-recovery uses this to
    decide whether to auto-rerun the diagnostic)."""
    with SessionLocal() as db:
        row = (
            db.query(DiagnosticThread.terminal_state)
            .filter(DiagnosticThread.id == thread_id)
            .first()
        )
        if not row or not row[0]:
            return "IN_PROGRESS"
        return row[0].value


@with_db_retry
def create_refresh_token(db: Session, user_id: UUID, token: str) -> OAuthRefreshToken:
    hashed_token = security.hash_token(token)
    expires_at = datetime.now(timezone.utc) + timedelta(
        days=security.REFRESH_TOKEN_EXPIRE_DAYS
    )

    db_token = OAuthRefreshToken(
        user_id=user_id, token_hash=hashed_token, expires_at=expires_at
    )
    db.add(db_token)
    db.commit()
    db.refresh(db_token)
    return db_token


@with_db_retry
def purge_expired_refresh_tokens(db: Session) -> int:
    """Delete all OAuthRefreshToken rows where expires_at is in the past.

    Returns the number of rows deleted.  Call this from a startup hook or
    a scheduled maintenance endpoint — no scheduler is registered here.
    """
    deleted = (
        db.query(OAuthRefreshToken)
        .filter(OAuthRefreshToken.expires_at < datetime.now(timezone.utc))
        .delete(synchronize_session=False)
    )
    db.commit()
    return deleted


@with_db_retry
def verify_and_delete_refresh_token(db: Session, token: str) -> Optional[models.User]:
    """
    Verifies a refresh token. If valid, deletes it (to prevent reuse)
    and returns the associated User. Returns None if invalid or expired.
    """
    hashed_token = security.hash_token(token)
    db_token = (
        db.query(OAuthRefreshToken)
        .filter(OAuthRefreshToken.token_hash == hashed_token)
        .first()
    )

    if not db_token:
        return None

    # Check expiration
    if db_token.expires_at < datetime.now(timezone.utc):
        db.delete(db_token)
        db.commit()
        return None

    user = db_token.user

    # Delete the token immediately (Refresh Token Rotation for high security)
    db.delete(db_token)
    db.commit()

    return user


# --- Evaluation Metrics CRUD ---


@with_db_retry
def insert_evaluation_metric(payload: dict) -> int:
    """Insert a sync-side evaluation row.

    `payload` keys:
        thread_id, response_id, device_id, user_id (UUID|None),
        badge, quality_badge, safety_badge,
        sync_faithfulness, sync_answer_relevance,
        sync_context_relevance, sync_completeness, sync_reasons (dict),
        sync_duration_ms, thresholds_snapshot (dict).
    """
    if not AUDIT_LOGGING_ENABLED:
        return -1
    with SessionLocal() as db:
        row = EvaluationMetric(
            thread_id=payload["thread_id"],
            response_id=payload["response_id"],
            device_id=payload["device_id"],
            user_id=payload.get("user_id"),
            badge=payload["badge"],
            quality_badge=payload.get("quality_badge"),
            safety_badge=payload.get("safety_badge"),
            attempt_number=payload.get("attempt_number", 1),
            sync_faithfulness=payload.get("sync_faithfulness"),
            sync_answer_relevance=payload.get("sync_answer_relevance"),
            sync_context_relevance=payload.get("sync_context_relevance"),
            sync_completeness=payload.get("sync_completeness"),
            sync_reasons=payload.get("sync_reasons") or {},
            sync_duration_ms=payload.get("sync_duration_ms"),
            thresholds_snapshot=payload["thresholds_snapshot"],
            step_verdicts=payload.get("step_verdicts"),
            safety_survival=payload.get("safety_survival"),
            safety_confidence=payload.get("safety_confidence"),
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id


@with_db_retry
def set_thread_flag(thread_id: str, flagged: bool) -> bool:
    """Toggle the admin-review flag on a diagnostic thread.

    Returns True on update, False if the row was missing. Idempotent — writing
    the same value is a no-op at the DB level.
    """
    with SessionLocal() as db:
        row = (
            db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
        )
        if not row:
            return False
        row.flagged_for_review = flagged
        db.commit()
        return True


@with_db_retry
def get_evaluations_for_thread(thread_id: str) -> List[dict]:
    """Return one EvaluationMetric dict per response_id for the given thread.

    When a response_id has multiple rows (attempt 1 + retry), the highest-id
    row wins — that's the row the live UI surfaced as the final verdict.
    Returned as plain dicts so callers can use the data after the session
    closes.
    """
    with SessionLocal() as db:
        latest_ids = (
            db.query(func.max(EvaluationMetric.id).label("max_id"))
            .filter(EvaluationMetric.thread_id == thread_id)
            .group_by(EvaluationMetric.response_id)
            .subquery()
        )
        rows = (
            db.query(EvaluationMetric)
            .filter(EvaluationMetric.id.in_(select(latest_ids.c.max_id)))
            .order_by(EvaluationMetric.created_at.asc())
            .all()
        )
        return [
            {
                "response_id": r.response_id,
                "badge": r.badge,
                "quality_badge": r.quality_badge,
                "safety_badge": r.safety_badge,
                "attempt_number": r.attempt_number,
                "sync_faithfulness": r.sync_faithfulness,
                "sync_answer_relevance": r.sync_answer_relevance,
                "sync_reasons": r.sync_reasons or {},
                "sync_duration_ms": r.sync_duration_ms,
                "async_context_relevance": r.async_context_relevance,
                "async_completeness": r.async_completeness,
                "async_evidence": r.async_evidence,
                "async_duration_ms": r.async_duration_ms,
                "async_completed_at": (
                    r.async_completed_at.isoformat() if r.async_completed_at else None
                ),
                "step_verdicts": r.step_verdicts,
            }
            for r in rows
        ]


@with_db_retry
def update_evaluation_metric_async(response_id: str, async_payload: dict) -> bool:
    """Patch the async-side columns onto an existing row keyed by response_id.

    `async_payload` keys (all optional except completed_at):
        async_faithfulness, async_answer_relevance,
        async_context_relevance, async_completeness,
        async_evidence (dict|None), async_duration_ms,
        async_completed_at (datetime).
    """
    if not AUDIT_LOGGING_ENABLED:
        return False
    with SessionLocal() as db:
        row = (
            db.query(EvaluationMetric)
            .filter(EvaluationMetric.response_id == response_id)
            .order_by(EvaluationMetric.id.desc())
            .first()
        )
        if not row:
            return False
        for k in (
            "async_faithfulness",
            "async_answer_relevance",
            "async_context_relevance",
            "async_completeness",
            "async_safety_coverage",
            "async_evidence",
            "async_duration_ms",
            "async_completed_at",
            "badge",
            "badge_source",
            "shadow_step_verdicts",
        ):
            if k in async_payload:
                setattr(row, k, async_payload[k])
        db.commit()
        return True


@with_db_retry
def get_evaluation_metric_by_response_id(response_id: str):
    """Get the latest EvaluationMetric row for a response_id (ORM instance).

    Row is expunged from the session so attributes remain accessible after close.
    """
    with SessionLocal() as db:
        row = (
            db.query(EvaluationMetric)
            .filter(EvaluationMetric.response_id == response_id)
            .order_by(EvaluationMetric.id.desc())
            .first()
        )
        if row:
            db.expunge(row)
        return row


@with_db_retry
def record_override(response_id: str, reason: Optional[str]) -> bool:
    with SessionLocal() as db:
        row = (
            db.query(EvaluationMetric)
            .filter(EvaluationMetric.response_id == response_id)
            .order_by(EvaluationMetric.id.desc())
            .first()
        )
        if not row:
            return False
        row.override = True
        row.override_reason = reason
        db.commit()
        return True


# --- Execution Log CRUD ---


@with_db_retry
def bulk_insert_execution_logs(logs: list[dict], *, chain_id: str | None = None) -> int:
    """Batch INSERT execution log rows with hash-chain integrity.

    When *chain_id* is provided each row receives a sequential
    ``record_hash`` linking it to the previous record in the chain.
    Returns count inserted.
    """
    if not logs:
        return 0
    if not AUDIT_LOGGING_ENABLED:
        return 0

    from core.hash_chain import GENESIS_HASH, compute_record_hash

    with SessionLocal() as db:
        # Resolve the tail hash of this chain (if it already has records).
        prev_hash = GENESIS_HASH
        if chain_id:
            last_row = (
                db.query(AgentExecutionLog.record_hash)
                .filter(
                    AgentExecutionLog.chain_id == chain_id,
                    AgentExecutionLog.record_hash.isnot(None),
                )
                .order_by(AgentExecutionLog.id.desc())
                .first()
            )
            if last_row and last_row[0]:
                prev_hash = last_row[0]

        rows = []
        for entry in logs:
            _sa = entry.get("started_at")
            created_at_str = (
                _sa.isoformat() if hasattr(_sa, "isoformat") else str(_sa or "")
            )
            rec_hash = compute_record_hash(
                previous_hash=prev_hash,
                operation_type=entry["operation_type"],
                document_id=entry.get("document_id"),
                node_name=entry["node_name"],
                created_at=created_at_str,
                metadata=entry.get("metadata"),
            )
            rows.append(
                AgentExecutionLog(
                    trace_id=entry.get("trace_id"),
                    thread_id=entry.get("thread_id"),
                    document_id=entry.get("document_id"),
                    response_id=entry.get("response_id"),
                    operation_type=entry["operation_type"],
                    node_name=entry["node_name"],
                    model=entry.get("model"),
                    input_tokens=entry.get("input_tokens"),
                    output_tokens=entry.get("output_tokens"),
                    latency_ms=entry.get("latency_ms"),
                    started_at=entry.get("started_at"),
                    completed_at=entry.get("completed_at"),
                    error=entry.get("error"),
                    metadata_=entry.get("metadata"),
                    record_hash=rec_hash,
                    chain_id=chain_id,
                )
            )
            prev_hash = rec_hash

        db.add_all(rows)
        db.commit()
        return len(rows)


@with_db_retry
def get_execution_logs(thread_id: str) -> list[dict]:
    """All logs for a thread, ordered by started_at."""
    with SessionLocal() as db:
        rows = (
            db.query(AgentExecutionLog)
            .filter(AgentExecutionLog.thread_id == thread_id)
            .order_by(AgentExecutionLog.started_at)
            .all()
        )
        return [
            {
                "id": r.id,
                "trace_id": r.trace_id,
                "thread_id": r.thread_id,
                "document_id": r.document_id,
                "response_id": r.response_id,
                "operation_type": r.operation_type,
                "node_name": r.node_name,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "error": r.error,
                "metadata": r.metadata_,
                "created_at": r.created_at,
            }
            for r in rows
        ]


@with_db_retry
def get_execution_logs_by_trace(trace_id: str) -> list[dict]:
    """All logs for a trace, ordered by started_at."""
    with SessionLocal() as db:
        rows = (
            db.query(AgentExecutionLog)
            .filter(AgentExecutionLog.trace_id == trace_id)
            .order_by(AgentExecutionLog.started_at)
            .all()
        )
        return [
            {
                "id": r.id,
                "trace_id": r.trace_id,
                "thread_id": r.thread_id,
                "document_id": r.document_id,
                "response_id": r.response_id,
                "operation_type": r.operation_type,
                "node_name": r.node_name,
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
                "started_at": r.started_at,
                "completed_at": r.completed_at,
                "error": r.error,
                "metadata": r.metadata_,
                "created_at": r.created_at,
            }
            for r in rows
        ]


@with_db_retry
def verify_chain_integrity(chain_id: str) -> tuple[bool, list[int]]:
    """Walk a hash chain and verify every record's hash.

    Returns ``(is_valid, broken_record_ids)``.  An empty chain is valid.
    """
    from core.hash_chain import GENESIS_HASH, compute_record_hash

    with SessionLocal() as db:
        rows = (
            db.query(AgentExecutionLog)
            .filter(
                AgentExecutionLog.chain_id == chain_id,
                AgentExecutionLog.record_hash.isnot(None),
            )
            .order_by(AgentExecutionLog.id.asc())
            .all()
        )
        if not rows:
            return True, []

        broken: list[int] = []
        prev_hash = GENESIS_HASH
        for r in rows:
            expected = compute_record_hash(
                previous_hash=prev_hash,
                operation_type=r.operation_type,
                document_id=r.document_id,
                node_name=r.node_name,
                created_at=r.started_at.isoformat()
                if hasattr(r.started_at, "isoformat")
                else str(r.started_at or ""),
                metadata=r.metadata_,
            )
            if r.record_hash != expected:
                broken.append(r.id)
            prev_hash = expected  # walk recomputed chain to isolate tamper point
        return len(broken) == 0, broken


@with_db_retry
def get_execution_summary(thread_id: str) -> dict:
    """Aggregate: total LLM calls, total tokens, total latency, per-node breakdown."""
    with SessionLocal() as db:
        rows = (
            db.query(AgentExecutionLog)
            .filter(AgentExecutionLog.thread_id == thread_id)
            .all()
        )
        if not rows:
            return {
                "total_calls": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_latency_ms": 0,
                "by_node": {},
            }

        total_input = sum(r.input_tokens or 0 for r in rows)
        total_output = sum(r.output_tokens or 0 for r in rows)
        total_latency = sum(r.latency_ms or 0 for r in rows)
        llm_calls = [r for r in rows if r.operation_type == "llm_call"]

        by_node: dict[str, dict] = {}
        for r in rows:
            node = r.node_name
            if node not in by_node:
                by_node[node] = {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "latency_ms": 0,
                }
            by_node[node]["calls"] += 1
            by_node[node]["input_tokens"] += r.input_tokens or 0
            by_node[node]["output_tokens"] += r.output_tokens or 0
            by_node[node]["latency_ms"] += r.latency_ms or 0

        return {
            "total_calls": len(llm_calls),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_latency_ms": total_latency,
            "by_node": by_node,
        }


@with_db_retry
def get_evaluation_config() -> dict:
    """Returns the current thresholds dict + recommendations payload.

    Falls back to the seeded defaults if the row is missing (defensive — the
    migration always seeds, but a fresh dev DB without migration would 500).
    """
    with SessionLocal() as db:
        row = db.query(EvaluationConfig).filter(EvaluationConfig.id == 1).first()
        if not row:
            return {
                "thresholds": {
                    "green": {
                        "faithfulness": 0.50,
                        "answer_relevance": 0.30,
                        "context_relevance": 0.30,
                        "completeness": 0.40,
                    },
                    "red": {
                        "faithfulness": 0.30,
                        "answer_relevance": 0.15,
                        "context_relevance": 0.15,
                        "completeness": 0.20,
                    },
                },
                "recommendations": None,
                "updated_at": None,
            }
        return {
            "thresholds": row.thresholds,
            "recommendations": row.recommendations,
            "updated_at": row.updated_at,
        }


# Feedback CRUD (R1, R2, R7)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def create_response_feedback(
    user_id: UUID,
    response_id: str,
    thread_id: str,
    feedback_type_str: str,
    device_id: str,
    equipment_category: str,
    weight: float,
    correction_text: str | None = None,
    correction_category: str | None = None,
    satisfaction_rating: int | None = None,
    operator_role: str | None = None,
    time_on_feedback_ms: int | None = None,
) -> ResponseFeedback:
    """Upsert response feedback. Toggles off if same type resubmitted, switches if different."""
    fb_type = FeedbackType(feedback_type_str)
    eq_cat = EquipmentCategory(equipment_category)
    with SessionLocal() as db:
        existing = (
            db.query(ResponseFeedback)
            .filter(
                ResponseFeedback.response_id == response_id,
                ResponseFeedback.user_id == user_id,
            )
            .first()
        )
        if existing:
            if existing.feedback_type == fb_type:
                # Toggle off — snapshot fields before delete to avoid DetachedInstanceError
                snapshot = ResponseFeedback(
                    id=existing.id,
                    response_id=existing.response_id,
                    thread_id=existing.thread_id,
                    user_id=existing.user_id,
                    device_id=existing.device_id,
                    equipment_category=existing.equipment_category,
                    feedback_type=existing.feedback_type,
                    weight=existing.weight,
                    created_at=existing.created_at,
                )
                db.delete(existing)
                db.commit()
                return snapshot
            # Switch type
            existing.feedback_type = fb_type
            existing.correction_text = correction_text
            existing.correction_category = correction_category
            existing.satisfaction_rating = satisfaction_rating
            existing.weight = weight
            db.commit()
            db.refresh(existing)
            return existing

        fb = ResponseFeedback(
            response_id=response_id,
            thread_id=thread_id,
            user_id=user_id,
            device_id=device_id,
            equipment_category=eq_cat,
            feedback_type=fb_type,
            correction_text=correction_text,
            correction_category=correction_category,
            satisfaction_rating=satisfaction_rating,
            weight=weight,
            operator_role=operator_role,
            time_on_feedback_ms=time_on_feedback_ms,
        )
        db.add(fb)
        db.commit()
        db.refresh(fb)
        return fb


@with_db_retry
def create_step_feedback(
    user_id: UUID,
    response_id: str,
    thread_id: str,
    step_index: int,
    device_id: str,
    equipment_category: str,
    weight: float,
    action: str | None = None,
    correction_text: str | None = None,
    correction_type: str | None = None,
    severity: str | None = None,
    time_on_step_ms: int | None = None,
) -> StepFeedback | None:
    """Create or update step feedback.

    When ``action`` is provided, uses upsert semantics — inserts a new row or
    updates the existing one (never deletes).  When ``action`` is None, preserves
    legacy toggle behaviour for backward compatibility with the current frontend.
    """
    eq_cat = EquipmentCategory(equipment_category)
    with SessionLocal() as db:
        existing = (
            db.query(StepFeedback)
            .filter(
                StepFeedback.response_id == response_id,
                StepFeedback.step_index == step_index,
                StepFeedback.user_id == user_id,
            )
            .first()
        )

        if action is not None:
            # Upsert: insert or update with canonical action
            if existing:
                existing.action = action
                existing.correction_text = correction_text
                existing.weight = weight
                existing.correction_type = correction_type
                existing.severity = severity
                existing.time_on_step_ms = time_on_step_ms
                db.commit()
                db.refresh(existing)
                return existing
        else:
            # Legacy toggle: delete if exists
            if existing:
                db.delete(existing)
                db.commit()
                return None

        sf = StepFeedback(
            response_id=response_id,
            step_index=step_index,
            thread_id=thread_id,
            user_id=user_id,
            device_id=device_id,
            equipment_category=eq_cat,
            action=action,
            correction_text=correction_text,
            correction_type=correction_type,
            severity=severity,
            time_on_step_ms=time_on_step_ms,
            weight=weight,
        )
        db.add(sf)
        db.commit()
        db.refresh(sf)
        return sf


@with_db_retry
def create_source_report(
    user_id: UUID,
    chunk_id: str,
    response_id: str,
    thread_id: str,
    reason: str | None = None,
) -> SourceReport:
    with SessionLocal() as db:
        existing = (
            db.query(SourceReport)
            .filter(
                SourceReport.chunk_id == chunk_id,
                SourceReport.response_id == response_id,
                SourceReport.user_id == user_id,
            )
            .first()
        )
        if existing:
            return existing
        sr = SourceReport(
            chunk_id=chunk_id,
            response_id=response_id,
            thread_id=thread_id,
            user_id=user_id,
            reason=reason,
        )
        db.add(sr)
        db.commit()
        db.refresh(sr)
        return sr


@with_db_retry
def get_feedback_for_response(response_id: str, user_id: UUID) -> dict:
    """Return the current user's feedback state for a response."""
    with SessionLocal() as db:
        fb = (
            db.query(ResponseFeedback)
            .filter(
                ResponseFeedback.response_id == response_id,
                ResponseFeedback.user_id == user_id,
                ResponseFeedback.retracted.is_not(True),
            )
            .first()
        )
        step_rows = (
            db.query(
                StepFeedback.step_index,
                StepFeedback.action,
                StepFeedback.correction_text,
            )
            .filter(
                StepFeedback.response_id == response_id,
                StepFeedback.user_id == user_id,
                StepFeedback.retracted.is_not(True),
            )
            .all()
        )
        return {
            "feedback_type": fb.feedback_type.value if fb else None,
            "correction_text": fb.correction_text if fb else None,
            "correction_category": fb.correction_category if fb else None,
            "satisfaction_rating": fb.satisfaction_rating if fb else None,
            "flagged_steps": [
                {"step_index": row[0], "action": row[1], "correction_text": row[2]}
                for row in step_rows
            ],
        }


@with_db_retry
def save_response_chunk_links(
    response_id: str,
    chunk_ids: list[str],
    step_attributions: list[dict] | None = None,
    retrieval_scores: dict | None = None,
):
    """Store which chunks contributed to a response.

    When *step_attributions* is provided (list of StepVerdict dicts from
    InlineEvaluator), creates per-step enriched rows with grounding labels
    and retrieval scores.  Otherwise falls back to flat chunk_ids linking.
    """
    if not chunk_ids and not step_attributions:
        return
    if not AUDIT_LOGGING_ENABLED:
        return
    with SessionLocal() as db:
        if step_attributions:
            # Collect chunk_ids that will get step-attributed replacements
            attributed_cids: set[str] = set()
            for v in step_attributions:
                for cid in v.get("source_chunk_ids") or []:
                    attributed_cids.add(cid)

            # Delete bare rows ONLY for chunks that will get enriched
            # replacements — preserve bare links for unattributed chunks
            # so every retrieved chunk retains at least a bare row.
            if attributed_cids:
                db.query(ResponseChunkLink).filter(
                    ResponseChunkLink.response_id == response_id,
                    ResponseChunkLink.step_index.is_(None),
                    ResponseChunkLink.chunk_id.in_(attributed_cids),
                ).delete(synchronize_session="fetch")

            for verdict in step_attributions:
                step_idx = int(verdict["step_id"].replace("s-", ""))
                for cid in verdict.get("source_chunk_ids") or []:
                    cosine = None
                    reranker = None
                    if retrieval_scores:
                        scores = retrieval_scores.get("manual", {}).get(cid)
                        if scores is None:
                            scores = retrieval_scores.get("safety", {}).get(cid)
                        if isinstance(scores, dict):
                            cosine = scores.get("cosine")
                            reranker = scores.get("reranker")
                    db.add(
                        ResponseChunkLink(
                            response_id=response_id,
                            chunk_id=cid,
                            step_index=step_idx,
                            grounding_label=verdict.get("grounding_label"),
                            similarity_score=cosine,
                            reranker_score=reranker,
                        )
                    )
        else:
            for cid in chunk_ids:
                db.add(ResponseChunkLink(response_id=response_id, chunk_id=cid))
        db.commit()


@with_db_retry
def get_chunk_ids_for_response(response_id: str) -> list[str]:
    """Get unique chunk IDs that contributed to a response."""
    with SessionLocal() as db:
        rows = (
            db.query(ResponseChunkLink.chunk_id)
            .filter(ResponseChunkLink.response_id == response_id)
            .distinct()
            .all()
        )
        return [r[0] for r in rows]


@with_db_retry
def get_step_attribution(response_id: str) -> list[dict]:
    """Get per-step attribution data for a response, joined with source doc metadata."""
    with SessionLocal() as db:
        results = (
            db.query(
                ResponseChunkLink.step_index,
                ResponseChunkLink.chunk_id,
                ResponseChunkLink.grounding_label,
                ResponseChunkLink.similarity_score,
                ResponseChunkLink.reranker_score,
                DocumentChunkMultimodal.text,
                DocumentMultimodal.intuitive_name,
                DocumentChunkMultimodal.page_start,
                DocumentChunkMultimodal.level,
                DocumentMultimodal.doc_type,
            )
            .join(
                DocumentChunkMultimodal,
                ResponseChunkLink.chunk_id == DocumentChunkMultimodal.id,
            )
            .join(
                DocumentMultimodal,
                DocumentChunkMultimodal.document_id == DocumentMultimodal.id,
            )
            .filter(
                ResponseChunkLink.response_id == response_id,
                ResponseChunkLink.step_index.isnot(None),
            )
            .order_by(ResponseChunkLink.step_index)
            .all()
        )
        return [
            {
                "step_index": r.step_index,
                "chunk_id": r.chunk_id,
                "grounding_label": r.grounding_label,
                "similarity_score": r.similarity_score,
                "reranker_score": r.reranker_score,
                "chunk_text_preview": r.text[:200] if r.text else None,
                "doc_title": r.intuitive_name,
                "page_start": r.page_start,
                "level": r.level,
                "doc_type": r.doc_type,
            }
            for r in results
        ]


@with_db_retry
def get_all_retrieved_chunks(response_id: str) -> list[dict]:
    """Get ALL chunks linked to a response (attributed + unattributed)."""
    with SessionLocal() as db:
        results = (
            db.query(
                ResponseChunkLink.step_index,
                ResponseChunkLink.chunk_id,
                ResponseChunkLink.grounding_label,
                ResponseChunkLink.similarity_score,
                ResponseChunkLink.reranker_score,
                DocumentChunkMultimodal.text,
                DocumentMultimodal.intuitive_name,
                DocumentChunkMultimodal.page_start,
                DocumentChunkMultimodal.level,
                DocumentMultimodal.doc_type,
            )
            .join(
                DocumentChunkMultimodal,
                ResponseChunkLink.chunk_id == DocumentChunkMultimodal.id,
            )
            .join(
                DocumentMultimodal,
                DocumentChunkMultimodal.document_id == DocumentMultimodal.id,
            )
            .filter(ResponseChunkLink.response_id == response_id)
            .order_by(
                ResponseChunkLink.step_index.asc().nullslast(),
                ResponseChunkLink.similarity_score.desc().nullslast(),
            )
            .all()
        )
        return [
            {
                "step_index": r.step_index,
                "chunk_id": r.chunk_id,
                "grounding_label": r.grounding_label,
                "similarity_score": r.similarity_score,
                "reranker_score": r.reranker_score,
                "chunk_text_preview": r.text[:200] if r.text else None,
                "doc_title": r.intuitive_name,
                "page_start": r.page_start,
                "level": r.level,
                "doc_type": r.doc_type,
            }
            for r in results
        ]


# Grounding-aware feedback aggregation weights.
# Action polarity: how the technician's verdict on a step translates to a
# positive or negative signal for the underlying source chunk.
_ACTION_POLARITY: dict[str, float] = {
    "ACCEPT": 1.0,
    "INCORRECT": -1.0,
    "MODIFY": -0.5,
    "SKIP": 0.0,  # neutral — excluded from aggregation
}
# Grounding weight: how responsible a chunk is for a step's content.
_GROUNDING_WEIGHT: dict[str, float] = {
    "verbatim": 1.0,
    "paraphrased": 0.8,
    "synthesized": 0.5,
    "ungrounded": 0.1,
}
_GROUNDING_DEFAULT_WEIGHT = 0.5  # legacy rows without grounding_label


@with_db_retry
def recompute_chunk_feedback_cache(chunk_ids: list[str]):
    """Recompute feedback boost scores for the given chunks.

    Aggregates two signal sources:
    1. **ResponseFeedback** (plan-level thumbs up/down) — unchanged polarity.
    2. **StepFeedback** (per-step verdicts) — weighted by *action* polarity
       (ACCEPT=positive, INCORRECT=negative, MODIFY=partial negative,
       SKIP=excluded) and *grounding_label* from the response_chunk_link
       join (verbatim=1.0 … ungrounded=0.1).
    """
    if not chunk_ids:
        return
    with SessionLocal() as db:
        for chunk_id in chunk_ids:
            # Get all response_ids that used this chunk
            response_ids = (
                db.query(ResponseChunkLink.response_id)
                .filter(ResponseChunkLink.chunk_id == chunk_id)
                .all()
            )
            resp_id_list = [r[0] for r in response_ids]

            pos_weight = 0.0
            neg_weight = 0.0
            pos_count = 0
            neg_count = 0

            if resp_id_list:
                # --- Plan-level response feedback (thumbs up/down) ---
                pos_rows = (
                    db.query(
                        func.count(),
                        func.coalesce(func.sum(ResponseFeedback.weight), 0.0),
                    )
                    .filter(
                        ResponseFeedback.response_id.in_(resp_id_list),
                        ResponseFeedback.feedback_type == FeedbackType.THUMBS_UP,
                        ResponseFeedback.retracted.is_not(True),
                    )
                    .first()
                )
                pos_count, pos_weight = int(pos_rows[0]), float(pos_rows[1])

                neg_rows = (
                    db.query(
                        func.count(),
                        func.coalesce(func.sum(ResponseFeedback.weight), 0.0),
                    )
                    .filter(
                        ResponseFeedback.response_id.in_(resp_id_list),
                        ResponseFeedback.feedback_type == FeedbackType.THUMBS_DOWN,
                        ResponseFeedback.retracted.is_not(True),
                    )
                    .first()
                )
                neg_count, neg_weight = int(neg_rows[0]), float(neg_rows[1])

                # --- Grounding-aware step feedback aggregation ---
                # Join step_feedback → response_chunk_link to resolve
                # which chunks each step references + grounding quality.
                step_rows = (
                    db.query(
                        StepFeedback.action,
                        StepFeedback.weight,
                        ResponseChunkLink.grounding_label,
                    )
                    .join(
                        ResponseChunkLink,
                        and_(
                            ResponseChunkLink.response_id == StepFeedback.response_id,
                            ResponseChunkLink.step_index == StepFeedback.step_index,
                            ResponseChunkLink.chunk_id == chunk_id,
                        ),
                    )
                    .filter(
                        StepFeedback.response_id.in_(resp_id_list),
                        StepFeedback.retracted.is_not(True),
                    )
                    .all()
                )

                for action, weight, grounding_label in step_rows:
                    polarity = _ACTION_POLARITY.get(action, 0.0)
                    if polarity == 0.0:
                        continue
                    g_weight = _GROUNDING_WEIGHT.get(
                        grounding_label, _GROUNDING_DEFAULT_WEIGHT
                    )
                    signal = polarity * g_weight * (weight or 1.0)
                    if signal > 0:
                        pos_count += 1
                        pos_weight += signal
                    else:
                        neg_count += 1
                        neg_weight += abs(signal)

            net = pos_weight - neg_weight
            existing = (
                db.query(ChunkFeedbackCache)
                .filter(ChunkFeedbackCache.chunk_id == chunk_id)
                .first()
            )
            if existing:
                existing.positive_count = pos_count
                existing.negative_count = neg_count
                existing.weighted_positive = pos_weight
                existing.weighted_negative = neg_weight
                existing.net_score = net
                existing.last_recomputed = func.now()
            else:
                db.add(
                    ChunkFeedbackCache(
                        chunk_id=chunk_id,
                        positive_count=pos_count,
                        negative_count=neg_count,
                        weighted_positive=pos_weight,
                        weighted_negative=neg_weight,
                        net_score=net,
                    )
                )
            db.commit()


@with_db_retry
def get_diagnostic_thread(thread_id: str) -> DiagnosticThread | None:
    with SessionLocal() as db:
        return (
            db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
        )


# ──────────────────────────────────────────────────────────────────────
# Thread Terminal State (R2)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def update_thread_terminal_state(
    thread_id: str,
    terminal_state: str,
    node_timestamps: dict | None = None,
):
    with SessionLocal() as db:
        thread = (
            db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
        )
        if not thread:
            return
        thread.terminal_state = ThreadTerminalState(terminal_state)
        if node_timestamps:
            thread.node_timestamps = node_timestamps
        db.commit()


@with_db_retry
def get_recent_threads_for_device(
    device_id: str, days: int = 7, user_id: UUID | None = None
) -> list[dict]:
    """Get recent threads for a device that haven't had their outcome recorded."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with SessionLocal() as db:
        q = db.query(DiagnosticThread).filter(
            DiagnosticThread.device_id == device_id,
            DiagnosticThread.created_at >= cutoff,
            DiagnosticThread.outcome_resolved.is_(None),
            DiagnosticThread.terminal_state != ThreadTerminalState.ABANDONED,
        )
        if user_id:
            q = q.filter(DiagnosticThread.user_id == user_id)
        threads = q.order_by(DiagnosticThread.created_at.desc()).limit(5).all()
        return [
            {
                "thread_id": t.id,
                "device_id": t.device_id,
                "date": t.created_at.strftime("%d/%m/%Y") if t.created_at else "",
                "terminal_state": t.terminal_state.value
                if t.terminal_state
                else "IN_PROGRESS",
            }
            for t in threads
        ]


@with_db_retry
def record_outcome(thread_id: str, resolved: bool, user_id: UUID | None = None) -> bool:
    with SessionLocal() as db:
        thread = (
            db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
        )
        if not thread:
            return False
        if user_id and thread.user_id != user_id:
            return False
        thread.outcome_resolved = resolved
        thread.outcome_recorded_at = func.now()
        db.commit()
        return True


# ──────────────────────────────────────────────────────────────────────
# Feedback-Weighted Retrieval (R4)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def apply_feedback_boost(
    results: list[dict],
    alpha: float = 0.05,
    enable: bool = True,
) -> list[dict]:
    """Post-retrieval reranking using normalized, cached feedback scores.

    For each chunk, the raw ``net_score`` is normalized by total signal weight
    and clamped to [-1, 1] so that heavily-reviewed chunks cannot overwhelm
    cosine similarity.  ``alpha`` controls the maximum ranking shift (default
    0.05 means feedback can move a chunk by at most ±0.05 in similarity space).
    """
    if not enable or not results:
        return results
    chunk_ids = [r["id"] for r in results if r.get("id")]
    if not chunk_ids:
        return results

    with SessionLocal() as db:
        cache_rows = (
            db.query(ChunkFeedbackCache)
            .filter(ChunkFeedbackCache.chunk_id.in_(chunk_ids))
            .all()
        )
        boost_map: dict[str, float] = {}
        for c in cache_rows:
            total = c.weighted_positive + abs(c.weighted_negative)
            if total > 0:
                normalized = max(-1.0, min(1.0, c.net_score / total))
            else:
                normalized = 0.0
            boost_map[c.chunk_id] = normalized

    for r in results:
        cid = r.get("id")
        base_score = r.get("score", 0.0)
        boost = boost_map.get(cid, 0.0)
        r["feedback_boost"] = boost
        r["adjusted_score"] = base_score + alpha * boost

    results.sort(key=lambda r: r.get("adjusted_score", 0.0), reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────
# Anti-Examples (R8)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def create_anti_example(
    symptom_pattern: str,
    incorrect_procedure: str,
    correct_alternative: str | None,
    severity: str,
    equipment_category: str,
    device_id: str | None,
    source_type: str,
    source_id: str | None,
    created_by: UUID,
    embedding: list[float] | None = None,
) -> AntiExample:
    from core.models import AntiExampleSeverity, AntiExampleSource

    with SessionLocal() as db:
        ae = AntiExample(
            symptom_pattern=symptom_pattern,
            symptom_embedding=embedding,
            incorrect_procedure=incorrect_procedure,
            correct_alternative=correct_alternative,
            severity=AntiExampleSeverity(severity),
            equipment_category=EquipmentCategory(equipment_category),
            device_id=device_id,
            source_type=AntiExampleSource(source_type),
            source_id=source_id,
            created_by=created_by,
            requires_second_approval=(severity == "SAFETY_CRITICAL"),
        )
        db.add(ae)
        db.commit()
        db.refresh(ae)
        return ae


@with_db_retry
def search_anti_examples(
    query_embedding: list[float],
    equipment_category: str,
    device_id: str | None = None,
    limit: int = 3,
    max_distance: float = 0.45,
) -> list[dict]:
    """Semantic search for active anti-examples matching symptom.

    Args:
        max_distance: Cosine distance cutoff (0.45 ≈ 0.55 similarity). Results
            farther than this are discarded to avoid injecting irrelevant
            anti-examples into RepairPlanner context.
    """
    with SessionLocal() as db:
        eq_cat = EquipmentCategory(equipment_category)
        dist_expr = AntiExample.symptom_embedding.cast(HALFVEC(3072)).cosine_distance(
            query_embedding
        )
        stmt = select(AntiExample, dist_expr.label("dist")).filter(
            AntiExample.is_active.is_(True),
            AntiExample.superseded_by.is_(None),
            AntiExample.symptom_embedding.is_not(None),
            or_(
                AntiExample.equipment_category == eq_cat,
                AntiExample.equipment_category == EquipmentCategory.GENERAL,
            ),
        )
        if device_id:
            stmt = stmt.filter(
                or_(AntiExample.device_id == device_id, AntiExample.device_id.is_(None))
            )

        stmt = stmt.order_by(dist_expr).limit(limit)

        rows = db.execute(stmt).all()
        return [
            {
                "id": str(ae.id),
                "symptom_pattern": ae.symptom_pattern,
                "incorrect_procedure": ae.incorrect_procedure,
                "correct_alternative": ae.correct_alternative,
                "severity": ae.severity.value,
            }
            for ae, dist in rows
            if dist <= max_distance
        ]


@with_db_retry
def get_anti_examples(
    equipment_category: str | None = None,
    active_only: bool = True,
) -> list[AntiExample]:
    with SessionLocal() as db:
        q = db.query(AntiExample)
        if active_only:
            q = q.filter(AntiExample.is_active.is_(True))
        if equipment_category:
            q = q.filter(
                AntiExample.equipment_category == EquipmentCategory(equipment_category)
            )
        return q.order_by(AntiExample.created_at.desc()).all()


@with_db_retry
def supersede_anti_example(old_id: UUID, new_id: UUID):
    with SessionLocal() as db:
        old = db.query(AntiExample).filter(AntiExample.id == old_id).first()
        if old:
            old.superseded_by = new_id
            old.superseded_at = func.now()
            old.is_active = False
            db.commit()


@with_db_retry
def deactivate_anti_example(ae_id: UUID) -> bool:
    with SessionLocal() as db:
        ae = db.query(AntiExample).filter(AntiExample.id == ae_id).first()
        if not ae:
            return False
        ae.is_active = False
        db.commit()
        return True


# ──────────────────────────────────────────────────────────────────────
# Vector Space Blacklisting (R9)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def blacklist_chunk(chunk_id: str, reason: str, admin_id: UUID) -> dict | None:
    """Blacklist a chunk. Returns affected parent summaries for admin awareness."""
    with SessionLocal() as db:
        chunk = (
            db.query(DocumentChunkMultimodal)
            .filter(DocumentChunkMultimodal.id == chunk_id)
            .first()
        )
        if not chunk:
            return None
        chunk.blacklisted = True
        chunk.blacklisted_reason = reason
        chunk.blacklisted_by = admin_id
        chunk.blacklisted_at = func.now()

        # Find parent summaries that may be affected
        affected_parents = []
        if chunk.level == 0 and chunk.document_id:
            parents = (
                db.query(DocumentChunkMultimodal)
                .filter(
                    DocumentChunkMultimodal.document_id == chunk.document_id,
                    DocumentChunkMultimodal.level > 0,
                )
                .all()
            )
            affected_parents = [
                {"id": p.id, "level": p.level, "text_preview": (p.text or "")[:200]}
                for p in parents
            ]

        db.commit()
        return {"affected_parents": affected_parents}


@with_db_retry
def blacklist_chunks(
    chunk_ids: list[str],
    blacklisted_by: UUID,
    reason: str | None = None,
) -> int:
    """Blacklist multiple chunks in a single transaction and write one audit
    row per chunk with event_type='blacklist'. Returns the number of chunks
    actually flipped (skips already-blacklisted or unknown ids).

    Used by the BLACKLIST_CHUNK verdict and REPLACE_MANUAL ingestion swap.
    """
    if not chunk_ids:
        return 0

    flipped = 0
    with SessionLocal() as db:
        chunks = (
            db.query(DocumentChunkMultimodal)
            .filter(DocumentChunkMultimodal.id.in_(chunk_ids))
            .all()
        )
        now = func.now()
        for chunk in chunks:
            if chunk.blacklisted:
                continue
            chunk.blacklisted = True
            chunk.blacklisted_reason = reason
            chunk.blacklisted_by = blacklisted_by
            chunk.blacklisted_at = now
            db.add(
                UnlearningAuditLog(
                    event_type="blacklist",
                    target_type="chunk",
                    target_id=str(chunk.id),
                    performed_by=blacklisted_by,
                    reason=reason,
                    event_metadata={"document_id": chunk.document_id},
                )
            )
            flipped += 1
        db.commit()
    return flipped


@with_db_retry
def get_chunks_for_response(response_id: str) -> list[dict]:
    """Detail-panel citation list: returns chunk + parent document metadata
    for every chunk linked to ``response_id`` via ResponseChunkLink."""
    with SessionLocal() as db:
        rows = (
            db.query(DocumentChunkMultimodal, DocumentMultimodal)
            .join(
                ResponseChunkLink,
                ResponseChunkLink.chunk_id == DocumentChunkMultimodal.id,
            )
            .join(
                DocumentMultimodal,
                DocumentMultimodal.id == DocumentChunkMultimodal.document_id,
            )
            .filter(ResponseChunkLink.response_id == response_id)
            .all()
        )
        return [
            {
                "chunk_id": str(c.id),
                "document_id": c.document_id,
                "document_name": d.intuitive_name or d.original_filename or "",
                "document_filepath": d.filepath or "",
                "pages": c.pages,
                "text_preview": (c.text or "")[:280],
                "blacklisted": bool(c.blacklisted),
            }
            for c, d in rows
        ]


@with_db_retry
def get_chunk_ids_for_document(document_id: int) -> list[str]:
    """All chunk ids belonging to a document, used by REPLACE_MANUAL."""
    with SessionLocal() as db:
        rows = (
            db.query(DocumentChunkMultimodal.id)
            .filter(DocumentChunkMultimodal.document_id == document_id)
            .all()
        )
        return [str(r[0]) for r in rows]


@with_db_retry
def unblacklist_chunk(chunk_id: str):
    with SessionLocal() as db:
        chunk = (
            db.query(DocumentChunkMultimodal)
            .filter(DocumentChunkMultimodal.id == chunk_id)
            .first()
        )
        if chunk:
            chunk.blacklisted = False
            chunk.blacklisted_reason = None
            chunk.blacklisted_by = None
            chunk.blacklisted_at = None
            db.commit()


@with_db_retry
def clear_blacklist_for_document(document_id: int) -> int:
    """Clear blacklist flags for all chunks of a document. Returns count cleared."""
    with SessionLocal() as db:
        count = (
            db.query(DocumentChunkMultimodal)
            .filter(
                DocumentChunkMultimodal.document_id == document_id,
                DocumentChunkMultimodal.blacklisted.is_(True),
            )
            .update(
                {
                    DocumentChunkMultimodal.blacklisted: False,
                    DocumentChunkMultimodal.blacklisted_reason: None,
                    DocumentChunkMultimodal.blacklisted_by: None,
                    DocumentChunkMultimodal.blacklisted_at: None,
                }
            )
        )
        db.commit()
        return count


@with_db_retry
def get_blacklisted_chunks() -> list[dict]:
    with SessionLocal() as db:
        chunks = (
            db.query(DocumentChunkMultimodal, DocumentMultimodal.intuitive_name)
            .join(DocumentMultimodal)
            .filter(DocumentChunkMultimodal.blacklisted.is_(True))
            .all()
        )
        return [
            {
                "chunk_id": c.id,
                "document_name": name or "",
                "text_preview": (c.text or "")[:200],
                "reason": c.blacklisted_reason or "",
                "blacklisted_at": c.blacklisted_at,
            }
            for c, name in chunks
        ]


@with_db_retry
def get_blacklist_candidates() -> list[dict]:
    """Chunks with 5+ negative feedback and neg/pos ratio > 3:1."""
    with SessionLocal() as db:
        candidates = (
            db.query(ChunkFeedbackCache)
            .filter(
                ChunkFeedbackCache.negative_count >= 5,
                ChunkFeedbackCache.positive_count > 0,
                (ChunkFeedbackCache.negative_count / ChunkFeedbackCache.positive_count)
                > 3,
            )
            .all()
        )
        # Also include chunks with 5+ negatives and zero positives
        zero_pos = (
            db.query(ChunkFeedbackCache)
            .filter(
                ChunkFeedbackCache.negative_count >= 5,
                ChunkFeedbackCache.positive_count == 0,
            )
            .all()
        )
        all_candidates = {c.chunk_id: c for c in candidates}
        for c in zero_pos:
            all_candidates[c.chunk_id] = c

        chunk_ids = list(all_candidates.keys())
        if not chunk_ids:
            return []
        chunks = (
            db.query(DocumentChunkMultimodal, DocumentMultimodal.intuitive_name)
            .join(DocumentMultimodal)
            .filter(
                DocumentChunkMultimodal.id.in_(chunk_ids),
                DocumentChunkMultimodal.blacklisted.is_not(True),
            )
            .all()
        )
        return [
            {
                "chunk_id": c.id,
                "document_name": name or "",
                "text_preview": (c.text or "")[:200],
                "negative_count": all_candidates[c.id].negative_count,
                "positive_count": all_candidates[c.id].positive_count,
                "net_score": all_candidates[c.id].net_score,
            }
            for c, name in chunks
        ]


# ──────────────────────────────────────────────────────────────────────
# Deterministic Rules (R10)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def match_deterministic_rules(
    query_text: str,
    device_id: str,
    equipment_category: str,
) -> dict | None:
    """Find the highest-priority active rule matching all keywords."""
    with SessionLocal() as db:
        eq_cat = EquipmentCategory(equipment_category)
        rules = (
            db.query(DeterministicRule)
            .filter(
                DeterministicRule.is_active.is_(True),
                DeterministicRule.approved_by.is_not(None),
                or_(
                    DeterministicRule.equipment_category == eq_cat,
                    DeterministicRule.equipment_category == EquipmentCategory.GENERAL,
                ),
                or_(
                    DeterministicRule.device_id == device_id,
                    DeterministicRule.device_id.is_(None),
                ),
                or_(
                    DeterministicRule.expires_at.is_(None),
                    DeterministicRule.expires_at > func.now(),
                ),
            )
            .order_by(
                DeterministicRule.priority.desc(), DeterministicRule.created_at.desc()
            )
            .all()
        )

        query_lower = query_text.lower()
        for rule in rules:
            if all(kw.lower() in query_lower for kw in rule.trigger_keywords):
                return {
                    "id": str(rule.id),
                    "override_response": rule.override_response,
                    "override_safety_protocols": rule.override_safety_protocols,
                    "source": rule.source.value,
                    "trigger_keywords": rule.trigger_keywords,
                    "priority": rule.priority,
                }
        return None


@with_db_retry
def create_deterministic_rule(
    device_id: str | None,
    equipment_category: str,
    trigger_keywords: list[str],
    override_response: str,
    override_safety_protocols: str | None,
    priority: int,
    source: str,
    created_by: UUID,
    expires_at: datetime | None = None,
    review_date: datetime | None = None,
) -> DeterministicRule:
    from core.models import RuleSource

    with SessionLocal() as db:
        rule = DeterministicRule(
            device_id=device_id,
            equipment_category=EquipmentCategory(equipment_category),
            trigger_keywords=trigger_keywords,
            override_response=override_response,
            override_safety_protocols=override_safety_protocols,
            priority=priority,
            source=RuleSource(source),
            created_by=created_by,
            expires_at=expires_at,
            review_date=review_date,
        )
        db.add(rule)
        db.commit()
        db.refresh(rule)
        return rule


@with_db_retry
def get_deterministic_rules(active_only: bool = True) -> list[DeterministicRule]:
    with SessionLocal() as db:
        q = db.query(DeterministicRule)
        if active_only:
            q = q.filter(DeterministicRule.is_active.is_(True))
        return q.order_by(DeterministicRule.priority.desc()).all()


@with_db_retry
def approve_deterministic_rule(rule_id: UUID, admin_id: UUID) -> bool:
    with SessionLocal() as db:
        rule = (
            db.query(DeterministicRule).filter(DeterministicRule.id == rule_id).first()
        )
        if not rule:
            return False
        rule.is_active = True
        rule.approved_by = admin_id
        rule.approved_at = func.now()
        db.commit()
        return True


@with_db_retry
def deactivate_deterministic_rule(rule_id: UUID) -> bool:
    with SessionLocal() as db:
        rule = (
            db.query(DeterministicRule).filter(DeterministicRule.id == rule_id).first()
        )
        if not rule:
            return False
        rule.is_active = False
        db.commit()
        return True


# ──────────────────────────────────────────────────────────────────────
# Prompt Amendments (R6)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def get_active_amendments(
    target_prompt: str, equipment_category: str | None = None
) -> list[str]:
    """Get active amendment texts for a prompt, optionally filtered by category."""
    with SessionLocal() as db:
        q = db.query(PromptAmendment.amendment_text).filter(
            PromptAmendment.target_prompt == target_prompt,
            PromptAmendment.is_active.is_(True),
        )
        if equipment_category:
            eq_cat = EquipmentCategory(equipment_category)
            q = q.filter(
                or_(
                    PromptAmendment.equipment_category == eq_cat,
                    PromptAmendment.equipment_category.is_(None),
                )
            )
        else:
            q = q.filter(PromptAmendment.equipment_category.is_(None))
        rows = q.order_by(PromptAmendment.created_at.asc()).all()
        return [r[0] for r in rows]


@with_db_retry
def create_prompt_amendment(
    target_prompt: str,
    amendment_text: str,
    reason: str,
    created_by: UUID,
    equipment_category: str | None = None,
) -> PromptAmendment:
    with SessionLocal() as db:
        eq_cat = EquipmentCategory(equipment_category) if equipment_category else None
        pa = PromptAmendment(
            target_prompt=target_prompt,
            equipment_category=eq_cat,
            amendment_text=amendment_text,
            reason=reason,
            created_by=created_by,
        )
        db.add(pa)
        db.commit()
        db.refresh(pa)
        return pa


@with_db_retry
def get_prompt_amendments(
    target_prompt: str | None = None, active_only: bool = False
) -> list[PromptAmendment]:
    with SessionLocal() as db:
        q = db.query(PromptAmendment)
        if target_prompt:
            q = q.filter(PromptAmendment.target_prompt == target_prompt)
        if active_only:
            q = q.filter(PromptAmendment.is_active.is_(True))
        return q.order_by(PromptAmendment.created_at.desc()).all()


@with_db_retry
def activate_prompt_amendment(amendment_id: UUID, admin_id: UUID) -> bool:
    with SessionLocal() as db:
        pa = (
            db.query(PromptAmendment).filter(PromptAmendment.id == amendment_id).first()
        )
        if not pa:
            return False
        pa.is_active = True
        pa.approved_by = admin_id
        pa.approved_at = func.now()
        db.commit()
        return True


@with_db_retry
def update_evaluation_config(
    new_thresholds: dict,
    user_id: Optional[UUID],
    reason: Optional[str],
    source: str,
) -> dict:
    """Atomically update the thresholds row and append a history entry."""
    with SessionLocal() as db:
        row = db.query(EvaluationConfig).filter(EvaluationConfig.id == 1).first()
        if row is None:
            row = EvaluationConfig(id=1, thresholds=new_thresholds, updated_by=user_id)
            db.add(row)
        else:
            row.thresholds = new_thresholds
            row.updated_by = user_id
        db.add(
            EvaluationConfigHistory(
                thresholds=new_thresholds,
                changed_by=user_id,
                change_reason=reason,
                source=source,
            )
        )
        db.commit()
        db.refresh(row)
        return {
            "thresholds": row.thresholds,
            "recommendations": row.recommendations,
            "updated_at": row.updated_at,
        }


@with_db_retry
def update_evaluation_recommendations(recommendations: Optional[dict]) -> None:
    with SessionLocal() as db:
        row = db.query(EvaluationConfig).filter(EvaluationConfig.id == 1).first()
        if row:
            row.recommendations = recommendations
            db.commit()


@with_db_retry
def deactivate_prompt_amendment(amendment_id: UUID) -> bool:
    with SessionLocal() as db:
        pa = (
            db.query(PromptAmendment).filter(PromptAmendment.id == amendment_id).first()
        )
        if not pa:
            return False
        pa.is_active = False
        pa.deactivated_at = func.now()
        db.commit()
        return True


# ──────────────────────────────────────────────────────────────────────
# Preference Pairs (R3)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def create_preference_pair(
    source_type: str,
    source_id: str,
    thread_id: str | None,
    device_id: str,
    equipment_category: str,
    symptom_context: str,
    original_output: str,
    preferred_output: str,
    embedding: list[float] | None = None,
) -> PreferencePair | None:
    """Create a preference pair if one doesn't already exist for this source."""
    from core.models import PreferencePairSource

    with SessionLocal() as db:
        existing = (
            db.query(PreferencePair)
            .filter(
                PreferencePair.source_type == PreferencePairSource(source_type),
                PreferencePair.source_id == source_id,
            )
            .first()
        )
        if existing:
            return None

        pp = PreferencePair(
            source_type=PreferencePairSource(source_type),
            source_id=source_id,
            thread_id=thread_id,
            device_id=device_id,
            equipment_category=EquipmentCategory(equipment_category),
            symptom_context=symptom_context,
            original_output=original_output,
            preferred_output=preferred_output,
            context_embedding=embedding,
        )
        db.add(pp)
        db.commit()
        db.refresh(pp)
        return pp


@with_db_retry
def search_preference_pairs(
    query_embedding: list[float],
    equipment_category: str,
    limit: int = 3,
    max_distance: float = 0.45,
) -> list[dict]:
    """Semantic search for preference pairs matching symptom context.

    Args:
        max_distance: Cosine distance cutoff (0.45 ≈ 0.55 similarity). Prevents
            irrelevant preference pairs from influencing RepairPlanner output.
    """
    with SessionLocal() as db:
        eq_cat = EquipmentCategory(equipment_category)
        dist_expr = PreferencePair.context_embedding.cast(
            HALFVEC(3072)
        ).cosine_distance(query_embedding)
        stmt = (
            select(PreferencePair, dist_expr.label("dist"))
            .filter(
                PreferencePair.context_embedding.is_not(None),
                or_(
                    PreferencePair.equipment_category == eq_cat,
                    PreferencePair.equipment_category == EquipmentCategory.GENERAL,
                ),
            )
            .order_by(dist_expr)
            .limit(limit)
        )
        rows = db.execute(stmt).all()
        return [
            {
                "symptom_context": pp.symptom_context,
                "original_output": pp.original_output[:500],
                "preferred_output": pp.preferred_output[:500],
            }
            for pp, dist in rows
            if dist <= max_distance
        ]


# ──────────────────────────────────────────────────────────────────────
# Quiet Hours (R12)
# ──────────────────────────────────────────────────────────────────────

_quiet_hours_cache: dict[str, tuple[bool, float]] = {}
_QUIET_HOURS_TTL = 300  # 5 minutes


@with_db_retry
def check_quiet_hours_eligibility(device_id: str) -> bool:
    """Check if device has 10+ completed sessions with >90% positive feedback.

    Results are cached per device_id for 5 minutes — the underlying data
    (cumulative session count, feedback ratios) changes slowly.
    """
    cached = _quiet_hours_cache.get(device_id)
    if cached is not None:
        value, ts = cached
        if _time.monotonic() - ts < _QUIET_HOURS_TTL:
            return value

    with SessionLocal() as db:
        thread_id_subq = (
            db.query(DiagnosticThread.id)
            .filter(
                DiagnosticThread.device_id == device_id,
                DiagnosticThread.terminal_state == ThreadTerminalState.COMPLETED,
            )
            .subquery()
        )
        thread_count = db.query(func.count()).select_from(thread_id_subq).scalar()
        if not thread_count or thread_count < 10:
            _quiet_hours_cache[device_id] = (False, _time.monotonic())
            return False

        total_fb = (
            db.query(func.count())
            .select_from(ResponseFeedback)
            .filter(
                ResponseFeedback.thread_id.in_(select(thread_id_subq)),
                ResponseFeedback.retracted.is_not(True),
            )
            .scalar()
        )
        if not total_fb or total_fb < 10:
            _quiet_hours_cache[device_id] = (False, _time.monotonic())
            return False

        positive_fb = (
            db.query(func.coalesce(func.sum(ResponseFeedback.weight), 0.0))
            .filter(
                ResponseFeedback.thread_id.in_(select(thread_id_subq)),
                ResponseFeedback.feedback_type == FeedbackType.THUMBS_UP,
                ResponseFeedback.retracted.is_not(True),
            )
            .scalar()
        )
        total_weight = (
            db.query(func.coalesce(func.sum(ResponseFeedback.weight), 0.0))
            .filter(
                ResponseFeedback.thread_id.in_(select(thread_id_subq)),
                ResponseFeedback.retracted.is_not(True),
            )
            .scalar()
        )
        if total_weight == 0:
            _quiet_hours_cache[device_id] = (False, _time.monotonic())
            return False
        result = (positive_fb / total_weight) > 0.9
        _quiet_hours_cache[device_id] = (result, _time.monotonic())
        return result


# ──────────────────────────────────────────────────────────────────────
# Unlearning Audit Log (R11)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def log_unlearning_event(
    event_type: str,
    target_type: str,
    target_id: str,
    performed_by: UUID,
    reason: str | None = None,
    metadata: dict | None = None,
):
    with SessionLocal() as db:
        entry = UnlearningAuditLog(
            event_type=event_type,
            target_type=target_type,
            target_id=target_id,
            performed_by=performed_by,
            reason=reason,
            event_metadata=metadata,
        )
        db.add(entry)
        db.commit()


@with_db_retry
def get_audit_log(
    event_type: str | None = None,
    target_type: str | None = None,
    performed_by: UUID | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[UnlearningAuditLog], int]:
    with SessionLocal() as db:
        q = db.query(UnlearningAuditLog)
        if event_type:
            q = q.filter(UnlearningAuditLog.event_type == event_type)
        if target_type:
            q = q.filter(UnlearningAuditLog.target_type == target_type)
        if performed_by:
            q = q.filter(UnlearningAuditLog.performed_by == performed_by)
        total = q.count()
        items = (
            q.order_by(UnlearningAuditLog.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total


# ──────────────────────────────────────────────────────────────────────
# Retraction (R7)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def retract_response_feedback(feedback_id: UUID, reason: str, admin_id: UUID) -> bool:
    with SessionLocal() as db:
        fb = (
            db.query(ResponseFeedback)
            .filter(ResponseFeedback.id == feedback_id)
            .first()
        )
        if not fb:
            return False
        fb.retracted = True
        fb.retracted_reason = reason
        fb.retracted_by = admin_id
        fb.retracted_at = func.now()
        db.commit()
        return True


@with_db_retry
def retract_step_feedback(feedback_id: UUID, reason: str, admin_id: UUID) -> bool:
    with SessionLocal() as db:
        fb = db.query(StepFeedback).filter(StepFeedback.id == feedback_id).first()
        if not fb:
            return False
        fb.retracted = True
        fb.retracted_reason = reason
        fb.retracted_by = admin_id
        fb.retracted_at = func.now()
        db.commit()
        return True


@with_db_retry
def get_all_feedback(
    feedback_type_filter: str | None = None,
    equipment_category: str | None = None,
    retracted: bool | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[ResponseFeedback], int]:
    with SessionLocal() as db:
        q = db.query(ResponseFeedback)
        if feedback_type_filter:
            q = q.filter(
                ResponseFeedback.feedback_type == FeedbackType(feedback_type_filter)
            )
        if equipment_category:
            q = q.filter(
                ResponseFeedback.equipment_category
                == EquipmentCategory(equipment_category)
            )
        if retracted is not None:
            q = q.filter(ResponseFeedback.retracted == retracted)
        total = q.count()
        items = (
            q.order_by(ResponseFeedback.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total


@with_db_retry
def get_evaluation_config_history(limit: int = 20) -> list:
    with SessionLocal() as db:
        return (
            db.query(EvaluationConfigHistory)
            .order_by(EvaluationConfigHistory.changed_at.desc())
            .limit(limit)
            .all()
        )


# ──────────────────────────────────────────────────────────────────────
# Escalation (H5)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def create_escalation_flag(
    thread_id: str,
    flagged_by: UUID,
    response_id: str | None = None,
    reason: str | None = None,
    assigned_reviewer: UUID | None = None,
) -> EscalationFlag:
    with SessionLocal() as db:
        flag = EscalationFlag(
            thread_id=thread_id,
            response_id=response_id,
            flagged_by=flagged_by,
            reason=reason,
            assigned_reviewer=assigned_reviewer,
        )
        db.add(flag)
        # Denormalize onto thread
        thread = (
            db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
        )
        if thread:
            thread.is_escalated = True
        db.commit()
        db.refresh(flag)
        return flag


@with_db_retry
def get_escalation_flags(thread_id: str) -> list[EscalationFlag]:
    with SessionLocal() as db:
        return (
            db.query(EscalationFlag)
            .filter(EscalationFlag.thread_id == thread_id)
            .order_by(EscalationFlag.created_at.asc())
            .all()
        )


# ──────────────────────────────────────────────────────────────────────
# Async Review Queue (replaces sync interrupt_gates flow)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def create_review_task(
    thread_id: str,
    flagged_by: UUID | None,
    source: ReviewTaskSource,
    priority: ReviewTaskPriority,
    *,
    response_id: str | None = None,
    trigger_reason: str | None = None,
    payload_snapshot: dict | None = None,
    reason: str | None = None,
) -> EscalationFlag:
    """Create an async review task. Field tech keeps working; admins/experts
    see the task in their dashboards. Marks the thread as flagged_for_review.
    """
    with SessionLocal() as db:
        task = EscalationFlag(
            thread_id=thread_id,
            response_id=response_id,
            flagged_by=flagged_by,
            reason=reason,
            source=source,
            priority=priority,
            status=ReviewTaskStatus.NEW,
            trigger_reason=trigger_reason,
            payload_snapshot=payload_snapshot or {},
        )
        db.add(task)
        thread = (
            db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
        )
        if thread:
            thread.is_escalated = True
            thread.flagged_for_review = True
        db.commit()
        db.refresh(task)
        return task


@with_db_retry
def get_review_task(task_id: UUID) -> EscalationFlag | None:
    with SessionLocal() as db:
        return db.query(EscalationFlag).filter(EscalationFlag.id == task_id).first()


@with_db_retry
def list_review_tasks(
    *,
    status: ReviewTaskStatus | None = None,
    priority: ReviewTaskPriority | None = None,
    assigned_to: UUID | None = None,
    include_snoozed: bool = False,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[EscalationFlag], int]:
    """Queue accessor with filters. Sorted by priority (P0 first), then age.

    ``assigned_to`` returns tasks claimed by that user. ``include_snoozed``
    is False by default — SNOOZED tasks are hidden until snoozed_until passes.
    """
    from sqlalchemy import case

    with SessionLocal() as db:
        q = db.query(EscalationFlag)
        if status is not None:
            q = q.filter(EscalationFlag.status == status)
        if priority is not None:
            q = q.filter(EscalationFlag.priority == priority)
        if assigned_to is not None:
            q = q.filter(EscalationFlag.claimed_by == assigned_to)
        if not include_snoozed:
            now = datetime.now(timezone.utc)
            q = q.filter(
                (EscalationFlag.status != ReviewTaskStatus.SNOOZED)
                | (EscalationFlag.snoozed_until <= now)
            )

        priority_order = case(
            (EscalationFlag.priority == ReviewTaskPriority.P0, 0),
            (EscalationFlag.priority == ReviewTaskPriority.P1, 1),
            (EscalationFlag.priority == ReviewTaskPriority.P2, 2),
            (EscalationFlag.priority == ReviewTaskPriority.P3, 3),
            else_=4,
        )
        total = q.count()
        items = (
            q.order_by(priority_order, EscalationFlag.created_at.asc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return items, total


@with_db_retry
def claim_review_task(task_id: UUID, expert_id: UUID) -> EscalationFlag | None:
    """Atomically claim a NEW task. Returns None if already claimed."""
    with SessionLocal() as db:
        task = (
            db.query(EscalationFlag)
            .filter(EscalationFlag.id == task_id)
            .with_for_update()
            .first()
        )
        if not task:
            return None
        if task.status not in (ReviewTaskStatus.NEW, ReviewTaskStatus.SNOOZED):
            return None  # already claimed or resolved
        task.status = ReviewTaskStatus.CLAIMED
        task.claimed_by = expert_id
        task.claimed_at = datetime.now(timezone.utc)
        task.snoozed_until = None
        db.commit()
        db.refresh(task)
        return task


@with_db_retry
def snooze_review_task(
    task_id: UUID, expert_id: UUID, snooze_until: datetime
) -> EscalationFlag | None:
    """Defer a task. Returns it to the queue at ``snooze_until``."""
    with SessionLocal() as db:
        task = db.query(EscalationFlag).filter(EscalationFlag.id == task_id).first()
        if not task:
            return None
        task.status = ReviewTaskStatus.SNOOZED
        task.snoozed_until = snooze_until
        db.commit()
        db.refresh(task)
        return task


@with_db_retry
def mark_review_task_in_review(task_id: UUID, expert_id: UUID) -> EscalationFlag | None:
    """Move a task to IN_REVIEW after work has begun (e.g. REPLACE_MANUAL
    upload accepted). No-op if already IN_REVIEW or RESOLVED."""
    with SessionLocal() as db:
        task = db.query(EscalationFlag).filter(EscalationFlag.id == task_id).first()
        if not task:
            return None
        if task.status == ReviewTaskStatus.RESOLVED:
            return task
        task.status = ReviewTaskStatus.IN_REVIEW
        task.claimed_by = task.claimed_by or expert_id
        task.claimed_at = task.claimed_at or datetime.now(timezone.utc)
        task.snoozed_until = None
        db.commit()
        db.refresh(task)
        return task


@with_db_retry
def resolve_review_task(task_id: UUID, expert_id: UUID) -> EscalationFlag | None:
    """Mark a task RESOLVED after a verdict has been written. Uses SELECT FOR UPDATE to prevent concurrent verdicts."""
    with SessionLocal() as db:
        task = (
            db.query(EscalationFlag)
            .filter(EscalationFlag.id == task_id)
            .with_for_update()
            .first()
        )
        if not task:
            return None
        if task.status == ReviewTaskStatus.RESOLVED:
            return None
        task.status = ReviewTaskStatus.RESOLVED
        task.claimed_by = task.claimed_by or expert_id
        task.resolved_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(task)
        return task


# ──────────────────────────────────────────────────────────────────────
# Notifications (in-app expert/admin alerts)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def create_notification(
    user_id: UUID,
    kind: NotificationKind,
    title: str,
    *,
    body: str | None = None,
    link: str | None = None,
    task_id: UUID | None = None,
) -> Notification:
    with SessionLocal() as db:
        n = Notification(
            user_id=user_id,
            kind=kind,
            title=title,
            body=body,
            link=link,
            task_id=task_id,
        )
        db.add(n)
        db.commit()
        db.refresh(n)
        return n


@with_db_retry
def list_unread_notifications(user_id: UUID, limit: int = 50) -> list[Notification]:
    with SessionLocal() as db:
        return (
            db.query(Notification)
            .filter(Notification.user_id == user_id, Notification.read_at.is_(None))
            .order_by(Notification.created_at.desc())
            .limit(limit)
            .all()
        )


@with_db_retry
def mark_notifications_read(user_id: UUID, notification_ids: list[UUID]) -> int:
    with SessionLocal() as db:
        now = datetime.now(timezone.utc)
        count = (
            db.query(Notification)
            .filter(
                Notification.user_id == user_id,
                Notification.id.in_(notification_ids),
                Notification.read_at.is_(None),
            )
            .update({"read_at": now}, synchronize_session=False)
        )
        db.commit()
        return count


# ──────────────────────────────────────────────────────────────────────
# Review Verdicts (H6 — contract with Workstream R)
# ──────────────────────────────────────────────────────────────────────


@with_db_retry
def create_review_verdict(
    thread_id: str,
    response_id: str,
    node_name: str,
    verdict: str,
    reviewer_id: UUID,
    step_index: int | None = None,
    correct_procedure: str | None = None,
    error_category: str | None = None,
    notes: str | None = None,
) -> ReviewVerdict:
    with SessionLocal() as db:
        rv = ReviewVerdict(
            thread_id=thread_id,
            response_id=response_id,
            node_name=node_name,
            step_index=step_index,
            verdict=VerdictEnum(verdict),
            correct_procedure=correct_procedure,
            error_category=ErrorCategory(error_category) if error_category else None,
            reviewer_id=reviewer_id,
            notes=notes,
        )
        db.add(rv)
        db.commit()
        db.refresh(rv)
        return rv


@with_db_retry
def get_verdicts_for_thread(thread_id: str) -> list[ReviewVerdict]:
    with SessionLocal() as db:
        return (
            db.query(ReviewVerdict)
            .filter(ReviewVerdict.thread_id == thread_id)
            .order_by(ReviewVerdict.created_at.asc())
            .all()
        )


@with_db_retry
def get_review_queue(
    status_filter: str | None = None,
    search: str | None = None,
    sort_by: str = "date",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict], int]:
    """Paginated review queue with cross-field search and evaluation metrics."""
    with SessionLocal() as db:
        q = db.query(DiagnosticThread).options(selectinload(DiagnosticThread.user))

        if status_filter:
            try:
                q = q.filter(
                    DiagnosticThread.review_status == ReviewStatus(status_filter)
                )
            except ValueError:
                return [], 0

        if search:
            escaped = search.replace("%", r"\%").replace("_", r"\_")
            pattern = f"%{escaped}%"
            q = q.outerjoin(User, DiagnosticThread.user_id == User.id).filter(
                or_(
                    DiagnosticThread.device_id.ilike(pattern),
                    DiagnosticThread.id.ilike(pattern),
                    User.email.ilike(pattern),
                )
            )

        total = q.count()

        sort_col_map = {
            "date": DiagnosticThread.created_at,
            "device_id": DiagnosticThread.device_id,
        }
        col = sort_col_map.get(sort_by, DiagnosticThread.created_at)
        order = col.asc() if sort_order == "asc" else col.desc().nullslast()

        threads = (
            q.order_by(
                DiagnosticThread.is_escalated.desc(),
                DiagnosticThread.risk_tier.desc().nullslast(),
                order,
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )

        # Batch-fetch latest evaluation metrics for all threads on this page
        thread_ids = [t.id for t in threads]
        eval_map: dict[str, dict] = {}
        if thread_ids:
            # Subquery: latest eval row per thread (highest id = final verdict)
            latest_eval = (
                db.query(
                    EvaluationMetric.thread_id,
                    func.max(EvaluationMetric.id).label("max_id"),
                )
                .filter(EvaluationMetric.thread_id.in_(thread_ids))
                .group_by(EvaluationMetric.thread_id)
                .subquery()
            )
            eval_rows = (
                db.query(EvaluationMetric)
                .filter(EvaluationMetric.id.in_(select(latest_eval.c.max_id)))
                .all()
            )
            for r in eval_rows:
                eval_map[r.thread_id] = {
                    "badge": r.badge,
                    "quality_badge": r.quality_badge,
                    "safety_badge": r.safety_badge,
                    "sync_faithfulness": r.sync_faithfulness,
                    "sync_answer_relevance": r.sync_answer_relevance,
                }

        items = []
        for t in threads:
            ev = eval_map.get(t.id, {})
            items.append(
                {
                    "thread_id": t.id,
                    "device_id": t.device_id,
                    "date": t.created_at.isoformat() if t.created_at else "",
                    "review_status": t.review_status.value
                    if t.review_status
                    else "PENDING_REVIEW",
                    "is_escalated": t.is_escalated or False,
                    "risk_tier": t.risk_tier or 0,
                    "user_email": t.user.email if t.user else None,
                    "terminal_state": t.terminal_state.value
                    if t.terminal_state
                    else "IN_PROGRESS",
                    "badge": ev.get("badge"),
                    "quality_badge": ev.get("quality_badge"),
                    "safety_badge": ev.get("safety_badge"),
                    "faithfulness": ev.get("sync_faithfulness"),
                    "answer_relevance": ev.get("sync_answer_relevance"),
                }
            )
        return items, total


@with_db_retry
def bulk_approve_sessions(thread_ids: list[str], reviewer_id: UUID) -> int:
    with SessionLocal() as db:
        count = (
            db.query(DiagnosticThread)
            .filter(
                DiagnosticThread.id.in_(thread_ids),
                DiagnosticThread.review_status == ReviewStatus.PENDING_REVIEW,
            )
            .update(
                {
                    DiagnosticThread.review_status: ReviewStatus.BULK_APPROVED,
                    DiagnosticThread.reviewed_by: reviewer_id,
                    DiagnosticThread.reviewed_at: func.now(),
                }
            )
        )
        db.commit()
        return count


@with_db_retry
def update_thread_review_status(
    thread_id: str,
    status: str,
    reviewer_id: UUID,
) -> bool:
    with SessionLocal() as db:
        thread = (
            db.query(DiagnosticThread).filter(DiagnosticThread.id == thread_id).first()
        )
        if not thread:
            return False
        thread.review_status = ReviewStatus(status)
        thread.reviewed_by = reviewer_id
        thread.reviewed_at = func.now()
        db.commit()
        return True


@with_db_retry
def query_evaluation_metrics(filters: dict) -> List[EvaluationMetric]:
    """Driver for analytics endpoints. Supports filters:
        start, end (datetime), device_id, user_id (UUID), badge,
        limit (int, capped 200), offset (int).
    Returns ORM rows ordered by created_at DESC.
    """
    limit = min(int(filters.get("limit") or 50), 200)
    offset = int(filters.get("offset") or 0)
    with SessionLocal() as db:
        q = db.query(EvaluationMetric)
        if filters.get("start"):
            q = q.filter(EvaluationMetric.created_at >= filters["start"])
        if filters.get("end"):
            q = q.filter(EvaluationMetric.created_at <= filters["end"])
        if filters.get("device_id"):
            q = q.filter(EvaluationMetric.device_id == filters["device_id"])
        if filters.get("user_id"):
            q = q.filter(EvaluationMetric.user_id == filters["user_id"])
        if filters.get("badge"):
            q = q.filter(EvaluationMetric.badge == filters["badge"])
        return (
            q.order_by(EvaluationMetric.created_at.desc())
            .limit(limit)
            .offset(offset)
            .all()
        )


@with_db_retry
def count_evaluation_metrics(filters: dict) -> int:
    with SessionLocal() as db:
        q = db.query(func.count(EvaluationMetric.id))
        if filters.get("start"):
            q = q.filter(EvaluationMetric.created_at >= filters["start"])
        if filters.get("end"):
            q = q.filter(EvaluationMetric.created_at <= filters["end"])
        if filters.get("device_id"):
            q = q.filter(EvaluationMetric.device_id == filters["device_id"])
        if filters.get("user_id"):
            q = q.filter(EvaluationMetric.user_id == filters["user_id"])
        if filters.get("badge"):
            q = q.filter(EvaluationMetric.badge == filters["badge"])
        return q.scalar() or 0
