import os
import base64
import logging
import time
from typing import Optional, TypedDict

from langchain_core.tools import tool

from core.crud import (
    multimodal_semantic_search,
    apply_feedback_boost,
    get_manual_walkthrough,
    fetch_parent_chunks,
)
from core.config import FEEDBACK_BOOST_ALPHA, RETRIEVAL_MODE
from core.eval_config import get_thresholds
from modules.embeddings import embed
from modules.reranker import llm_rerank_chunks

logger = logging.getLogger(__name__)


class RetrievedChunk(TypedDict):
    id: str
    score: float | None  # None for procedural walkthrough chunks (no similarity)
    text: str
    page_start: int | None
    section_header: str | None
    images: list[dict]  # [{path, caption}]


@tool("Technical_Manual_Search")
def manual_search_tool(
    query: str,
    device_id: str,
    image_base64: Optional[str] = None,
    procedural: bool = False,
    search_terms: Optional[list[str]] = None,
    document_id: Optional[int] = None,
    rerank_min_score: int = 0,
    rerank_top_k: int = 5,
    retrieval_mode: str = RETRIEVAL_MODE,
) -> dict:
    """Search the technical manuals for mechanical/electrical specs, troubleshooting guides, and repair procedures."""
    logger.info(
        f"[TOOL] Manual search: '{query}' (device={device_id}, procedural={procedural})"
    )

    # Procedural shortcut: for walkthrough-style questions ("how to
    # start assemble", "walk me through setup"), the reading order IS
    # the answer. Bypass embed → similarity → RRF → rerank entirely and
    # pull every leaf chunk (level=0) for the device's manual in
    # sequence order. When search_terms are provided, the walkthrough is
    # narrowed to the matching section (e.g., only "Display Replacement"
    # chunks instead of the entire manual).
    chunk_scores: dict = {}
    pre_rerank_scores: dict = {}
    retrieval_trace: dict = {}

    if procedural:
        t0 = time.time()
        walkthrough = get_manual_walkthrough(
            device_id, search_terms=search_terms, limit=80
        )
        walkthrough_query_ms = int((time.time() - t0) * 1000)
        if not walkthrough:
            return {"markdown": "Data not present.", "image_keys": []}
        top_chunks = walkthrough
        retrieval_trace = {
            "walkthrough_query_ms": walkthrough_query_ms,
            "total_chunks": len(walkthrough),
        }
    else:
        # Decode any attached image once up front.
        img_bytes: Optional[bytes] = None
        if image_base64:
            try:
                img_bytes = base64.b64decode(image_base64)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[TOOL] image_base64 decode failed: {e}")
                img_bytes = None

        image_bytes_list = [img_bytes] if img_bytes else None
        t0 = time.time()
        query_text_emb = embed(text=query, image_bytes_list=image_bytes_list)
        query_embed_ms = int((time.time() - t0) * 1000)
        if not query_text_emb:
            logger.error("[TOOL] Failed to embed query")
            return {
                "markdown": "Search failed: could not embed query.",
                "image_keys": [],
            }

        vector_results, manual_search_ms = multimodal_semantic_search(
            query_text_emb,
            "manual",
            device_id,
            document_id=document_id,
            retrieval_mode=retrieval_mode,
        )
        boost_alpha = get_thresholds().get(
            "feedback_boost_weight", FEEDBACK_BOOST_ALPHA
        )
        boosted_results = apply_feedback_boost(vector_results, alpha=boost_alpha)
        if not boosted_results:
            return {"markdown": "Data not present.", "image_keys": []}

        # Snapshot pre-rerank scores for tracing
        pre_rerank_scores = {
            r["id"]: {
                "cosine": r["score"],
                "adjusted": r.get("adjusted_score", r["score"]),
            }
            for r in boosted_results
            if r.get("id")
        }

        # Stage 2: LLM reranker — skip when too few candidates to discriminate
        if len(boosted_results) <= 2:
            top_chunks = boosted_results
            scores_map: dict = {}
            reranker_meta: dict = {"skipped": True, "reason": "too_few_candidates"}
        else:
            top_chunks, scores_map, reranker_meta = llm_rerank_chunks(
                query,
                image_base64 or "",
                boosted_results,
                min_score=rerank_min_score,
                top_k=rerank_top_k,
            )

        # Build per-chunk nested scores
        for chunk in top_chunks:
            cid = chunk.get("id")
            if cid and cid in pre_rerank_scores:
                chunk_scores[cid] = {
                    "cosine": pre_rerank_scores[cid]["cosine"],
                    "adjusted": pre_rerank_scores[cid]["adjusted"],
                    "reranker": scores_map.get(cid),
                }

        retrieval_trace = {
            "query_embed_ms": query_embed_ms,
            "manual_search_ms": manual_search_ms,
            "candidates_before_rerank": len(boosted_results),
            "candidates_after_rerank": len(top_chunks),
            "feedback_boost_applied": True,
            "chunks_boosted": sum(
                1 for r in boosted_results if r.get("feedback_boost", 0) != 0
            ),
        }
        if not reranker_meta.get("skipped"):
            retrieval_trace["reranker_llm_ms"] = reranker_meta.get("latency_ms")
            retrieval_trace["reranker_input_tokens"] = reranker_meta.get("input_tokens")
            retrieval_trace["reranker_output_tokens"] = reranker_meta.get(
                "output_tokens"
            )
            retrieval_trace["reranker_fallback_triggered"] = reranker_meta.get(
                "fallback_triggered", False
            )
            retrieval_trace["min_score_filter_applied"] = reranker_meta.get(
                "min_score_applied", False
            )

    # Parent-child text swap: replace table-row children with full parent
    # table text, deduplicating so each table appears only once.
    parent_ids = {c.get("parent_id") for c in top_chunks if c.get("parent_id")}
    if parent_ids:
        parents = fetch_parent_chunks(parent_ids)
        seen_parents: set[str] = set()
        expanded: list[dict] = []
        for chunk in top_chunks:
            pid = chunk.get("parent_id")
            if pid:
                if pid not in seen_parents:
                    seen_parents.add(pid)
                    parent = parents.get(pid)
                    if parent:
                        chunk = {
                            **chunk,
                            "text": parent["text"],
                            "pages": parent["pages"],
                            "images": parent["images"],
                        }
                        expanded.append(chunk)
                # Duplicate children of same parent → skip
            else:
                expanded.append(chunk)
        top_chunks = expanded

    chunk_blocks: list[dict] = []
    image_keys: list[str] = []
    gcs_bucket = os.getenv("GCS_BUCKET_NAME", "").replace('"', "").replace("'", "")

    for res in top_chunks:
        seq = res.get("sequence_index")
        seq_suffix = f" | Seq {seq}" if seq is not None else ""
        block = (
            f"---[{res['intuitive_name']} | doc: {res['doc_path']} | Page {res['pages']}{seq_suffix}]---\n"
            f"{res['text']}\n"
        )

        block_image_keys: list[str] = []
        block_image_urls: list[dict] = []
        if res["images"]:
            block += "\nRelevant Diagrams:\n"
            for img in res["images"]:
                img_path = img["path"]
                if img_path.startswith("http"):
                    full_url = img_path
                else:
                    full_url = f"https://storage.googleapis.com/{gcs_bucket}/{img_path}"
                block += f"![{img.get('caption', 'Diagram')}]({full_url})\n"
                image_keys.append(img_path)
                block_image_keys.append(img_path)
                block_image_urls.append(
                    {"url": full_url, "caption": img.get("caption", "Diagram")}
                )

        chunk_blocks.append(
            {
                "id": res.get("id"),
                "score": res.get("score"),
                "sequence_index": seq,
                "page_start": res.get("page_start"),
                "section_header": res.get("section_header"),
                "parent_id": res.get("parent_id"),
                "reranker_score": res.get("reranker_score"),
                "block": block,
                "image_keys": block_image_keys,
                "image_urls": block_image_urls,
            }
        )

    # Preserve first-seen order for image_keys instead of randomizing
    # via set(). The planner node only fetches the first 3, so order
    # matters.
    seen_keys: set[str] = set()
    ordered_image_keys: list[str] = []
    for k in image_keys:
        if k and k not in seen_keys:
            seen_keys.add(k)
            ordered_image_keys.append(k)

    return {
        "markdown": "\n\n".join(b["block"] for b in chunk_blocks),
        "image_keys": ordered_image_keys,
        "chunks": chunk_blocks,
        "chunk_scores": chunk_scores,
        "pre_rerank_scores": pre_rerank_scores,
        "retrieval_trace": retrieval_trace,
    }


# ---------------------------------------------------------------------------
# Path B: Safety-augmented supplementary retrieval for SafetyExtractor
# ---------------------------------------------------------------------------

SAFETY_BOOST_SCORE = 0.05

# Minimum hard-filter results before falling back to soft-boost search.
_MIN_HARD_FILTER_RESULTS = 3


def safety_augmented_search(
    equipment_type: str,
    device_id: str,
    top_k: int = 10,
) -> list[dict]:
    """Retrieve safety-relevant chunks for SafetyExtractor Path B.

    Two-phase approach (merged from legacy SafetyEvaluator Path B):
      1. Hard-filter: vector search restricted to has_safety_content=True chunks.
         Guarantees all results are safety-tagged.
      2. Soft-boost fallback: if hard-filter yields < _MIN_HARD_FILTER_RESULTS,
         runs a general search with +0.05 score boost for safety-tagged chunks.
    Results are deduped and returned sorted by score.
    """
    safety_query = f"{equipment_type} safety hazard warning caution lockout PPE"

    try:
        query_embedding = embed(text=safety_query)
    except Exception as e:
        logger.warning(f"[SafetyExtractor] Path B embedding failed: {e}")
        return []

    # Phase 1: Hard-filter search (safety-tagged chunks only)
    chunks: list[dict] = []
    try:
        hard_results, _ms = multimodal_semantic_search(
            query_text_emb=query_embedding,
            device_id=device_id,
            doc_type="manual",
            safety_only=True,
        )
        for row in hard_results[:top_k]:
            chunks.append(
                {
                    "id": row.get("id", ""),
                    "text": row.get("text", ""),
                    "score": row.get("score", 0.0),
                    "has_safety_content": True,
                    "section_header": row.get("section_header"),
                }
            )
    except Exception as e:
        logger.warning(f"[SafetyExtractor] Path B hard-filter search failed: {e}")

    # Phase 2: Soft-boost fallback if hard-filter returned insufficient results
    if len(chunks) < _MIN_HARD_FILTER_RESULTS:
        try:
            soft_results, _ms = multimodal_semantic_search(
                query_text_emb=query_embedding,
                device_id=device_id,
                doc_type="manual",
            )
            seen_ids = {c["id"] for c in chunks}
            for row in soft_results[:top_k]:
                cid = row.get("id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    score = row.get("score", 0.0)
                    if row.get("has_safety_content"):
                        score = (score or 0.0) + SAFETY_BOOST_SCORE
                    chunks.append(
                        {
                            "id": cid,
                            "text": row.get("text", ""),
                            "score": score,
                            "has_safety_content": row.get("has_safety_content", False),
                            "section_header": row.get("section_header"),
                        }
                    )
        except Exception as e:
            logger.warning(f"[SafetyExtractor] Path B soft-boost fallback failed: {e}")

    chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return chunks[:top_k]
