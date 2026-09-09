# backend/modules/graph/nodes.py
import json
import logging
import os
import re
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

from google.genai import types
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from datetime import datetime, timezone
from core.config import (
    llm_pro_langchain,
    MODEL_FLASH,
    MODEL_PRO,
    QUERY_REWRITE_ENABLED,
)
from pipeline.telemetry import (
    make_node_telemetry,
    tracked_generate_with_retry,
    tracked_invoke,
    tracked_stream,
    tracked_stream_with_writer,
)
from core.crud import (
    insert_evaluation_metric,
    select_best_document,
    get_document_summaries,
    get_chunks_by_sequence_indices,
    set_thread_flag,
)
from core.eval_config import get_thresholds
from core.utils import parse_gemini_json
from modules.embeddings import embed
from pipeline.eval_logic import (
    INLINE_METRIC_KEYS,
    MAX_INLINE_EVAL_IMAGE_PARTS,
    ArtifactType,
    classify_artifact,
    compute_quality_badge,
    extract_context,
    extract_evaluable_text,
    extract_image_parts,
    extract_manual_context,
    extract_question,
    is_plan_failure,
)
from pipeline.state import DiagnosticState, SupervisorIntakeSchema
from modules.chunk_grouping import (
    build_grouped_context,
    extract_cited_chunk_ids,
    filter_context_by_chunk_ids,
    group_chunks,
)
from pipeline.tools import manual_search_tool
from pipeline.utils import extract_gcs_uris
from prompts.system_prompts import (
    get_prompt,
    SYSTEM_PROMPT_ROOT_CAUSE,
    SYSTEM_PROMPT_REPAIR_PLANNER,
    SYSTEM_PROMPT_QUERY_REWRITER,
    PROMPT_FOLLOWUP_RESPONDER,
)

logger = logging.getLogger(__name__)


def with_node_timeout(seconds=60, fallback: dict | None = None):
    """Decorator to add a timeout to LangGraph node functions.
    Uses a background thread to enforce the timeout cross-platform (no SIGALRM on macOS threads).
    On timeout, returns the fallback dict instead of raising — prevents stream-killing crashes.
    Copies contextvars so LangGraph's get_stream_writer/get_config work inside the thread.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import contextvars

            result = [None]
            exception = [None]
            ctx = contextvars.copy_context()

            def target():
                try:
                    result[0] = ctx.run(func, *args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout=seconds)
            if thread.is_alive():
                logger.error(
                    f"Node {func.__name__} timed out after {seconds}s — returning fallback"
                )
                timeout_telemetry = [
                    {
                        "node_name": func.__name__,
                        "operation_type": "node",
                        "model": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "latency_ms": int(seconds * 1000),
                        "started_at": datetime.now(timezone.utc),
                        "completed_at": datetime.now(timezone.utc),
                        "error": f"timeout after {seconds}s",
                        "metadata": None,
                    }
                ]
                if fallback is not None:
                    fb = {**fallback, "execution_telemetry": timeout_telemetry}
                    return fb
                return {"execution_telemetry": timeout_telemetry}
            if exception[0]:
                raise exception[0]
            return result[0]

        return wrapper

    return decorator


def extract_clean_string(llm_response) -> str:
    content = llm_response.content
    if isinstance(content, list):
        return "".join(
            [
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ]
        )
    return str(content)


@with_node_timeout(
    seconds=120,
    fallback={
        "route": "troubleshoot",
        "extracted_part": "",
        "equipment_type": "heavy_industrial",
        "search_queries": ["diagnostic error"],
        "detected_image_type": "none",
        "is_procedural": False,
        "is_scoped_procedural": False,
    },
)
def intent_router(state: DiagnosticState) -> dict:
    messages = state.get("messages", [])
    if not messages:
        return {
            "messages": [{"role": "assistant", "content": "No user message detected."}]
        }

    last_user_msg = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    device_id = state.get("device_id", "Unknown Device")

    user_text = ""
    has_image = False
    if isinstance(last_user_msg.get("content"), str):
        user_text = last_user_msg["content"]
    elif isinstance(last_user_msg.get("content"), list):
        for item in last_user_msg["content"]:
            if item.get("type") == "text":
                user_text += item.get("data", "")
            elif item.get("type") == "image":
                has_image = True

    previous_plan = "No prior plan."
    for m in reversed(messages[:-1]):
        if m.get("role") == "assistant" and isinstance(m.get("content"), dict):
            for e in m["content"].get("events", []):
                if "RepairPlanner" in e and e["RepairPlanner"].get("final_response"):
                    previous_plan = json.dumps(e["RepairPlanner"]["final_response"])
                    break
            if previous_plan != "No prior plan.":
                break

    # R5: Device history digest — fetched before the LLM call so the
    # prompt template's {device_history} placeholder gets populated.
    device_history = ""
    try:
        from modules.device_history import get_device_history_digest

        device_history = get_device_history_digest(device_id)
    except Exception as e:
        logger.debug(f"Device history unavailable: {e}")

    prompt_name = (
        "PROMPT_INTENT_ROUTER_VISION" if has_image else "PROMPT_INTENT_ROUTER_TEXT"
    )
    prompt_choice = get_prompt(prompt_name, state.get("equipment_type"))
    parser = PydanticOutputParser(pydantic_object=SupervisorIntakeSchema)
    prompt = ChatPromptTemplate.from_messages(
        [("system", prompt_choice), ("user", "{format_instructions}")]
    )
    chain = prompt | llm_pro_langchain | parser

    invoke_result, telemetry = tracked_invoke(
        chain,
        {
            "previous_plan": previous_plan,
            "user_text": user_text,
            "device_id": device_id,
            "device_history": device_history,
            "format_instructions": parser.get_format_instructions(),
        },
        "IntentRouter",
        MODEL_PRO,
    )
    if invoke_result is None:
        result = SupervisorIntakeSchema(
            intent="troubleshoot",
            extracted_part="",
            equipment_type="heavy_industrial",
            search_queries=["diagnostic error"],
        )
    else:
        result = invoke_result

    route = "mismatch" if result.is_device_mismatch else result.intent

    # L2-E01: Enrich telemetry with routing decision metadata
    telemetry["metadata"] = {
        **(telemetry.get("metadata") or {}),
        "classified_intent": route,
        "equipment_type": result.equipment_type,
        "routing_decision": route,
        "is_procedural": result.is_procedural,
    }

    return {
        "route": route,
        "extracted_part": result.extracted_part,
        "equipment_type": result.equipment_type,
        "search_queries": result.search_queries,
        "detected_image_type": result.detected_image_type,
        "is_procedural": result.is_procedural,
        "is_scoped_procedural": result.is_scoped_procedural,
        "safety_warning": result.safety_warning,
        "corrected_intent": result.corrected_intent,
        "device_history_digest": device_history,
        "execution_telemetry": [telemetry],
    }


def symptom_analyzer(state: DiagnosticState) -> dict:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    queries = state.get("search_queries", [])

    if not queries or not QUERY_REWRITE_ENABLED:
        if not queries:
            queries = ["system diagnostics", "troubleshooting procedures"]
        return {
            "search_queries": queries,
            "execution_telemetry": [
                make_node_telemetry(
                    "SymptomAnalyzer",
                    started_at,
                    int((time.perf_counter() - t0) * 1000),
                    metadata={
                        "equipment_category": state.get("equipment_type"),
                        "rewritten_queries": queries,
                        "rewrite_skipped": True,
                        "query_rewrite_enabled": QUERY_REWRITE_ENABLED,
                    },
                )
            ],
        }

    # Extract the user's original observation for context
    observation = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "human":
            observation = msg.get("content", "")
            break
        elif isinstance(msg, HumanMessage):
            observation = msg.content if isinstance(msg.content, str) else ""
            break
    if not observation:
        observation = queries[0]

    try:
        prompt = SYSTEM_PROMPT_QUERY_REWRITER.format(
            observation=observation[:500],
            initial_queries=json.dumps(queries[:3]),
        )
        response, rewrite_telemetry = tracked_generate_with_retry(
            node_name="SymptomAnalyzer",
            model=MODEL_FLASH,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        if response is None:
            raise ValueError("query rewrite generation returned None")
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
            raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE).strip()
        rewritten = json.loads(raw)
        if isinstance(rewritten, list) and len(rewritten) >= 2:
            queries = [str(q) for q in rewritten[:3]]
            logger.info(f"[SymptomAnalyzer] Rewrote queries: {queries}")
    except Exception as e:
        rewrite_telemetry = None
        logger.warning(f"[SymptomAnalyzer] Query rewrite failed, using originals: {e}")

    telemetry_entries = [
        make_node_telemetry(
            "SymptomAnalyzer",
            started_at,
            int((time.perf_counter() - t0) * 1000),
            metadata={
                "equipment_category": state.get("equipment_type"),
                "rewritten_queries": queries,
                "rewrite_skipped": False,
                "query_rewrite_enabled": QUERY_REWRITE_ENABLED,
            },
        )
    ]
    if rewrite_telemetry:
        telemetry_entries.append(rewrite_telemetry)

    return {
        "search_queries": queries,
        "execution_telemetry": telemetry_entries,
    }


def direct_replacement_analyzer(state: DiagnosticState) -> dict:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    queries = state.get("search_queries", [])
    if not queries:
        queries = [f"{state.get('extracted_part', 'component')} replacement procedure"]
    return {
        "search_queries": queries,
        "is_procedural": state.get("is_procedural", False),
        "execution_telemetry": [
            make_node_telemetry(
                "DirectReplacementAnalyzer",
                started_at,
                int((time.perf_counter() - t0) * 1000),
                metadata={
                    "replacement_identified": bool(state.get("extracted_part")),
                    "replacement_part": state.get("extracted_part", ""),
                },
            )
        ],
    }


@with_node_timeout(
    seconds=90,
    fallback={
        "retrieved_manuals": [],
        "retrieved_image_map": {},
    },
)
def knowledge_retriever(state: DiagnosticState) -> dict:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    device_id = state.get("device_id", "Unknown")
    queries = state.get("search_queries", [])
    is_procedural = state.get("is_procedural", False)

    manuals_results = []
    # Enriched chunk records: each entry carries the chunk's images alongside
    # its text, page_start, sequence_index, and section_header so the
    # RepairPlanner can match steps to their source chunks via text overlap.
    ordered: list[dict] = []

    is_scoped = state.get("is_scoped_procedural", False)

    manual_scores: dict = {}
    pre_rerank_all: dict = {}
    manual_traces: list[dict] = []
    neighbor_expansion: dict | None = None
    # P3 telemetry enrichment fields
    retrieval_path: str = "semantic"
    fallback_to_walkthrough: bool = False
    dedup_count: int = 0
    document_selected: int | None = None

    if is_procedural and not is_scoped:
        # Full walkthrough: single call returns the relevant section in
        # reading order.  search_terms drives section-header filtering so
        # "replace screen" only returns display-replacement chunks, not
        # the entire manual.
        retrieval_path = "full_walkthrough"
        result = manual_search_tool.invoke(
            {
                "query": queries[0] if queries else "assembly",
                "device_id": device_id,
                "procedural": True,
                "search_terms": queries or None,
            }
        )
        if isinstance(result, dict):
            for chunk in result.get("chunks", []):
                ordered.append(
                    {
                        "id": str(chunk.get("id", "")),
                        "score": chunk.get("score"),
                        "reranker_score": chunk.get("reranker_score"),
                        "images": chunk.get("image_urls", []),
                        "text": chunk.get("block", ""),
                        "page_start": chunk.get("page_start"),
                        "sequence_index": chunk.get("sequence_index"),
                        "section_header": chunk.get("section_header"),
                    }
                )
            manual_scores.update(result.get("chunk_scores", {}))
            pre_rerank_all.update(result.get("pre_rerank_scores", {}))
            if result.get("retrieval_trace"):
                manual_traces.append(result["retrieval_trace"])
            result = result.get("markdown", str(result))
        if result and "Data not present" not in result:
            manuals_results.append(str(result))

    elif is_procedural and is_scoped:
        # Scoped procedural: use semantic search to find only the relevant
        # chunks for the specific step/section, then sort by reading order.
        retrieval_path = "scoped_procedural"
        selected_document_id: int | None = None
        if queries:
            first_query_emb = embed(text=queries[0])
            if first_query_emb:
                selected_document_id = select_best_document(first_query_emb, device_id)
                document_selected = selected_document_id

        top_queries = queries[:3]
        seen_chunk_texts: set[str] = set()
        with ThreadPoolExecutor(max_workers=3) as executor:
            manual_futures = [
                executor.submit(
                    manual_search_tool.invoke,
                    {
                        "query": q,
                        "device_id": device_id,
                        "document_id": selected_document_id,
                        "rerank_min_score": 5,
                        "rerank_top_k": 7,
                    },
                )
                for q in top_queries
            ]

            for future in as_completed(manual_futures):
                try:
                    res = future.result()
                except Exception as e:
                    logger.error(f"[KnowledgeRetriever] Scoped search failed: {e}")
                    continue
                if isinstance(res, dict):
                    for chunk in res.get("chunks", []):
                        chunk_key = chunk.get("block", "")[:80]
                        if chunk_key and chunk_key not in seen_chunk_texts:
                            seen_chunk_texts.add(chunk_key)
                            ordered.append(
                                {
                                    "id": str(chunk.get("id", "")),
                                    "score": chunk.get("score"),
                                    "reranker_score": chunk.get("reranker_score"),
                                    "images": chunk.get("image_urls", []),
                                    "text": chunk.get("block", ""),
                                    "page_start": chunk.get("page_start"),
                                    "sequence_index": chunk.get("sequence_index"),
                                    "section_header": chunk.get("section_header"),
                                }
                            )
                        else:
                            dedup_count += 1
                    manual_scores.update(res.get("chunk_scores", {}))
                    pre_rerank_all.update(res.get("pre_rerank_scores", {}))
                    if res.get("retrieval_trace"):
                        manual_traces.append(res["retrieval_trace"])
                    res_md = res.get("markdown", str(res))
                    if res_md and "Data not present" not in res_md:
                        manuals_results.append(str(res_md))

        # Neighbor expansion: include ±1 adjacent chunks for context
        if ordered and selected_document_id is not None:
            retrieved_seq = {
                c["sequence_index"]
                for c in ordered
                if c.get("sequence_index") is not None
            }
            neighbor_seq = set()
            for idx in retrieved_seq:
                neighbor_seq.add(idx - 1)
                neighbor_seq.add(idx + 1)
            needed_seq = {s for s in (neighbor_seq - retrieved_seq) if s >= 0}

            if needed_seq:
                neighbor_added = 0
                neighbor_duplicates_skipped = 0
                t_nb = time.time()
                neighbor_chunks = get_chunks_by_sequence_indices(
                    selected_document_id, needed_seq
                )
                neighbor_query_ms = int((time.time() - t_nb) * 1000)
                for nc in neighbor_chunks:
                    chunk_key = nc.get("text", "")[:80]
                    if chunk_key and chunk_key not in seen_chunk_texts:
                        seen_chunk_texts.add(chunk_key)
                        neighbor_added += 1
                        # Build markdown block for the neighbor chunk
                        nc_block = f"---[{nc['intuitive_name']} | doc: {nc['doc_path']} | Page {nc['pages']}]---\n{nc['text']}\n"
                        nc_image_urls: list[dict] = []
                        gcs_bucket = (
                            os.getenv("GCS_BUCKET_NAME", "")
                            .replace('"', "")
                            .replace("'", "")
                        )
                        for img in nc.get("images") or []:
                            img_path = img["path"]
                            full_url = (
                                img_path
                                if img_path.startswith("http")
                                else f"https://storage.googleapis.com/{gcs_bucket}/{img_path}"
                            )
                            nc_block += (
                                f"![{img.get('caption', 'Diagram')}]({full_url})\n"
                            )
                            nc_image_urls.append(
                                {
                                    "url": full_url,
                                    "caption": img.get("caption", "Diagram"),
                                }
                            )
                        ordered.append(
                            {
                                "id": str(nc.get("id", "")),
                                "score": None,
                                "reranker_score": None,
                                "images": nc_image_urls,
                                "text": nc_block,
                                "page_start": nc.get("page_start"),
                                "sequence_index": nc.get("sequence_index"),
                                "section_header": nc.get("section_header"),
                            }
                        )
                        manuals_results.append(nc_block)
                    else:
                        neighbor_duplicates_skipped += 1
                neighbor_expansion = {
                    "requested": len(needed_seq),
                    "query_ms": neighbor_query_ms,
                    "added": neighbor_added,
                    "duplicates_skipped": neighbor_duplicates_skipped,
                }
                logger.info(
                    f"[KnowledgeRetriever] Neighbor expansion: added {neighbor_added}, "
                    f"skipped {neighbor_duplicates_skipped} duplicates (seq_indices={sorted(needed_seq)})"
                )

        # Re-sort by reading order (semantic search returns by relevance)
        ordered.sort(
            key=lambda c: (c.get("sequence_index") or 9999, c.get("page_start") or 9999)
        )

        # Fallback: if scoped search yielded too few results, switch to
        # full walkthrough so the technician gets a useful response.
        if len(ordered) < 2:
            logger.warning(
                f"[KnowledgeRetriever] Scoped search returned {len(ordered)} chunks — "
                f"falling back to full walkthrough for device={device_id}"
            )
            ordered.clear()
            manuals_results.clear()
            manual_scores.clear()
            pre_rerank_all.clear()
            manual_traces.clear()
            neighbor_expansion = None
            is_scoped = False
            fallback_to_walkthrough = True
            retrieval_path = "full_walkthrough"
            result = manual_search_tool.invoke(
                {
                    "query": queries[0] if queries else "assembly",
                    "device_id": device_id,
                    "procedural": True,
                    "search_terms": queries or None,
                }
            )
            if isinstance(result, dict):
                for chunk in result.get("chunks", []):
                    ordered.append(
                        {
                            "id": str(chunk.get("id", "")),
                            "score": chunk.get("score"),
                            "images": chunk.get("image_urls", []),
                            "text": chunk.get("block", ""),
                            "page_start": chunk.get("page_start"),
                            "sequence_index": chunk.get("sequence_index"),
                            "section_header": chunk.get("section_header"),
                        }
                    )
                manual_scores.update(result.get("chunk_scores", {}))
                pre_rerank_all.update(result.get("pre_rerank_scores", {}))
                if result.get("retrieval_trace"):
                    manual_traces.append(result["retrieval_trace"])
                result = result.get("markdown", str(result))
            if result and "Data not present" not in result:
                manuals_results.append(str(result))

    else:
        # Enhancement 1: Two-stage document selection.
        # Embed the first query once and use summary nodes (level>=1) to
        # identify the single most relevant document before leaf retrieval.
        selected_document_id: int | None = None
        if queries:
            first_query_emb = embed(text=queries[0])
            if first_query_emb:
                selected_document_id = select_best_document(first_query_emb, device_id)
                document_selected = selected_document_id
            else:
                logger.warning(
                    "[KnowledgeRetriever] Failed to embed first query for "
                    "document selection — falling back to unrestricted search"
                )

        # Non-procedural: use up to 3 queries for similarity search (concurrent).
        # Pass selected_document_id (may be None for graceful fallback).
        top_queries = queries[:3]
        with ThreadPoolExecutor(max_workers=3) as executor:
            manual_futures = [
                executor.submit(
                    manual_search_tool.invoke,
                    {
                        "query": q,
                        "device_id": device_id,
                        "document_id": selected_document_id,
                        "rerank_top_k": 7,
                    },
                )
                for q in top_queries
            ]

            for future in as_completed(manual_futures):
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"[KnowledgeRetriever] Manual search failed: {e}")
                    continue
                if isinstance(result, dict):
                    for chunk in result.get("chunks", []):
                        ordered.append(
                            {
                                "id": str(chunk.get("id", "")),
                                "score": chunk.get("score"),
                                "reranker_score": chunk.get("reranker_score"),
                                "images": chunk.get("image_urls", []),
                                "text": chunk.get("block", ""),
                                "page_start": chunk.get("page_start"),
                                "sequence_index": chunk.get("sequence_index"),
                                "section_header": chunk.get("section_header"),
                            }
                        )
                    manual_scores.update(result.get("chunk_scores", {}))
                    pre_rerank_all.update(result.get("pre_rerank_scores", {}))
                    if result.get("retrieval_trace"):
                        manual_traces.append(result["retrieval_trace"])
                    result = result.get("markdown", str(result))
                if result and "Data not present" not in result:
                    manuals_results.append(str(result))

        # Enhancement 2: Prepend document-level summary context so
        # downstream nodes (RootCauseAnalyzer, RepairPlanner) see the
        # manual's structural overview, not just isolated leaf chunks.
        if selected_document_id is not None:
            summaries = get_document_summaries(selected_document_id, max_level=2)
            if summaries:
                overview_lines = ["--- MANUAL OVERVIEW ---"]
                for s in summaries:
                    header = s["section_header"]
                    level_label = "Chapter" if s["level"] >= 2 else "Section"
                    if header:
                        overview_lines.append(
                            f"[{level_label} Summary — {header} (pp. {s['pages']})]\n{s['text']}"
                        )
                    else:
                        overview_lines.append(
                            f"[{level_label} Summary (pp. {s['pages']})]\n{s['text']}"
                        )
                manuals_results.insert(0, "\n\n".join(overview_lines))
                logger.info(
                    f"[KnowledgeRetriever] Prepended {len(summaries)} summary "
                    f"nodes from doc_id={selected_document_id} as Manual Overview"
                )

    # Build retrieval scores (nested per-chunk dicts)
    retrieval_scores: dict = {"manual": manual_scores, "pre_rerank": pre_rerank_all}

    # Build combined retrieval trace
    if is_procedural and is_scoped:
        mode = "scoped_procedural"
    elif is_procedural:
        mode = "full_procedural"
    else:
        mode = "semantic"
    retrieval_trace: dict = {
        "mode": mode,
        "manual_queries": manual_traces,
    }
    if neighbor_expansion is not None:
        retrieval_trace["neighbor_expansion"] = neighbor_expansion

    image_map = {"ordered": ordered}
    total_imgs = sum(len(entry.get("images", [])) for entry in ordered)
    logger.info(
        f"[KnowledgeRetriever] Built image_map: {total_imgs} images across {len(ordered)} chunks"
    )

    # R1: Collect chunk IDs for feedback linkage
    chunk_ids = [entry.get("id") for entry in ordered if entry.get("id")]

    # Chunk grouping: reduce redundancy for downstream LLM calls
    groups = group_chunks(ordered)
    grouped_context = build_grouped_context(groups)
    logger.info(
        f"[KnowledgeRetriever] Chunk grouping: {len(ordered)} chunks -> "
        f"{len(groups)} groups, "
        f"{sum(len(g.representatives) for g in groups)} representatives"
    )

    retrieval_latency_ms = int((time.perf_counter() - t0) * 1000)
    parallel_query_count = (
        len(queries[:3]) if not (is_procedural and not is_scoped) else 1
    )
    result_dict = {
        "retrieved_manuals": manuals_results,
        "retrieved_image_map": image_map,
        "retrieved_chunk_ids": chunk_ids,
        "retrieval_scores": retrieval_scores,
        "retrieval_trace": retrieval_trace,
        "chunk_groups": groups,
        "grouped_manual_context": grouped_context,
        "execution_telemetry": [
            make_node_telemetry(
                "KnowledgeRetriever",
                started_at,
                retrieval_latency_ms,
                metadata={
                    "retrieval_trace": retrieval_trace,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "reranking_applied": any(
                        t.get("reranker_applied") for t in manual_traces
                    ),
                    "top_k": len(chunk_ids),
                    "chunk_count": len(ordered),
                    "query_text": state.get("search_queries", []),
                    "retrieved_chunk_ids": chunk_ids,
                    "relevance_scores": retrieval_scores,
                    "retrieval_path": retrieval_path,
                    "parallel_query_count": parallel_query_count,
                    "neighbor_expansion_applied": neighbor_expansion is not None,
                    "fallback_to_walkthrough": fallback_to_walkthrough,
                    "document_selected": document_selected,
                    "dedup_count": dedup_count,
                },
            )
        ],
    }

    # When scoped search fell back to full walkthrough, update state so
    # RepairPlanner uses FULL COVERAGE rules instead of SCOPED OVERRIDE.
    if is_procedural and state.get("is_scoped_procedural", False) and not is_scoped:
        result_dict["is_scoped_procedural"] = False

    return result_dict


@with_node_timeout(
    seconds=120,
    fallback={
        "root_causes": [
            "Root cause analysis timed out. Proceeding with available context."
        ],
    },
)
def root_cause_analyzer(state: DiagnosticState) -> dict:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    # Short-circuit for procedural queries — no fault to diagnose, skip the
    # expensive Gemini Pro call (saves 30-120s).
    if state.get("is_procedural", False):
        return {
            "root_causes": [
                "User requested procedural instructions. Proceeding to visual guide."
            ],
            "execution_telemetry": [
                make_node_telemetry(
                    "RootCauseAnalyzer",
                    started_at,
                    int((time.perf_counter() - t0) * 1000),
                    metadata={"skipped": "procedural", "procedural_shortcircuit": True},
                )
            ],
        }

    # Retrieval quality gate — short-circuit if no relevant context
    chunk_ids = state.get("retrieved_chunk_ids") or []
    raw_context = (
        state.get("grouped_manual_context")
        or "\n\n".join(state.get("retrieved_manuals", []))
        or ""
    )
    # Check retrieval similarity — low scores indicate out-of-KB query
    _MIN_COSINE = 0.62
    retrieval_scores = state.get("retrieval_scores") or {}
    max_cosine = 0.0
    for bucket in retrieval_scores.values():
        if isinstance(bucket, dict):
            for scores in bucket.values():
                if isinstance(scores, dict):
                    max_cosine = max(max_cosine, scores.get("cosine", 0) or 0)

    gate_reason = None
    if not chunk_ids or len(raw_context.strip()) < 50:
        gate_reason = "no_retrieval"
    elif max_cosine < _MIN_COSINE:
        gate_reason = "low_similarity"

    if gate_reason:
        return {
            "root_causes": ["No relevant documentation was found for this query."],
            "execution_telemetry": [
                make_node_telemetry(
                    "RootCauseAnalyzer",
                    started_at,
                    int((time.perf_counter() - t0) * 1000),
                    metadata={
                        "gate": gate_reason,
                        "chunk_count": len(chunk_ids),
                        "context_chars": len(raw_context.strip()),
                        "max_cosine": round(max_cosine, 4),
                    },
                )
            ],
        }

    user_msg = next(
        (m for m in reversed(state.get("messages", [])) if m.get("role") == "user"),
        None,
    )
    user_text = str(user_msg.get("content", "")) if user_msg else "No user message."
    context = raw_context or "No relevant documentation found."

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT_ROOT_CAUSE),
            HumanMessage(
                content=f"Device ID: {state.get('device_id')}\n\nDocumentation:\n{context}\n\nQuery:\n{user_text}"
            ),
        ]
    )
    writer = get_stream_writer()
    cause, telemetry = tracked_stream(
        llm_pro_langchain,
        prompt.format_messages(),
        "RootCauseAnalyzer",
        MODEL_PRO,
        writer=writer,
    )
    if cause is None:
        cause = "Error: Failed to generate root cause analysis."
    # L2-E05: Enrich with root cause analysis metadata
    telemetry["metadata"] = {
        **(telemetry.get("metadata") or {}),
        "root_cause_analysis": cause[:500] if cause else None,
        "procedural_shortcircuit": False,
    }
    return {
        "root_causes": [cause],
        "execution_telemetry": [telemetry],
    }


@with_node_timeout(seconds=60)
def followup_responder(state: DiagnosticState) -> dict:
    user_msg = next(
        (m for m in reversed(state.get("messages", [])) if m.get("role") == "user"),
        None,
    )
    user_text = str(user_msg.get("content", "")) if user_msg else "No message."

    previous_plan = ""
    for m in reversed(state.get("messages", [])[:-1]):
        if m.get("role") == "assistant" and isinstance(m.get("content"), dict):
            for e in m["content"].get("events", []):
                if "RepairPlanner" in e and e["RepairPlanner"].get("final_response"):
                    previous_plan = json.dumps(e["RepairPlanner"]["final_response"])
                    break
            if previous_plan:
                break

    context = "\n\n".join(state.get("retrieved_manuals", []))
    prompt = PROMPT_FOLLOWUP_RESPONDER.format(
        device_id=state.get("device_id"),
        previous_plan=previous_plan or "None",
        user_text=user_text,
        context_block=context,
    )

    response, telemetry = tracked_generate_with_retry(
        model=MODEL_FLASH,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.0
        ),
        node_name="FollowUpResponder",
    )
    if response is not None:
        try:
            res = parse_gemini_json(response.text)
        except Exception:
            res = {"answer": "Error generating response.", "suggested_follow_ups": []}
    else:
        res = {"answer": "Error generating response.", "suggested_follow_ups": []}

    res_id = str(uuid.uuid4())
    # P3 L2-E-NEW03: Enrich FollowUpResponder telemetry
    telemetry["metadata"] = {
        **(telemetry.get("metadata") or {}),
        "previous_plan_found": bool(previous_plan),
        "previous_plan_session_id": None,  # thread-level; not recoverable from messages
    }
    return {
        "final_response": res,
        "response_id": res_id,
        "messages": [
            {
                "role": "assistant",
                "content": {
                    "type": "timeline",
                    "events": [{"FollowUpResponder": {"final_response": res}}],
                },
            }
        ],
        "execution_telemetry": [telemetry],
    }


def _score_step_chunk(step_text: str, chunk: dict) -> float:
    """Score how well a repair step matches a retrieved chunk.

    Combines three deterministic signals:
      1. Text word overlap  (weight 0.55) — primary signal
      2. Page citation match (weight 0.25) — structural anchor
      3. Caption keyword overlap (weight 0.20) — supplementary
    """
    score = 0.0
    step_words = set(re.findall(r"\w{3,}", step_text.lower()))
    if not step_words:
        return score

    # Signal 1: text word overlap
    chunk_text = chunk.get("text", "")
    chunk_words = set(re.findall(r"\w{3,}", chunk_text.lower()))
    if chunk_words:
        overlap = len(step_words & chunk_words)
        union = len(step_words | chunk_words)
        text_score = overlap / union if union else 0.0
        score += 0.55 * min(text_score, 1.0)

    # Signal 2: page citation match (<cite ... page="N">)
    cited_pages: set[int] = set()
    for m in re.finditer(r'<cite[^>]+page=["\']([^"\']+)["\']', step_text):
        for p in re.findall(r"\d+", m.group(1)):
            cited_pages.add(int(p))
    chunk_page = chunk.get("page_start")
    if cited_pages and chunk_page is not None:
        if int(chunk_page) in cited_pages:
            score += 0.25

    # Signal 3: best caption keyword overlap
    images = chunk.get("images", [])
    if images:
        best_cap = 0.0
        for img in images:
            cap_words = set(re.findall(r"\w{3,}", img.get("caption", "").lower()))
            if cap_words:
                cap_overlap = len(cap_words & step_words)
                cap_union = len(cap_words | step_words)
                cap_score = cap_overlap / cap_union if cap_union else 0.0
                best_cap = max(best_cap, cap_score)
        score += 0.20 * min(best_cap, 1.0)

    return score


def _score_step_chunk_procedural(
    step_idx: int,
    n_steps: int,
    chunk_idx: int,
    n_chunks: int,
    step_text: str,
    chunk: dict,
) -> float:
    """Score step-chunk match for procedural mode using positional alignment.

    In procedural mode the LLM is instructed to emit steps in chunk reading
    order (FULL COVERAGE prompt rule).  Positional proximity is the primary
    signal, with page citation as a structural anchor and section-header
    matching to discriminate adjacent sections with overlapping vocabulary.
    """
    score = 0.0

    # Signal 1 (weight 0.50): positional proximity
    if n_steps > 1 and n_chunks > 1:
        step_pos = step_idx / (n_steps - 1)
        chunk_pos = chunk_idx / (n_chunks - 1)
        distance = abs(step_pos - chunk_pos)
        position_score = max(0.0, 1.0 - distance * 2.5)
    else:
        position_score = 1.0 if n_chunks == 1 else 0.5
    score += 0.50 * position_score

    # Signal 2 (weight 0.30): page citation match
    cited_pages: set[int] = set()
    for m in re.finditer(r'<cite[^>]+page=["\']([^"\']+)["\']', step_text):
        for p in re.findall(r"\d+", m.group(1)):
            cited_pages.add(int(p))
    chunk_page = chunk.get("page_start")
    if cited_pages and chunk_page is not None:
        if int(chunk_page) in cited_pages:
            score += 0.30

    # Signal 3 (weight 0.20): section header match
    chunk_header = (chunk.get("section_header") or "").lower().strip()
    if chunk_header:
        header_words = set(re.findall(r"\w{5,}", chunk_header))
        step_words = set(re.findall(r"\w{5,}", step_text.lower()))
        if header_words and step_words:
            overlap = len(header_words & step_words)
            header_score = overlap / len(header_words)
            score += 0.20 * min(header_score, 1.0)

    return score


@with_node_timeout(
    seconds=600,
    fallback={
        "repair_steps": ["Error: Repair plan generation timed out. Please try again."],
        "final_response": {
            "repair_steps": [
                "Error: Repair plan generation timed out. Please try again."
            ],
            "suggested_follow_ups": [],
        },
        "response_id": "timeout",
        "messages": [],
    },
)
def repair_planner(state: DiagnosticState) -> dict:
    device_id = state.get("device_id", "Unknown")
    root_causes = state.get("root_causes", [])
    raw_safety = state.get("safety_protocols", [])
    manuals = state.get("retrieved_manuals", [])

    # If we were called back by the safety judge (judge_feedback present),
    # bump the retry counter so the next route_from_safety_judge knows
    # not to loop again.
    judge_feedback = state.get("judge_feedback", "")
    safety_judge_retry = 1 if judge_feedback else 0

    rc_str = (
        root_causes[0]
        if root_causes
        else f"User requested direct replacement/repair for: {state.get('extracted_part', 'Target Component')}"
    )

    safety_block = ""
    if isinstance(raw_safety, list) and raw_safety:
        if isinstance(raw_safety[0], dict) and "instruction" in raw_safety[0]:
            # New SafetyProtocol format (from SafetyExtractor or DeterministicRuleChecker)
            lines = [
                f"- ⚠️ **{sp['severity']}:** {sp['instruction']}" for sp in raw_safety
            ]
            safety_block = "**MANDATORY SAFETY PROTOCOLS:**\n" + "\n".join(lines)
        elif isinstance(raw_safety[0], str):
            # Legacy string format
            safety_block = "**MANDATORY SAFETY PROTOCOLS:**\n" + "\n".join(
                f"- {p}" for p in raw_safety
            )
    elif isinstance(raw_safety, dict):
        # Legacy DeterministicRuleChecker dict format (historical sessions)
        rules = raw_safety.get("rules", [])
        if rules:
            lines = [f"- ⚠️ **{r['severity']}:** {r['text']}" for r in rules]
            safety_block = "**MANDATORY SAFETY PROTOCOLS:**\n" + "\n".join(lines)

    context_block = state.get("grouped_manual_context") or (
        "\n\n".join(manuals) if manuals else "No technical documentation retrieved."
    )

    # Tag chunks with [CHUNK-N] identifiers for LLM-controlled image
    # binding in full procedural mode (not scoped).
    image_data = state.get("retrieved_image_map", {})
    ordered_chunks_pre = image_data.get("ordered", [])
    is_full_proc = state.get("is_procedural", False) and not state.get(
        "is_scoped_procedural", False
    )
    if is_full_proc and ordered_chunks_pre:
        tagged_parts: list[str] = []
        for idx, chunk in enumerate(ordered_chunks_pre):
            tagged_parts.append(f"[CHUNK-{idx}]\n{chunk.get('text', '')}")
        context_block = "\n\n".join(tagged_parts)

    # --- INJECT MULTIMODAL JUDGE FEEDBACK ---
    judge_feedback = state.get("judge_feedback", "")
    feedback_block = ""
    if judge_feedback:
        chunk_ref_line = ""
        if is_full_proc and ordered_chunks_pre:
            chunk_ref_line = "\n> Correct any chunk_refs that pointed to the wrong diagram. Verify each step references the correct CHUNK-N for its images."
        feedback_block = f"""
> 🛑 **CRITICAL FEEDBACK FROM SAFETY JUDGE:**
> Your previous draft failed for the following reasons:
> {judge_feedback}
> YOU MUST FIX THESE ERRORS IN YOUR NEW PLAN. Do not repeat hallucinations. Ensure no critical steps or safety warnings from the diagrams/text are omitted.{chunk_ref_line}
"""

    # --- RLHF anti-example / preference-pair injection ---
    # Embed the root-cause / symptom and pull the closest matching memory rows.
    # Anti-examples warn the planner away from past mistakes; preference pairs
    # show it edits experts have already approved on similar symptoms.
    anti_example_block = ""
    anti_examples_injected = []
    ae_embedding = None
    equipment_type = state.get("equipment_type", "")
    try:
        from modules.embeddings import embed
        from core.crud import search_anti_examples

        symptom_text = rc_str
        ae_embedding = embed(text=symptom_text[:500])
        anti_examples = search_anti_examples(
            ae_embedding, equipment_type or "GENERAL", device_id=device_id
        )
        if anti_examples:
            ae_lines = ["## KNOWN INCORRECT APPROACHES — DO NOT RECOMMEND"]
            for ae in anti_examples:
                alt = (
                    f" Correct alternative: {ae['correct_alternative']}"
                    if ae.get("correct_alternative")
                    else ""
                )
                ae_lines.append(
                    f'- [{ae["severity"]}] For symptom "{ae["symptom_pattern"][:100]}": '
                    f'Do NOT recommend "{ae["incorrect_procedure"][:200]}".{alt}'
                )
            anti_example_block = "\n".join(ae_lines)
            anti_examples_injected = anti_examples
    except Exception as e:
        logger.debug(f"Anti-example retrieval failed: {e}")

    # R3: Preference pair injection
    preference_block = ""
    _pref_pair_count = 0
    try:
        from core.crud import search_preference_pairs

        if ae_embedding:
            pref_pairs = search_preference_pairs(
                ae_embedding, equipment_type or "GENERAL"
            )
            if pref_pairs:
                _pref_pair_count = len(pref_pairs)
                pp_lines = ["## PREFERRED APPROACHES (from past corrections)"]
                for pp in pref_pairs:
                    pp_lines.append(
                        f'- When symptom was "{pp["symptom_context"][:100]}": '
                        f'preferred approach was "{pp["preferred_output"][:200]}" '
                        f'(not "{pp["original_output"][:100]}")'
                    )
                preference_block = "\n".join(pp_lines)
    except Exception as e:
        logger.debug(f"Preference pair retrieval failed: {e}")

    prompt = f"""{SYSTEM_PROMPT_REPAIR_PLANNER}

Device ID: {device_id}

<RootCauseAnalysis>
{rc_str}
</RootCauseAnalysis>

{safety_block}

<RetrievedTechnicalDocumentation>
{context_block}
</RetrievedTechnicalDocumentation>
{feedback_block}
{anti_example_block}
{preference_block}
"""

    if is_full_proc and ordered_chunks_pre:
        prompt += """
FULL COVERAGE — PROCEDURAL WALKTHROUGH:
The retrieved chunks are in READING ORDER from the manual. You MUST:
1. Emit steps following CHUNK-0 → CHUNK-1 → CHUNK-2 → ... order exactly.
2. Cover EVERY chunk — do not skip any.
3. Do NOT reorder, group, or rearrange chunks. The chunk sequence IS the procedure sequence.
4. The ORDERING RULE above does NOT apply here — chunk order IS procedural order.

CHUNK-IMAGE BINDING:
Each retrieved chunk above is tagged [CHUNK-N]. In your JSON output each step MUST
include a "chunk_refs" array listing which CHUNK-N identifiers that step corresponds
to. This controls which diagrams appear with each step.
Generate the repair plan in JSON format:
{
  "steps": [{"text": "**REMOVE** the cover...", "chunk_refs": ["CHUNK-0"]}, ...],
  "suggested_follow_ups": ["Question 1?", ...]
}
If a step does not correspond to any chunk, use an empty array for chunk_refs.
"""
    else:
        prompt += """
Generate the repair plan in JSON format:
{
  "steps": ["Step 1...", "Step 2..."],
  "suggested_follow_ups": ["Question 1?", "Question 2?", "Question 3?"]
}
"""

    if state.get("is_scoped_procedural", False):
        prompt += """
SCOPED PROCEDURAL OVERRIDE:
The retrieved context covers ONLY the specific step or section the technician asked about.
Emit steps ONLY for the retrieved chunks — do NOT pad with steps from other sections.
The FULL COVERAGE rule does NOT apply here. Cover each retrieved chunk thoroughly but stay
strictly within the scoped content. Present steps in the order the chunks appear.
"""

    step_writer = get_stream_writer()
    rp_telemetry: dict = {}

    try:
        # MULTIMODAL PLANNER: AI physically looks at the diagrams while drafting
        contents_payload = [prompt] + extract_gcs_uris(context_block)

        plan_schema = (
            ProceduralRepairPlan
            if (is_full_proc and ordered_chunks_pre)
            else SimpleRepairPlan
        )

        raw_text, rp_telemetry = tracked_stream_with_writer(
            model=MODEL_FLASH,
            contents=contents_payload,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=plan_schema,
                temperature=0.0,
                http_options=types.HttpOptions(timeout=120_000),
            ),
            node_name="RepairPlanner",
            writer=step_writer,
        )

        res = parse_gemini_json(raw_text)
        suggested_follow_ups = res.get("suggested_follow_ups", [])

        llm_chunk_refs: list[list[int]] | None = None
        raw_steps = res.get("steps", [])

        if is_full_proc and ordered_chunks_pre:
            # Parse chunk_refs from LLM output (procedural image binding)
            repair_steps_list = []
            llm_chunk_refs = []
            for step_obj in raw_steps:
                if isinstance(step_obj, dict):
                    repair_steps_list.append(str(step_obj.get("text", "")))
                    indices: list[int] = []
                    for ref in step_obj.get("chunk_refs", []):
                        m = re.match(r"CHUNK-(\d+)", str(ref))
                        if m:
                            idx = int(m.group(1))
                            if 0 <= idx < len(ordered_chunks_pre):
                                indices.append(idx)
                    llm_chunk_refs.append(indices)
                else:
                    repair_steps_list.append(str(step_obj))
                    llm_chunk_refs.append([])
        else:

            def _extract_step_text(step: object) -> str:
                if isinstance(step, str):
                    return step
                if isinstance(step, dict):
                    for key in ("text", "action", "step"):
                        val = step.get(key)
                        if isinstance(val, str) and val.strip():
                            return val
                return str(step)

            repair_steps_list = [_extract_step_text(step) for step in raw_steps]

        suggested_follow_ups = [
            str(q) if not isinstance(q, str) else q for q in suggested_follow_ups
        ]

    except Exception as e:
        logger.error(f"RepairPlanner generation failed: {e}")
        repair_steps_list = ["Error: Failed to generate repair plan. Please try again."]
        suggested_follow_ups = []
        llm_chunk_refs = None

    # Post-process: inject images from retrieved chunks.
    # In full-procedural mode, use LLM-emitted chunk_refs when available;
    # fall back to positional scoring.  Non-procedural uses Jaccard scoring.
    ordered_chunks = ordered_chunks_pre
    is_proc = state.get("is_procedural", False)

    if ordered_chunks and repair_steps_list:
        n_steps = len(repair_steps_list)
        n_chunks = len(ordered_chunks)
        SCORE_THRESHOLD = 0.10

        step_to_chunks: list[list[int]] = [[] for _ in range(n_steps)]

        if is_full_proc and llm_chunk_refs and any(refs for refs in llm_chunk_refs):
            # LLM-controlled: use chunk references the LLM emitted
            for i, refs in enumerate(llm_chunk_refs):
                step_to_chunks[i] = refs
            logger.info("[RepairPlanner] Using LLM chunk_refs for image binding")
        elif is_proc:
            # Positional scoring fallback for procedural mode
            scores: list[list[float]] = []
            for i, step in enumerate(repair_steps_list):
                row = [
                    _score_step_chunk_procedural(i, n_steps, j, n_chunks, step, chunk)
                    for j, chunk in enumerate(ordered_chunks)
                ]
                scores.append(row)

            if is_full_proc and llm_chunk_refs is not None:
                logger.warning(
                    "[RepairPlanner] LLM emitted no chunk_refs, falling back to positional scoring"
                )

            # Greedy forward pass with monotonic constraint
            min_chunk_idx = 0
            for i in range(n_steps):
                best_j: int | None = None
                best_score = SCORE_THRESHOLD
                for j in range(min_chunk_idx, n_chunks):
                    if scores[i][j] > best_score:
                        best_score = scores[i][j]
                        best_j = j
                if best_j is not None:
                    step_to_chunks[i].append(best_j)
                    min_chunk_idx = best_j
        else:
            # Non-procedural: Jaccard scoring + global best match per step
            scores = []
            for step in repair_steps_list:
                row = [_score_step_chunk(step, chunk) for chunk in ordered_chunks]
                scores.append(row)
            for i in range(n_steps):
                best_j = None
                best_score = SCORE_THRESHOLD
                for j in range(n_chunks):
                    if scores[i][j] > best_score:
                        best_score = scores[i][j]
                        best_j = j
                if best_j is not None:
                    step_to_chunks[i].append(best_j)

        # Orphan rescue: assign unmatched image-bearing chunks to nearest step
        assigned_chunks: set[int] = set()
        for chunk_list in step_to_chunks:
            assigned_chunks.update(chunk_list)

        for j in range(n_chunks):
            if j in assigned_chunks or not ordered_chunks[j].get("images"):
                continue
            best_step: int | None = None
            best_dist = float("inf")
            for i, chunk_list in enumerate(step_to_chunks):
                if not chunk_list:
                    continue
                for mapped_j in chunk_list:
                    dist = abs(j - mapped_j)
                    if is_proc and dist > 1:
                        continue
                    if dist < best_dist:
                        best_dist = dist
                        best_step = i
            if best_step is not None:
                step_to_chunks[best_step].append(j)

        # Build structured image sidecar per step, deduplicating by URL
        injected_count = 0
        used_urls: set[str] = set()
        step_images: list[list[dict]] = [[] for _ in range(n_steps)]

        for i, chunk_indices in enumerate(step_to_chunks):
            if not chunk_indices:
                continue
            for chunk_idx in chunk_indices:
                for img in ordered_chunks[chunk_idx].get("images", []):
                    if img["url"] not in used_urls:
                        used_urls.add(img["url"])
                        step_images[i].append(
                            {"url": img["url"], "caption": img["caption"]}
                        )
            if step_images[i]:
                injected_count += 1

        repair_steps_list = [
            {"text": step, "images": step_images[i]}
            for i, step in enumerate(repair_steps_list)
        ]

        all_urls = {img["url"] for ch in ordered_chunks for img in ch.get("images", [])}
        remaining = len(all_urls - used_urls)
        logger.info(
            f"[RepairPlanner] Injected images into {injected_count}/{n_steps} steps, "
            f"{remaining} unmatched images dropped (procedural={is_proc})"
        )
        if repair_steps_list:
            logger.info(
                f"[RepairPlanner] Sample step[0] (first 300 chars): {repr(repair_steps_list[0])[:300]}"
            )

        # Emit image sidecar so frontend can render images during streaming
        # (before the full RepairPlanner event flushes post-evaluation).
        image_sidecar: dict[str, list[dict]] = {}
        for i, step in enumerate(repair_steps_list):
            if isinstance(step, dict) and step.get("images"):
                image_sidecar[str(i)] = step["images"]
        if image_sidecar:
            step_writer({"type": "repair_images", "images": image_sidecar})

    # Assign stable step IDs (s-0, s-1, ...) to procedural steps.
    # suggested_follow_ups are excluded — they are not evaluated.
    for i, step in enumerate(repair_steps_list):
        if isinstance(step, dict):
            step["step_id"] = f"s-{i}"
        else:
            step_dict: dict = {"text": step, "step_id": f"s-{i}"}
            # Propagate chunk_refs so InlineEvaluator can scope faithfulness
            # context to only the chunks cited by each step.
            if llm_chunk_refs and i < len(llm_chunk_refs) and llm_chunk_refs[i]:
                step_dict["chunk_refs"] = [f"CHUNK-{idx}" for idx in llm_chunk_refs[i]]
            repair_steps_list[i] = step_dict

    final_dict = {
        "root_cause": rc_str,
        "repair_steps": repair_steps_list,
        "suggested_follow_ups": suggested_follow_ups,
    }

    response_id = str(uuid.uuid4())

    events = []
    if state.get("route"):
        events.append({"IntentRouter": {"route": state.get("route")}})

    sq = state.get("search_queries")
    if sq:
        if state.get("route") == "replace_part":
            events.append({"DirectReplacementAnalyzer": {"search_queries": sq}})
        else:
            events.append({"SymptomAnalyzer": {"search_queries": sq}})

    if state.get("retrieved_manuals"):
        events.append({"KnowledgeRetriever": {"status": "Complete"}})

    rc = state.get("root_causes")
    if rc:
        events.append({"RootCauseAnalyzer": {"root_causes": rc}})

    events.append(
        {
            "RepairPlanner": {
                "repair_steps": repair_steps_list,
                "final_response": final_dict,
                "response_id": response_id,
            }
        }
    )

    # R1: Save response-chunk links for feedback propagation
    try:
        from core.crud import save_response_chunk_links

        chunk_ids = state.get("retrieved_chunk_ids", [])
        if chunk_ids and response_id:
            save_response_chunk_links(response_id, chunk_ids)
    except Exception as e:
        logger.debug(f"Failed to save response-chunk links: {e}")

    # L2-E08: Enrich RepairPlanner telemetry with safety and plan metadata
    rules_list = []
    if isinstance(raw_safety, dict):
        rules_list = raw_safety.get("rules", [])
    elif isinstance(raw_safety, list):
        rules_list = raw_safety
    step_count = len(repair_steps_list) if repair_steps_list else 0

    # Determine image binding method used
    if is_full_proc and llm_chunk_refs and any(refs for refs in llm_chunk_refs):
        _binding_method = "llm_chunk_refs"
    elif is_proc:
        _binding_method = "positional"
    else:
        _binding_method = "jaccard"

    # Count orphan images rescued (unmatched image chunks assigned to nearest step)
    _orphan_rescued = 0
    if ordered_chunks and repair_steps_list:
        assigned_all: set[int] = set()
        for chunk_list in step_to_chunks:
            assigned_all.update(chunk_list)
        _orphan_rescued = sum(
            1
            for j in range(len(ordered_chunks))
            if j not in assigned_all and ordered_chunks[j].get("images")
        )

    rp_telemetry["metadata"] = {
        **(rp_telemetry.get("metadata") or {}),
        "safety_protocols_injected": bool(safety_block),
        "injected_protocol_count": len(rules_list),
        "plan_step_count": step_count,
        "temperature_used": 0.0,
        "judge_feedback_injected": bool(feedback_block),
        "anti_examples_count": len(anti_examples_injected),
        "preference_pairs_count": _pref_pair_count,
        "image_binding_method": _binding_method,
        "images_bound": injected_count if ordered_chunks and repair_steps_list else 0,
        "orphan_images_rescued": _orphan_rescued,
    }

    return {
        "repair_steps": repair_steps_list,
        "final_response": final_dict,
        "response_id": response_id,
        "safety_judge_retry_count": safety_judge_retry,
        "anti_examples_injected": anti_examples_injected,
        "messages": [
            {
                "role": "assistant",
                "content": {
                    "type": "timeline",
                    "response_id": response_id,
                    "events": events,
                },
            }
        ],
        "execution_telemetry": [rp_telemetry],
    }


# --- REPAIR PLANNER RESPONSE SCHEMAS ---
class ProceduralStep(BaseModel):
    text: str = Field(
        description="The step instruction text with markdown bold for action verbs."
    )
    chunk_refs: list[str] = Field(
        default_factory=list,
        description="CHUNK-N identifiers this step corresponds to.",
    )


class ProceduralRepairPlan(BaseModel):
    steps: list[ProceduralStep] = Field(
        description="Ordered repair steps with chunk references."
    )
    suggested_follow_ups: list[str] = Field(
        default_factory=list,
        description="Follow-up questions the technician might ask.",
    )


class SimpleRepairPlan(BaseModel):
    steps: list[str] = Field(description="Ordered repair steps as plain text strings.")
    suggested_follow_ups: list[str] = Field(
        default_factory=list,
        description="Follow-up questions the technician might ask.",
    )


# =============================================================================
# DUAL-LOOP RAGAS EVALUATION
# =============================================================================


def _build_metric_row(
    state: dict,
    inline_eval: dict,
    thresholds: dict,
    config,
    attempt_number: int,
) -> dict:
    scores = inline_eval.get("scores", {})
    return {
        "thread_id": (config or {}).get("configurable", {}).get("thread_id", "unknown"),
        "response_id": state.get("response_id") or "unknown",
        "device_id": state.get("device_id") or "Unknown",
        "user_id": None,  # filled in by agent_manager if available; node has no auth context
        "badge": inline_eval.get("badge", "gray"),
        "quality_badge": inline_eval.get("quality_badge"),
        "attempt_number": attempt_number,
        "sync_faithfulness": scores.get("faithfulness"),
        "sync_answer_relevance": scores.get("answer_relevance"),
        # context_relevance + completeness are owned by ShadowEvaluator and
        # patched into the row via update_evaluation_metric_async.
        "sync_context_relevance": None,
        "sync_completeness": None,
        "sync_reasons": inline_eval.get("reasons", {}),
        "sync_duration_ms": inline_eval.get("duration_ms"),
        "thresholds_snapshot": thresholds,
        "step_verdicts": inline_eval.get("step_verdicts"),
        "safety_survival": inline_eval.get("safety_survival"),
    }


_RE_UUID = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)
_RE_SEQ = re.compile(r"Seq\s+(\d+)")
_RE_PAGE = re.compile(r"Page\s+(\d+)")
_RE_COMPACT_PAGE = re.compile(r":(\d+)$")


def _resolve_source_chunk_ids(
    step_verdicts: list[dict],
    ordered_chunks: list[dict],
) -> None:
    """Resolve LLM-emitted source_chunk_ids to actual chunk UUIDs in-place.

    The InlineEvaluator LLM outputs context-block headers as
    source_chunk_ids because that's how chunks appear in the prompt.
    Three observed formats:
      1. ``"doc.pdf | doc: manuals/doc.pdf | Page 101 | Seq 21"``
      2. ``"manuals/doc.pdf | Page 14"``
      3. ``"manuals/doc.pdf:1"``  (compact doc_path:page)

    These fail the FK constraint on ``response_chunk_link.chunk_id``.
    Build lookups from Seq index and page number to the real UUID,
    then replace non-UUID references in each verdict.
    """
    # Build lookup maps: seq_index → UUID (1:1), page_start → first UUID
    seq_map: dict[int, str] = {}
    page_map: dict[int, str] = {}
    for chunk in ordered_chunks:
        cid = chunk.get("id")
        if not cid:
            continue
        seq = chunk.get("sequence_index")
        if seq is not None:
            seq_map[int(seq)] = str(cid)
        ps = chunk.get("page_start")
        if ps is not None:
            page_map.setdefault(int(ps), str(cid))

    resolved_count = 0
    for verdict in step_verdicts:
        new_ids: list[str] = []
        for ref in verdict.get("source_chunk_ids") or []:
            if _RE_UUID.match(ref):
                new_ids.append(ref)
                continue
            # Try Seq index first (most reliable — unique per chunk)
            seq_match = _RE_SEQ.search(ref)
            if seq_match and int(seq_match.group(1)) in seq_map:
                new_ids.append(seq_map[int(seq_match.group(1))])
                resolved_count += 1
                continue
            # Try "Page N" pattern (maps to first chunk on that page)
            page_match = _RE_PAGE.search(ref)
            if page_match and int(page_match.group(1)) in page_map:
                new_ids.append(page_map[int(page_match.group(1))])
                resolved_count += 1
                continue
            # Try compact "path:page" pattern
            compact_match = _RE_COMPACT_PAGE.search(ref)
            if compact_match and int(compact_match.group(1)) in page_map:
                new_ids.append(page_map[int(compact_match.group(1))])
                resolved_count += 1
                continue
            # Unresolvable — drop to avoid FK violation
        verdict["source_chunk_ids"] = new_ids

    if resolved_count:
        logger.info(
            f"[InlineEvaluator] Resolved {resolved_count} source_chunk_ids "
            f"from header format to UUIDs"
        )


@with_node_timeout(
    seconds=200,
    fallback={
        "inline_eval": {
            "scores": {k: None for k in ("faithfulness", "answer_relevance")},
            "reasons": {
                k: "evaluator timeout" for k in ("faithfulness", "answer_relevance")
            },
            "badge": "gray",
            "quality_badge": "gray",
            "duration_ms": None,
            "error": "InlineEvaluator timeout",
            "response_id": None,
        },
        # Fail-open: a timed-out evaluator must not block the response or
        # trigger the retry path (which would burn another RepairPlanner round).
        "inline_eval_passed": True,
        "flagged_for_review": False,
        "is_plan_safe": True,
    },
)
def inline_evaluator(state: DiagnosticState, config: RunnableConfig) -> dict:
    """Sync judge — artifact-type-aware evaluation dispatch.

    Classifies the response into an artifact type and dispatches the
    appropriate evaluation strategy:

      REPAIR_PLAN   → full eval (whole-plan + per-step faithfulness);
                      badge gates retry loop
      INFORMATIONAL → answer-level eval (faithfulness + answer_relevance
                      on root_cause text); badge is informational only
      FOLLOW_UP     → answer-level eval against prior retrieval context;
                      badge is informational only
      NON_RETRIEVAL → skip eval entirely; no badge, no DB row

    Routing contract:
      - Always routes to END (no retry loop).
      - Red/gray badges on REPAIR_PLAN artifacts flag the thread for admin review.
      - All other artifact types are informational only.
    """
    start = time.time()

    final_response = state.get("final_response") or {}

    question = extract_question(state)
    context = extract_context(state)

    # --- Artifact classification ---
    artifact_type = classify_artifact(final_response, bool(context))
    response_id = state.get("response_id") or "unknown"

    # NON_RETRIEVAL: nothing to evaluate against — skip entirely.
    if artifact_type == ArtifactType.NON_RETRIEVAL:
        logger.info(
            f"[InlineEvaluator] skipping: artifact_type={artifact_type.value}, "
            f"response_id={response_id}"
        )
        return {
            "inline_eval": {
                "scores": {k: None for k in INLINE_METRIC_KEYS},
                "reasons": {k: "not_applicable" for k in INLINE_METRIC_KEYS},
                "badge": None,
                "quality_badge": None,
                "duration_ms": int((time.time() - start) * 1000),
                "error": None,
                "response_id": response_id,
            },
            "inline_eval_passed": True,
            "flagged_for_review": False,
            "is_plan_safe": True,
        }

    is_plan_artifact = artifact_type == ArtifactType.REPAIR_PLAN
    evaluable_text = extract_evaluable_text(final_response, artifact_type)
    # Faithfulness uses manual-only context (no safety docs) — it measures
    # "is the plan grounded in the technical manuals", not safety compliance.
    # Falls back to full context only when no manuals exist (degraded but
    # avoids an empty-context evaluation that would score everything 0).
    manual_context = extract_manual_context(state) or context

    # Scope faithfulness context to reduce tokens: full-proc mode scopes to
    # cited chunks via chunk_refs, non-full-proc scopes to representative
    # chunks (what the LLM was actually shown via grouped_manual_context).
    is_full_proc = state.get("is_procedural", False) and not state.get(
        "is_scoped_procedural", False
    )
    img_map_for_scope = state.get("retrieved_image_map") or {}
    all_ordered_chunks = img_map_for_scope.get("ordered", [])
    # Compute once — reused for both whole-plan and per-step scoping.
    cited_ids: set[str] = set()
    if is_plan_artifact and is_full_proc and all_ordered_chunks:
        cited_ids = extract_cited_chunk_ids(
            final_response.get("repair_steps", []),
            all_ordered_chunks,
        )
    if is_plan_artifact and all_ordered_chunks:
        if is_full_proc:
            if cited_ids:
                manual_context = filter_context_by_chunk_ids(
                    all_ordered_chunks, cited_ids, fallback_to_all=True
                )
                logger.info(
                    f"[InlineEvaluator] faithfulness context scoped: "
                    f"{len(cited_ids)} cited / {len(all_ordered_chunks)} total"
                )
        else:
            grouped_ctx = state.get("grouped_manual_context")
            if grouped_ctx:
                manual_context = grouped_ctx
                logger.info(
                    "[InlineEvaluator] faithfulness context scoped to "
                    "grouped representatives"
                )

    # Inline tier uses a tighter image cap than shadow — see eval_logic.py
    # for why (50 s node budget vs 180 s shadow budget).
    image_parts = extract_image_parts(state, max_images=MAX_INLINE_EVAL_IMAGE_PARTS)

    # Fail-open with gray + clear reason when inputs are missing OR when an
    # upstream node returned its error fallback. Scoring an "Error: …" string
    # burns LLM calls for no signal.
    plan_failed = is_plan_failure(evaluable_text)
    if not (question and evaluable_text and context) or plan_failed:
        if plan_failed:
            error_msg = "plan generation failed upstream — nothing to evaluate"
            reason_per_metric = "skipped: plan generation failed"
        else:
            error_msg = (
                f"missing: question={bool(question)} "
                f"evaluable_text={bool(evaluable_text)} context={bool(context)}"
            )
            reason_per_metric = "skipped: missing input"
        inline_eval = {
            "scores": {k: None for k in INLINE_METRIC_KEYS},
            "reasons": {k: reason_per_metric for k in INLINE_METRIC_KEYS},
            "badge": "gray",
            "quality_badge": "gray",
            "duration_ms": int((time.time() - start) * 1000),
            "error": error_msg,
            "response_id": response_id,
        }
        # Persist even the gray skip so the analytics dashboard reflects it.
        try:
            insert_evaluation_metric(
                _build_metric_row(state, inline_eval, get_thresholds(), config, 1)
            )
        except Exception as e:
            logger.error(f"InlineEvaluator persist (skip path) failed: {e}")
        # Don't block the user on an upstream failure — let the response
        # surface (with whatever error fallback the upstream node produced).
        return {
            "inline_eval": inline_eval,
            "inline_eval_passed": True,
            "flagged_for_review": False,
            "is_plan_safe": True,
        }

    thresholds = get_thresholds()

    scores: dict = {}
    reasons: dict = {}
    eval_telemetry: list[dict] = []
    step_verdicts: list[dict] | None = None

    # --- Pre-compute per-step inputs ---
    eval_steps: list[dict] = []
    if is_plan_artifact:
        raw_steps = final_response.get("repair_steps") or []
        for idx, s in enumerate(raw_steps):
            if isinstance(s, dict):
                eval_steps.append(
                    {"step_id": s.get("step_id", f"s-{idx}"), "text": s.get("text", "")}
                )
            else:
                eval_steps.append({"step_id": f"s-{idx}", "text": str(s)})

    # --- Single unified Flash-Lite call ---
    from pipeline.eval_logic import evaluate_unified

    result, telem = evaluate_unified(
        plan=evaluable_text,
        steps=eval_steps,
        context=manual_context,
        question=question,
        image_parts=image_parts,
    )
    # P3 L2-E10: Enrich InlineEvaluator telemetry with classification metadata
    _context_method = (
        "chunk_refs" if (is_full_proc and cited_ids) else "grouped_manual_context"
    )
    telem["metadata"] = {
        **(telem.get("metadata") or {}),
        "evaluation_category": artifact_type.value,
        "context_scoping_method": _context_method,
    }
    eval_telemetry.append(telem)

    # Derive scores from unified result
    if result.steps:
        faithful_count = sum(1 for s in result.steps if s.faithful)
        scores["faithfulness"] = faithful_count / len(result.steps)
        reasons["faithfulness"] = f"{faithful_count}/{len(result.steps)} steps grounded"
        step_verdicts = [v.model_dump() for v in result.steps]
    else:
        scores["faithfulness"] = None
        reasons["faithfulness"] = (
            result.answer_relevance_reason or "no steps to evaluate"
        )

    scores["answer_relevance"] = result.answer_relevance_score
    reasons["answer_relevance"] = result.answer_relevance_reason

    # Resolve LLM-emitted source_chunk_ids (header-format references like
    # "robotA.pdf | doc: manuals/robotA.pdf | Page 1 | Seq 0") to actual
    # chunk UUIDs so the FK constraint on response_chunk_link.chunk_id
    # succeeds.
    if step_verdicts:
        img_map = state.get("retrieved_image_map") or {}
        ordered_for_resolve = img_map.get("ordered", [])
        if ordered_for_resolve:
            _resolve_source_chunk_ids(step_verdicts, ordered_for_resolve)

    # Persist per-step attribution to response_chunk_link
    if step_verdicts and response_id != "unknown":
        try:
            from core.crud import save_response_chunk_links

            save_response_chunk_links(
                response_id=response_id,
                chunk_ids=[],
                step_attributions=step_verdicts,
                retrieval_scores=state.get("retrieval_scores"),
            )
        except Exception as e:
            logger.debug(f"Failed to save enriched chunk links: {e}")

    # Exclude metrics that produced no score — either timeout or
    # intentional null (vague query → answer_relevance N/A). Badge is
    # derived from the remaining metrics so a single missing metric
    # doesn't force gray and trigger a false-positive retry.
    available_keys = tuple(
        k
        for k in INLINE_METRIC_KEYS
        if reasons.get(k) != "timeout" and scores.get(k) is not None
    )
    badge_keys = available_keys if available_keys else INLINE_METRIC_KEYS
    quality_badge = compute_quality_badge(
        scores, thresholds, keys=badge_keys, step_verdicts=step_verdicts
    )
    badge = quality_badge
    duration_ms = int((time.time() - start) * 1000)

    # --- Safety protocol survival check (informational, does not gate) ---
    safety_survival = None
    safety_protocols = state.get("safety_protocols", [])
    extraction_meta = state.get("safety_extraction_metadata") or {}
    if (
        is_plan_artifact
        and safety_protocols
        and isinstance(safety_protocols, list)
        and isinstance(safety_protocols[0], dict)
        and extraction_meta.get("status") != "extraction_failed"
    ):
        plan_text_lower = evaluable_text.lower()
        present_count = 0
        missing_ids = []
        for sp in safety_protocols:
            instruction = sp.get("instruction", "")
            # Check if key phrases from the instruction appear in the plan.
            # Include words >3 chars (catches LOTO, PPE-adjacent terms).
            key_words = [
                w.lower().strip(".,;:!?()")
                for w in instruction.split()
                if len(w.strip(".,;:!?()")) > 3
            ]
            # Consider present if ≥40% of key words appear in the plan
            if key_words:
                matches = sum(1 for w in key_words if w in plan_text_lower)
                if matches / len(key_words) >= 0.4:
                    present_count += 1
                else:
                    missing_ids.append(sp.get("id", ""))
            else:
                present_count += 1  # Very short instructions assumed covered
        safety_survival = {
            "total": len(safety_protocols),
            "present_in_plan": present_count,
            "missing": missing_ids,
        }

    inline_eval = {
        "scores": scores,
        "reasons": reasons,
        "badge": badge,
        "quality_badge": quality_badge,
        "duration_ms": duration_ms,
        "error": None,
        "response_id": response_id,
        "step_verdicts": step_verdicts,
        "safety_survival": safety_survival,
        # L2-E10: Spec-aligned aliases for audit telemetry
        "per_step_verdicts": step_verdicts,
        "aggregate_faithfulness": scores.get("faithfulness"),
        "badge_color": badge,
        "thresholds_snapshot": thresholds,
    }

    # Badge is informational — no retry loop. Red/gray badges on plan
    # artifacts flag the thread for admin review but do not block the user.
    inline_eval_passed = True
    if is_plan_artifact:
        flagged_for_review = quality_badge in ("red", "gray")
    else:
        flagged_for_review = False

    try:
        insert_evaluation_metric(
            _build_metric_row(state, inline_eval, thresholds, config, 1)
        )
    except Exception as e:
        logger.error(f"InlineEvaluator persist failed: {e}")

    if flagged_for_review:
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        if thread_id:
            try:
                set_thread_flag(thread_id, True)
            except Exception as e:
                logger.warning(f"set_thread_flag failed for {thread_id}: {e}")
            try:
                from core.crud import create_review_task
                from core.database import SessionLocal
                from core.models import DiagnosticThread, ReviewTaskSource
                from modules.review_priority import compute_priority

                with SessionLocal() as db:
                    t = (
                        db.query(DiagnosticThread)
                        .filter(DiagnosticThread.id == thread_id)
                        .first()
                    )
                    flagged_by = t.user_id if t else None

                if flagged_by is None:
                    logger.warning(
                        f"No user_id found for thread {thread_id}; creating review task without owner"
                    )

                priority = compute_priority(
                    ReviewTaskSource.AUTO_EVAL,
                    metrics=scores,
                    badge=badge,
                    quality_badge=quality_badge,
                )
                formatted = (
                    ", ".join(
                        f"{k}={v:.2f}"
                        for k, v in (scores or {}).items()
                        if v is not None
                    )
                    or "no scores captured"
                )
                create_review_task(
                    thread_id=thread_id,
                    flagged_by=flagged_by,
                    source=ReviewTaskSource.AUTO_EVAL,
                    priority=priority,
                    response_id=response_id,
                    trigger_reason=f"InlineEvaluator red badge. Scores: {formatted}",
                    payload_snapshot={
                        "badge": badge,
                        "quality_badge": quality_badge,
                        "scores": scores,
                        "reasons": reasons,
                        "response_id": response_id,
                    },
                    reason="Auto-generated by InlineEvaluator",
                )
            except Exception as e:
                logger.warning(f"InlineEvaluator auto review-task failed: {e}")

    logger.info(
        f"[InlineEvaluator] quality={quality_badge} badge={badge} "
        f"passed={inline_eval_passed} flagged={flagged_for_review} "
        f"duration_ms={duration_ms} image_parts={len(image_parts)} scores={scores}"
    )

    # P3 L2-E10: Final telemetry enrichment with safety survival + review task metadata
    _safety_survival_result = None
    if safety_survival:
        total_sp = safety_survival.get("total", 0)
        present_sp = safety_survival.get("present_in_plan", 0)
        _safety_survival_result = present_sp / total_sp if total_sp > 0 else None
    if eval_telemetry:
        eval_telemetry[0]["metadata"] = {
            **(eval_telemetry[0].get("metadata") or {}),
            "safety_survival_check_result": _safety_survival_result,
            "flagged_for_review": flagged_for_review,
            "review_task_created": flagged_for_review,
        }

    return {
        "inline_eval": inline_eval,
        "inline_eval_passed": inline_eval_passed,
        "flagged_for_review": flagged_for_review,
        "is_plan_safe": quality_badge in ("green", "yellow", "gray"),  # legacy field
        "execution_telemetry": eval_telemetry,
    }


@with_node_timeout(
    seconds=10,
    fallback={
        "route": "troubleshoot",
    },
)
def unsafe_method_gate(state: DiagnosticState) -> dict:
    """Emit a safety warning for the technician, then re-dispatch via the
    user's underlying intent. The IntentRouter has already classified;
    no second LLM call here."""
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    warning = (
        state.get("safety_warning")
        or "An unsafe method was detected. Please use approved tools."
    )
    corrected = state.get("corrected_intent") or "troubleshoot"

    writer = get_stream_writer()
    if writer:
        writer(
            {
                "UnsafeMethodGate": {
                    "safety_warning": warning,
                    "corrected_intent": corrected,
                }
            }
        )

    return {
        "route": corrected,
        "execution_telemetry": [
            make_node_telemetry(
                "UnsafeMethodGate",
                started_at,
                int((time.perf_counter() - t0) * 1000),
                metadata={
                    "verdict": "unsafe",
                    "route_before": "unsafe_method",
                    "route_after": corrected,
                    "corrected_intent": corrected,
                    "safety_warning_emitted": True,
                    "safety_warning_preview": warning[:200],
                },
            )
        ],
    }


# ── R10: Deterministic Rule Checker ──────────────────────────────────
def deterministic_rule_checker(state: DiagnosticState) -> dict:
    """Check for deterministic rule matches before running the RAG pipeline."""
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    device_id = state.get("device_id", "Unknown")
    equipment_type = state.get("equipment_type", "GENERAL").upper()
    messages = state.get("messages", [])
    last_user_msg = next(
        (m for m in reversed(messages) if m.get("role") == "user"), None
    )
    if not last_user_msg:
        return {
            "deterministic_override": None,
            "execution_telemetry": [
                make_node_telemetry(
                    "DeterministicRuleChecker",
                    started_at,
                    int((time.perf_counter() - t0) * 1000),
                    metadata={
                        "skipped": True,
                        "rule_matched": False,
                        "matched_rule_id": None,
                        "equipment_type_filter": equipment_type,
                    },
                )
            ],
        }

    user_text = ""
    content = last_user_msg.get("content")
    if isinstance(content, str):
        user_text = content
    elif isinstance(content, list):
        user_text = " ".join(
            item.get("data", "") for item in content if item.get("type") == "text"
        )

    try:
        from core.crud import match_deterministic_rules

        rule = match_deterministic_rules(user_text, device_id, equipment_type)
    except Exception as e:
        logger.warning(f"Deterministic rule check failed: {e}")
        return {
            "deterministic_override": None,
            "execution_telemetry": [
                make_node_telemetry(
                    "DeterministicRuleChecker",
                    started_at,
                    int((time.perf_counter() - t0) * 1000),
                    error=str(e),
                    metadata={
                        "rule_matched": False,
                        "matched_rule_id": None,
                        "equipment_type_filter": equipment_type,
                    },
                )
            ],
        }

    if not rule:
        return {
            "deterministic_override": None,
            "execution_telemetry": [
                make_node_telemetry(
                    "DeterministicRuleChecker",
                    started_at,
                    int((time.perf_counter() - t0) * 1000),
                    metadata={
                        "rule_matched": False,
                        "matched_rule_id": None,
                        "equipment_type_filter": equipment_type,
                    },
                )
            ],
        }

    res_id = str(uuid.uuid4())
    final_response = {
        "repair_steps": [rule["override_response"]],
        "suggested_follow_ups": [],
    }
    safety_protocols = []
    if rule.get("override_safety_protocols"):
        from pipeline.safety_schemas import (
            SafetyCategory,
            SafetyProtocol,
            SafetySeverity,
        )

        safety_protocols = [
            SafetyProtocol(
                id="sp_rule_001",
                category=SafetyCategory.GENERAL,
                severity=SafetySeverity.WARNING,
                instruction=rule["override_safety_protocols"],
                source_chunk_ids=[],
                source_text_excerpt="Deterministic rule override",
                applies_throughout=True,
            ).model_dump()
        ]

    logger.info(
        f"[DeterministicRuleChecker] Rule {rule['id']} matched (keywords={rule['trigger_keywords']}, source={rule['source']})"
    )

    return {
        "final_response": final_response,
        "safety_protocols": safety_protocols,
        "response_id": res_id,
        "deterministic_override": rule,
        "messages": [
            {
                "role": "assistant",
                "content": {
                    "type": "timeline",
                    "response_id": res_id,
                    "events": [
                        {
                            "DeterministicOverride": {
                                "final_response": final_response,
                                "rule_id": rule["id"],
                                "source": rule["source"],
                                "response_id": res_id,
                            }
                        }
                    ],
                },
            }
        ],
        "execution_telemetry": [
            make_node_telemetry(
                "DeterministicRuleChecker",
                started_at,
                int((time.perf_counter() - t0) * 1000),
                metadata={
                    "rule_matched": True,
                    "rule_id": rule["id"],
                    "matched_rule_id": rule["id"],
                    "equipment_type_filter": equipment_type,
                    "trigger_keywords": rule.get("trigger_keywords", []),
                    "rule_source": rule.get("source"),
                    "rule_priority": rule.get("priority"),
                },
            )
        ],
    }


# human_gatekeeper removed: the interrupt-based human-in-the-loop was replaced
# by the inline retry gate + flagged_for_review admin alert. Yellow/red badges
# no longer pause the graph; red after retry exhaustion sets flagged_for_review
# on the diagnostic_threads row, which surfaces as a red dot in the admin
# history dashboard for expert follow-up.


# ---------------------------------------------------------------------------
# SafetyExtractor v2 — extraction-only node (before RepairPlanner)
# ---------------------------------------------------------------------------


def safety_extractor(state: DiagnosticState) -> dict:
    """Extract manufacturer safety precautions via dual-path retrieval.

    Path A: reuse KnowledgeRetriever's already-retrieved context.
    Path B: supplementary safety-augmented retrieval query.
    Merged context is passed to Gemini 2.5 Flash for structured extraction.

    On failure, returns empty protocols and metadata with status=extraction_failed.
    Never blocks the pipeline.
    """
    from pipeline.safety_schemas import (
        ExtractionConfidence,
        ExtractionStatus,
        SafetyExtractionMetadata,
        SafetyExtractionResponse,
    )
    from pipeline.tools import safety_augmented_search
    from prompts.system_prompts import SYSTEM_PROMPT_SAFETY_EXTRACTOR_V2

    t0 = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()

    device_id = state.get("device_id", "Unknown")
    equipment_type = state.get("equipment_type", "heavy_industrial")
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_query = content
            elif isinstance(content, list):
                user_query = " ".join(
                    p.get("data", p.get("text", ""))
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            break

    # Failure fallback
    def _fail(error_msg: str) -> dict:
        logger.warning(f"[SafetyExtractor] {error_msg}")
        metadata = SafetyExtractionMetadata(
            status=ExtractionStatus.EXTRACTION_FAILED,
            extraction_confidence=ExtractionConfidence.LOW,
            no_safety_content_found=True,
        ).model_dump()
        return {
            "safety_protocols": [],
            "safety_extraction_metadata": metadata,
            "execution_telemetry": [
                make_node_telemetry(
                    "SafetyExtractor",
                    started_at,
                    int((time.perf_counter() - t0) * 1000),
                    metadata={"error": error_msg},
                )
            ],
        }

    # --- Path A: chunks from KnowledgeRetriever ---
    retrieved_manuals = state.get("retrieved_manuals", [])
    chunk_groups = state.get("chunk_groups", [])

    # Build a lookup of Path A chunk IDs + text
    path_a_chunks: list[dict] = []
    if chunk_groups:
        for group in chunk_groups:
            if hasattr(group, "chunks"):
                for c in group.chunks:
                    path_a_chunks.append(
                        {"id": getattr(c, "id", ""), "text": getattr(c, "text", "")}
                    )
            elif isinstance(group, dict):
                for c in group.get("chunks", []):
                    cid = c.get("id", "") if isinstance(c, dict) else ""
                    ctxt = c.get("text", "") if isinstance(c, dict) else ""
                    path_a_chunks.append({"id": cid, "text": ctxt})

    # --- Path B: safety-augmented supplementary retrieval ---
    path_b_chunks = safety_augmented_search(
        equipment_type=equipment_type,
        device_id=device_id,
        top_k=10,
    )

    # Merge & deduplicate by chunk ID
    seen_ids: set[str] = set()
    merged_chunks: list[dict] = []
    for c in path_a_chunks:
        cid = c.get("id", "")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            merged_chunks.append(c)
    for c in path_b_chunks:
        cid = c.get("id", "")
        if cid and cid not in seen_ids:
            seen_ids.add(cid)
            merged_chunks.append(c)

    if not merged_chunks and not retrieved_manuals:
        return _fail("No context available for safety extraction")

    # Build context string for extraction prompt
    context_parts: list[str] = []
    for i, c in enumerate(merged_chunks):
        chunk_id = c.get("id", f"chunk_{i}")
        text = c.get("text", "")
        if text:
            context_parts.append(f"[CHUNK_ID: {chunk_id}]\n{text}")

    # If no structured chunks but we have raw manual text, use that
    if not context_parts and retrieved_manuals:
        for i, manual_text in enumerate(retrieved_manuals):
            context_parts.append(f"[CHUNK_ID: manual_{i}]\n{manual_text}")

    context_str = "\n\n---\n\n".join(context_parts)

    prompt = SYSTEM_PROMPT_SAFETY_EXTRACTOR_V2.format(
        equipment_type=equipment_type,
        user_query=user_query[:500],
        context=context_str,
    )

    # Call Gemini 2.5 Flash with structured output
    try:
        response, telemetry_entry = tracked_generate_with_retry(
            model=MODEL_FLASH,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                http_options=types.HttpOptions(timeout=60_000),
            ),
            node_name="SafetyExtractor",
        )
    except Exception as e:
        return _fail(f"LLM call failed: {e}")

    if response is None:
        return _fail("LLM returned None response")

    # Parse structured output
    try:
        raw = parse_gemini_json(response.text)
        extraction = SafetyExtractionResponse.model_validate(raw)
    except Exception as e:
        return _fail(f"Failed to parse extraction response: {e}")

    protocols = [p.model_dump() for p in extraction.safety_protocols]
    metadata = extraction.extraction_metadata.model_dump()

    # Override status based on extraction results
    if not protocols:
        metadata["no_safety_content_found"] = True
        metadata["status"] = ExtractionStatus.SUCCESS.value
    else:
        metadata["no_safety_content_found"] = False
        metadata["status"] = ExtractionStatus.SUCCESS.value

    metadata["total_chunks_scanned"] = len(merged_chunks)

    logger.info(
        f"[SafetyExtractor] Extracted {len(protocols)} protocols "
        f"(confidence={metadata.get('extraction_confidence')}, "
        f"chunks_scanned={len(merged_chunks)})"
    )

    telemetry_list = [telemetry_entry] if telemetry_entry else []
    telemetry_list.append(
        make_node_telemetry(
            "SafetyExtractor",
            started_at,
            int((time.perf_counter() - t0) * 1000),
            metadata={
                "protocols_extracted": len(protocols),
                "chunks_scanned": len(merged_chunks),
                "path_a_count": len(path_a_chunks),
                "path_b_count": len(path_b_chunks),
            },
        )
    )

    return {
        "safety_protocols": protocols,
        "safety_extraction_metadata": metadata,
        "execution_telemetry": telemetry_list,
    }
