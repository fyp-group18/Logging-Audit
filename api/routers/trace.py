# backend/api/routers/trace.py
"""Traceability endpoint — full AI reasoning chain for a diagnostic session."""

import logging

from fastapi import APIRouter, HTTPException

from api.agent_manager import agent_manager
from api.schemas import (
    TraceAuditOverhead,
    TraceChunkDetail,
    TraceConformance,
    TraceCrossLayerMetrics,
    TraceEvaluationScores,
    TraceFollowUp,
    TraceIntentRouting,
    TraceL1Metrics,
    TraceNodeExecution,
    TraceQueryProcessing,
    TraceQueryResult,
    TraceRaptorNode,
    TraceRaptorTree,
    TraceResponse,
    TraceRetrievalPipeline,
    TraceFeedback,
)
from core import crud, models
from core.database import SessionLocal
from core.models import (
    AgentExecutionLog,
    DocumentChunkMultimodal,
    DocumentMultimodal,
    ResponseChunkLink,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/trace", tags=["Trace"])


def _build_chunk_detail(
    chunk: DocumentChunkMultimodal,
    doc_name: str | None = None,
    similarity_score: float | None = None,
    reranker_score: int | None = None,
    grounding_label: str | None = None,
    step_index: int | None = None,
) -> TraceChunkDetail:
    return TraceChunkDetail(
        chunk_id=chunk.id,
        text_preview=(chunk.text or "")[:200],
        level=chunk.level or 0,
        document_id=chunk.document_id,
        document_name=doc_name,
        pages=chunk.pages,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_header=chunk.section_header,
        similarity_score=similarity_score,
        reranker_score=reranker_score,
        grounding_label=grounding_label,
        step_index=step_index,
        source_chunk_ids=chunk.source_chunk_ids,
        has_safety_content=chunk.has_safety_content,
    )


@router.get("/{response_id}", response_model=TraceResponse)
def get_trace(
    response_id: str,
):
    """Full traceability view for a diagnostic response."""
    # Resolve thread_id from evaluation_metrics or response_chunk_link
    thread_id: str | None = None
    with SessionLocal() as db:
        eval_row = (
            db.query(models.EvaluationMetric)
            .filter(models.EvaluationMetric.response_id == response_id)
            .first()
        )
        if eval_row:
            thread_id = eval_row.thread_id

    if not thread_id:
        # Try to find from execution logs
        with SessionLocal() as db:
            log_row = (
                db.query(models.AgentExecutionLog.thread_id)
                .filter(models.AgentExecutionLog.response_id == response_id)
                .first()
            )
            if log_row:
                thread_id = log_row.thread_id

    if not thread_id:
        raise HTTPException(
            status_code=404,
            detail=f"No session found for response_id={response_id}",
        )

    # Get thread info
    device_id: str | None = None
    with SessionLocal() as db:
        thread = (
            db.query(models.DiagnosticThread)
            .filter(models.DiagnosticThread.id == thread_id)
            .first()
        )
        if thread:
            device_id = thread.device_id

    # --- Section 1: Query Processing ---
    query_processing = TraceQueryProcessing()
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = agent_manager.graph.get_state(config)
        state = snapshot.values if snapshot and snapshot.values else {}

        # Extract user query from messages
        messages = state.get("messages", [])
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "human":
                query_processing.user_query = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        search_queries = state.get("search_queries", [])
        if isinstance(search_queries, list):
            query_processing.search_queries = search_queries

        # Extract retrieval trace for per-query results
        retrieval_trace = state.get("retrieval_trace", {})
        if isinstance(retrieval_trace, dict):
            for query_text, chunk_ids in retrieval_trace.items():
                if isinstance(chunk_ids, list):
                    query_processing.per_query_results.append(
                        TraceQueryResult(
                            query=query_text,
                            chunks=[],  # filled after chunk data is loaded
                        )
                    )
    except Exception:
        logger.warning("Failed to load checkpoint state for trace %s", response_id, exc_info=True)

    # --- Section 2: Retrieval Pipeline (chunk attribution) ---
    retrieval_pipeline = TraceRetrievalPipeline()
    retrieved_chunk_ids: set[str] = set()
    chunk_details_map: dict[str, TraceChunkDetail] = {}
    try:
        with SessionLocal() as db:
            # Detect procedural retrieval to choose ordering strategy
            kr_log = (
                db.query(AgentExecutionLog)
                .filter(
                    AgentExecutionLog.response_id == response_id,
                    AgentExecutionLog.node_name == "KnowledgeRetriever",
                )
                .first()
            )
            is_procedural_retrieval = bool(
                kr_log
                and kr_log.metadata_
                and kr_log.metadata_.get("retrieval_path", "").startswith("full_")
            )

            order_clause = (
                DocumentChunkMultimodal.sequence_index.asc().nulls_last()
                if is_procedural_retrieval
                else ResponseChunkLink.similarity_score.desc().nullslast()
            )

            # Get all response_chunk_links with chunk + doc details
            links = (
                db.query(
                    ResponseChunkLink,
                    DocumentChunkMultimodal,
                    DocumentMultimodal.intuitive_name,
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
                .order_by(order_clause)
                .all()
            )

            reranked: list[TraceChunkDetail] = []
            for link, chunk, doc_name in links:
                detail = _build_chunk_detail(
                    chunk,
                    doc_name=doc_name,
                    similarity_score=link.similarity_score,
                    reranker_score=link.reranker_score,
                    grounding_label=link.grounding_label,
                    step_index=link.step_index,
                )
                reranked.append(detail)
                retrieved_chunk_ids.add(chunk.id)
                chunk_details_map[chunk.id] = detail

            retrieval_pipeline.reranked_chunks = reranked
            retrieval_pipeline.chunks_after_filtering = len(reranked)

            # Build chunk allocation (which node consumed which chunks)
            allocation: dict[str, list[str]] = {}
            for detail in reranked:
                if detail.step_index is not None:
                    key = f"step_{detail.step_index}"
                elif detail.grounding_label:
                    key = detail.grounding_label
                else:
                    key = "unattributed"
                allocation.setdefault(key, []).append(detail.chunk_id)
            retrieval_pipeline.chunk_allocation = allocation

            # Get retrieval scores + pre-rerank candidates from checkpoint state
            try:
                config = {"configurable": {"thread_id": thread_id}}
                snapshot = agent_manager.graph.get_state(config)
                state = snapshot.values if snapshot and snapshot.values else {}
                retrieval_scores = state.get("retrieval_scores", {})
                if isinstance(retrieval_scores, dict):
                    retrieval_pipeline.reranker_model = retrieval_scores.get("reranker_model")
                    if ft := retrieval_scores.get("filtering_threshold"):
                        retrieval_pipeline.filtering_threshold = ft

                    # Compute filtered-out chunks: pre-rerank candidates minus final kept
                    pre_rerank = retrieval_scores.get("pre_rerank", {})
                    if pre_rerank:
                        retrieval_pipeline.total_chunks_retrieved = len(pre_rerank)
                        filtered_out_ids = set(pre_rerank.keys()) - retrieved_chunk_ids
                        if filtered_out_ids:
                            filtered_chunks = (
                                db.query(
                                    DocumentChunkMultimodal,
                                    DocumentMultimodal.intuitive_name,
                                )
                                .join(
                                    DocumentMultimodal,
                                    DocumentChunkMultimodal.document_id == DocumentMultimodal.id,
                                )
                                .filter(DocumentChunkMultimodal.id.in_(list(filtered_out_ids)))
                                .all()
                            )
                            for chunk, doc_name in filtered_chunks:
                                scores = pre_rerank.get(chunk.id, {})
                                retrieval_pipeline.filtered_out_chunks.append(
                                    _build_chunk_detail(
                                        chunk,
                                        doc_name=doc_name,
                                        similarity_score=scores.get("cosine"),
                                    )
                                )
            except Exception:
                pass

            if not retrieval_pipeline.total_chunks_retrieved:
                retrieval_pipeline.total_chunks_retrieved = len(reranked)

    except Exception:
        logger.warning("Failed to load chunk attribution for trace %s", response_id, exc_info=True)

    # Fill per-query chunk details
    for qr in query_processing.per_query_results:
        try:
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = agent_manager.graph.get_state(config)
            state = snapshot.values if snapshot and snapshot.values else {}
            retrieval_trace = state.get("retrieval_trace", {})
            if isinstance(retrieval_trace, dict):
                chunk_ids = retrieval_trace.get(qr.query, [])
                for cid in chunk_ids if isinstance(chunk_ids, list) else []:
                    if cid in chunk_details_map:
                        qr.chunks.append(chunk_details_map[cid])
        except Exception:
            pass

    # --- Section 3: Intent Routing ---
    intent_routing = TraceIntentRouting()
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = agent_manager.graph.get_state(config)
        state = snapshot.values if snapshot and snapshot.values else {}

        intent_routing.intent = state.get("route")

        # Extract node execution sequence from logs (exclude edge decisions,
        # background dispatchers, and normalize suffixed names like
        # "InlineEvaluator:unified" → "InlineEvaluator").
        exec_logs = crud.get_execution_logs(thread_id)
        node_sequence = []
        seen = set()
        for log in exec_logs:
            name = log.get("node_name", "")
            if not name or name.startswith("edge:") or log.get("operation_type") == "ingestion_step":
                continue
            # Strip suffixes (e.g. "InlineEvaluator:unified" → "InlineEvaluator")
            base_name = name.split(":")[0]
            # Skip background dispatchers that aren't pipeline nodes
            if base_name.endswith("_dispatch"):
                continue
            if base_name not in seen:
                node_sequence.append(base_name)
                seen.add(base_name)
        intent_routing.node_sequence = node_sequence

        # Extract routing rationale from IntentRouter execution log metadata
        for log in exec_logs:
            if log.get("node_name") == "IntentRouter":
                meta = log.get("metadata") or {}
                parts = []
                if ci := meta.get("classified_intent"):
                    parts.append(f"Classified intent: {ci}")
                if et := meta.get("equipment_type"):
                    parts.append(f"Equipment type: {et}")
                if meta.get("is_procedural"):
                    parts.append("Query is procedural")
                rd = meta.get("routing_decision")
                if rd and rd != meta.get("classified_intent"):
                    parts.append(f"Routing decision overridden to: {rd}")
                if parts:
                    intent_routing.routing_rationale = " | ".join(parts)
                break

        # Check for unsafe method gate info
        for log in exec_logs:
            if log.get("node_name") == "UnsafeMethodGate":
                meta = log.get("metadata") or {}
                intent_routing.unsafe_method_gate = {
                    "flagged": True,
                    "reason": meta.get("reason"),
                    "redirect_to": meta.get("redirect_to"),
                }
                break

        # Check for deterministic rule match
        for log in exec_logs:
            if log.get("node_name") == "DeterministicRuleChecker":
                meta = log.get("metadata") or {}
                if meta.get("rule_matched"):
                    intent_routing.deterministic_rule_match = {
                        "rule_id": meta.get("rule_id"),
                        "rule_keywords": meta.get("trigger_keywords"),
                        "response_preview": (meta.get("override_response") or "")[:200],
                    }
                break
    except Exception:
        logger.warning("Failed to load intent routing for trace %s", response_id, exc_info=True)

    # --- Section 4: Agent Reasoning Chain ---
    agent_reasoning: list[TraceNodeExecution] = []
    try:
        exec_logs = crud.get_execution_logs(thread_id)
        for log in exec_logs:
            if log.get("response_id") and log["response_id"] != response_id:
                continue
            sa = log.get("started_at")
            ca = log.get("completed_at")
            agent_reasoning.append(
                TraceNodeExecution(
                    node_name=log.get("node_name", ""),
                    model=log.get("model"),
                    started_at=sa.isoformat() if hasattr(sa, "isoformat") else sa,
                    completed_at=ca.isoformat() if hasattr(ca, "isoformat") else ca,
                    duration_ms=log.get("latency_ms"),
                    input_tokens=log.get("input_tokens"),
                    output_tokens=log.get("output_tokens"),
                    metadata=log.get("metadata"),
                    error=log.get("error"),
                )
            )
    except Exception:
        logger.warning("Failed to load execution logs for trace %s", response_id, exc_info=True)

    # --- Section 5: Evaluation Scores ---
    evaluation_scores = TraceEvaluationScores()
    eval_metric = None
    try:
        eval_metric = crud.get_evaluation_metric_by_response_id(response_id)
        if eval_metric:
            evaluation_scores = TraceEvaluationScores(
                sync_faithfulness=eval_metric.sync_faithfulness,
                sync_answer_relevance=eval_metric.sync_answer_relevance,
                sync_context_relevance=eval_metric.sync_context_relevance,
                sync_completeness=eval_metric.sync_completeness,
                async_faithfulness=eval_metric.async_faithfulness,
                async_answer_relevance=eval_metric.async_answer_relevance,
                async_context_relevance=eval_metric.async_context_relevance,
                async_completeness=eval_metric.async_completeness,
                async_safety_coverage=eval_metric.async_safety_coverage,
                safety_survival=eval_metric.safety_survival,
                badge=eval_metric.badge,
                quality_badge=eval_metric.quality_badge,
                safety_badge=eval_metric.safety_badge,
                step_verdicts=eval_metric.step_verdicts,
                thresholds_snapshot=eval_metric.thresholds_snapshot,
            )
    except Exception:
        logger.warning("Failed to load evaluation scores for trace %s", response_id, exc_info=True)

    # --- Cross-Layer Metrics ---
    cross_layer = TraceCrossLayerMetrics()
    try:
        from audit.cross_layer_metrics import (
            compute_provenance_score,
            compute_safety_provenance_completeness,
        )

        prov = compute_provenance_score(thread_id)
        prov_meta = prov.get("metadata", {})
        cross_layer.provenance_total_steps = prov_meta.get("total_steps", 0)
        cross_layer.provenance_traceable_steps = prov_meta.get("traceable_steps", 0)
        cross_layer.provenance_score = prov.get("value")

        exec_logs_raw = crud.get_execution_logs(thread_id)
        has_safety = any(
            log.get("node_name") == "SafetyExtractor" for log in exec_logs_raw
        )
        cross_layer.has_safety_extractor = has_safety
        if has_safety:
            sp = compute_safety_provenance_completeness(thread_id)
            sp_meta = sp.get("metadata", {})
            cross_layer.safety_provenance_total = sp_meta.get("safety_chunks_total", 0)
            cross_layer.safety_provenance_complete = sp_meta.get(
                "complete_chain_count", 0
            )
            cross_layer.safety_provenance_score = sp.get("value")

        if eval_metric:
            agreement_rows = []
            for metric_name in [
                "faithfulness",
                "answer_relevance",
                "context_relevance",
                "completeness",
            ]:
                sync_val = getattr(eval_metric, f"sync_{metric_name}", None)
                async_val = getattr(eval_metric, f"async_{metric_name}", None)
                delta = (
                    round(sync_val - async_val, 4)
                    if sync_val is not None and async_val is not None
                    else None
                )
                agreement_rows.append(
                    {
                        "metric": metric_name,
                        "inline_sync": sync_val,
                        "shadow_async": async_val,
                        "delta": delta,
                        "flagged": abs(delta) > 0.15 if delta is not None else False,
                    }
                )
            cross_layer.evaluator_agreement = agreement_rows
    except Exception:
        logger.warning(
            "Failed cross-layer metrics for %s", response_id, exc_info=True
        )

    # --- Trace Conformance ---
    trace_conformance = TraceConformance()
    try:
        from audit.process_mining import match_trace_variant

        tc = match_trace_variant(intent_routing.node_sequence)
        trace_conformance.conforming = tc["conforming"]
        trace_conformance.variant_label = tc.get("variant_label")
        if tc.get("deviating_edge"):
            trace_conformance.deviating_edge = {
                "from": tc["deviating_edge"][0],
                "to": tc["deviating_edge"][1],
            }
    except Exception:
        logger.warning(
            "Failed trace conformance for %s", response_id, exc_info=True
        )

    # --- Section 6: Follow-up Chain ---
    follow_up = TraceFollowUp()
    # Follow-up detection would require checkpoint state traversal
    # which is beyond current scope — left as empty for now

    # --- Section 7: RAPTOR Tree Context ---
    raptor_trees: list[TraceRaptorTree] = []
    try:
        if retrieved_chunk_ids:
            with SessionLocal() as db:
                # Get unique document IDs and build per-doc chunk mapping
                doc_chunk_rows = (
                    db.query(
                        DocumentChunkMultimodal.document_id,
                        DocumentChunkMultimodal.id,
                    )
                    .filter(DocumentChunkMultimodal.id.in_(list(retrieved_chunk_ids)))
                    .all()
                )
                chunks_by_doc: dict[int, set[str]] = {}
                for did, cid in doc_chunk_rows:
                    if did is not None:
                        chunks_by_doc.setdefault(did, set()).add(cid)
                doc_id_set = set(chunks_by_doc.keys())

                for doc_id in doc_id_set:
                    # Get doc name
                    doc = (
                        db.query(DocumentMultimodal.intuitive_name)
                        .filter(DocumentMultimodal.id == doc_id)
                        .first()
                    )
                    doc_name = doc.intuitive_name if doc else None

                    # Get all chunks for this document (all RAPTOR levels)
                    all_chunks = (
                        db.query(DocumentChunkMultimodal)
                        .filter(DocumentChunkMultimodal.document_id == doc_id)
                        .order_by(
                            DocumentChunkMultimodal.level.asc(),
                            DocumentChunkMultimodal.sequence_index.asc().nullslast(),
                        )
                        .all()
                    )

                    tree_nodes = []
                    for c in all_chunks:
                        tree_nodes.append(
                            TraceRaptorNode(
                                chunk_id=c.id,
                                level=c.level or 0,
                                text_preview=(c.text or "")[:150],
                                source_chunk_ids=c.source_chunk_ids,
                                was_retrieved=c.id in retrieved_chunk_ids,
                                pages=c.pages,
                                section_header=c.section_header,
                            )
                        )

                    # Compute L1 metrics for this document
                    l1_metrics = None
                    try:
                        from audit.l1_metrics import (
                            compute_kb_reconstructability_per_chunk,
                            compute_leaf_coverage,
                            compute_manifest_integrity,
                            get_ingestion_metadata,
                        )

                        lc = compute_leaf_coverage(doc_id)
                        lc_meta = lc.get("metadata", {})
                        levels = lc_meta.get("levels", {})

                        mi = compute_manifest_integrity(doc_id)
                        mi_meta = mi.get("metadata", {})

                        doc_retrieved = chunks_by_doc.get(doc_id, set())
                        kb = compute_kb_reconstructability_per_chunk(
                            doc_id, doc_retrieved
                        )
                        kb_meta = kb.get("metadata", {})

                        ing = get_ingestion_metadata(doc_id)

                        max_depth = max(levels.keys()) if levels else 0
                        leaf_count = lc_meta.get("leaf_count", 0)
                        total_count = lc_meta.get("total_chunks", 0)

                        l1_metrics = TraceL1Metrics(
                            leaf_coverage_d=lc.get("value"),
                            leaf_count=leaf_count,
                            summary_count=total_count - leaf_count,
                            total_count=total_count,
                            max_depth=max_depth,
                            kb_reconstructability=kb.get("value"),
                            chunks_with_provenance=kb_meta.get(
                                "chunks_with_provenance", 0
                            ),
                            chunks_total_retrieved=kb_meta.get(
                                "chunks_total_retrieved", 0
                            ),
                            manifest_valid=mi_meta.get("valid"),
                            manifest_entry_count=mi_meta.get("entries_checked", 0),
                            manifest_first_broken_at=mi_meta.get("first_broken_at"),
                            ingestion_timestamp=ing.get("ingestion_timestamp"),
                            embedding_model=ing.get("embedding_model"),
                            chunk_size=ing.get("chunk_size"),
                            chunk_overlap=ing.get("chunk_overlap"),
                            clusters_per_level=ing.get("clusters_per_level", []),
                        )
                    except Exception:
                        logger.warning(
                            "Failed L1 metrics for doc %d in trace %s",
                            doc_id,
                            response_id,
                            exc_info=True,
                        )

                    raptor_trees.append(
                        TraceRaptorTree(
                            document_id=doc_id,
                            document_name=doc_name,
                            nodes=tree_nodes,
                            l1_metrics=l1_metrics,
                        )
                    )
    except Exception:
        logger.warning("Failed to build RAPTOR trees for trace %s", response_id, exc_info=True)

    # --- Feedback ---
    feedback = TraceFeedback()
    try:
        with SessionLocal() as db:
            fb_data = crud.get_feedback_for_thread(db, thread_id)
            feedback.response_feedbacks = [
                {
                    "response_id": rf.response_id,
                    "feedback_type": rf.feedback_type.value if rf.feedback_type else None,
                    "correction_text": rf.correction_text,
                    "created_at": rf.created_at.isoformat() if rf.created_at else None,
                }
                for rf in fb_data.get("response_feedbacks", [])
            ]
            feedback.step_feedbacks = [
                {
                    "response_id": sf.response_id,
                    "step_index": sf.step_index,
                    "action": sf.action,
                    "correction_text": sf.correction_text,
                    "created_at": sf.created_at.isoformat() if sf.created_at else None,
                }
                for sf in fb_data.get("step_feedbacks", [])
            ]
    except Exception:
        logger.warning("Failed to load feedback for trace %s", response_id, exc_info=True)

    # --- Audit Overhead ---
    audit_overhead = TraceAuditOverhead()
    try:
        total_ms = sum(
            log.duration_ms for log in agent_reasoning if log.duration_ms
        )
        audit_overhead.total_execution_ms = total_ms if total_ms > 0 else None
        audit_overhead.instrumented = False
    except Exception:
        pass

    return TraceResponse(
        thread_id=thread_id,
        response_id=response_id,
        device_id=device_id,
        query_processing=query_processing,
        retrieval_pipeline=retrieval_pipeline,
        intent_routing=intent_routing,
        agent_reasoning=agent_reasoning,
        evaluation_scores=evaluation_scores,
        cross_layer_metrics=cross_layer,
        follow_up=follow_up,
        raptor_trees=raptor_trees,
        trace_conformance=trace_conformance,
        audit_overhead=audit_overhead,
        feedback=feedback,
    )
