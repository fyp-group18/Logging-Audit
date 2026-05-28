import datetime
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from typing import Dict, Any

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

from api.schemas import DiagnoseRequest
from core.crud import (
    bulk_insert_execution_logs,
    get_evaluations_for_thread,
    get_thread_terminal_state,
)
from core.database import DATABASE_URL_RAW, with_db_retry
from pipeline.telemetry import (
    collect_edge_telemetry,
    set_edge_telemetry_thread_id,
    state_fingerprint,
)
from pipeline.workflow import workflow

logger = logging.getLogger(__name__)


def _extract_initial_user_message(messages: list) -> dict | None:
    """Pull the first user message's text + image data URIs out of the
    checkpointed messages list. Refresh-recovery re-issues the diagnostic
    with this payload when the prior run never persisted a reply."""
    if not messages:
        return None
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or msg.get("type")
        if role not in ("user", "human"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return {"text": content, "images": []}
        if isinstance(content, list):
            text_parts: list[str] = []
            images: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    val = part.get("data") or part.get("text") or ""
                    if val:
                        text_parts.append(val)
                elif ptype == "image":
                    val = part.get("data") or ""
                    if val:
                        images.append(val)
                elif ptype == "image_url":
                    url_field = part.get("image_url")
                    if isinstance(url_field, dict):
                        val = url_field.get("url", "")
                    else:
                        val = url_field or ""
                    if val:
                        images.append(val)
            return {"text": " ".join(text_parts), "images": images}
        return None
    return None


def _json_default(o):
    # Dataclasses (e.g. ChunkGroup from knowledge_retriever) are not JSON-encodable
    # by default — convert via asdict. Sets (e.g. ChunkGroup.chunk_ids) become lists.
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, (set, frozenset)):
        return list(o)
    return str(o)


def _persist_execution_logs(
    trace_id: str, thread_id: str, response_id: str | None, records: list[dict]
) -> None:
    """Batch-persist telemetry records to agent_execution_logs.

    When a safety-judge retry occurred (RepairPlanner appears twice in the
    records list), the retry iteration gets a separate chain_id so each
    chain is independently hash-verifiable.
    """
    if not records:
        return

    def _make_rows(recs: list[dict]) -> list[dict]:
        return [
            {"trace_id": trace_id, "thread_id": thread_id,
             "response_id": response_id, **r}
            for r in recs
        ]

    # Detect safety-judge retry: second occurrence of RepairPlanner
    rp_indices = [
        i for i, r in enumerate(records)
        if r.get("node_name") == "RepairPlanner"
    ]
    if len(rp_indices) >= 2:
        split_at = rp_indices[1]
        retry_chain_id = str(uuid.uuid4())
        # Persist both batches in a single DB session to avoid partial writes
        from core.database import SessionLocal
        from core.hash_chain import GENESIS_HASH, compute_record_hash
        from core.models import AgentExecutionLog
        from core.config import AUDIT_LOGGING_ENABLED

        if not AUDIT_LOGGING_ENABLED:
            return
        with SessionLocal() as db:
            all_rows = []
            for batch_records, cid in [
                (records[:split_at], trace_id),
                (records[split_at:], retry_chain_id),
            ]:
                prev_hash = GENESIS_HASH
                for entry in _make_rows(batch_records):
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
                    all_rows.append(
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
                            chain_id=cid,
                        )
                    )
                    prev_hash = rec_hash
            db.add_all(all_rows)
            db.commit()
        logger.info(
            f"Persisted {split_at}+{len(records) - split_at} execution log rows "
            f"for trace={trace_id} (retry chain={retry_chain_id})"
        )
    else:
        bulk_insert_execution_logs(_make_rows(records), chain_id=trace_id)
        logger.info(f"Persisted {len(records)} execution log rows for trace={trace_id}")


class AgentManager:
    def __init__(self):
        # 1. Setup Database Connection Pool for Checkpointer
        self.db_pool = ConnectionPool(
            conninfo=DATABASE_URL_RAW,
            min_size=0,  # Scale-to-zero friendly: no idle connections held
            max_size=10,
            timeout=30,  # Max 30s wait for a connection from pool
            kwargs={
                "autocommit": True,
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            },
        )

        # 2. Initialize PostgresSaver
        self.checkpointer = PostgresSaver(self.db_pool)

        # 3. Create LangGraph Checkpoint Tables automatically on boot
        try:
            self.checkpointer.setup()
            logger.info("PostgresSaver Checkpointer tables verified/created.")
        except Exception as e:
            logger.warning(f"PostgresSaver setup warning (might already exist): {e}")

        # 4. Compile Graph WITH Persistence
        self.graph = workflow.compile(checkpointer=self.checkpointer)

        # Thread-safe bridge to pass the newest message from POST to GET stream route
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self.pending_inputs: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def build_input_data(request: DiagnoseRequest) -> dict:
        """Build LangGraph input dict from a DiagnoseRequest (no side effects)."""
        if request.images and len(request.images) > 0:
            content_list = [{"type": "text", "data": request.message}]
            for img_b64 in request.images:
                content_list.append({"type": "image", "data": img_b64})
            new_user_message = {"role": "user", "content": content_list}
        else:
            new_user_message = {"role": "user", "content": request.message}

        return {
            "device_id": request.device_id,
            "messages": [new_user_message],
        }

    def create_thread(self, request: DiagnoseRequest) -> str:
        try:
            thread_id = request.thread_id if request.thread_id else str(uuid.uuid4())
            input_data = self.build_input_data(request)

            with self._lock:
                self.pending_inputs[thread_id] = input_data

            return thread_id

        except Exception as e:
            logger.exception("CRITICAL ERROR in create_thread")
            raise ValueError(f"Failed to create diagnostic thread: {str(e)}")

    def stream_graph(self, thread_id: str, input_data: dict | None = None):
        if input_data is None:
            with self._lock:
                input_data = self.pending_inputs.pop(thread_id, None)

        if not input_data:
            yield f"data: {json.dumps({'event': 'COMPLETE'})}\n\n"
            return

        config = {"configurable": {"thread_id": thread_id}}

        # Capture user question before streaming (not available in final_state).
        seed_question = ""
        user_messages = input_data.get("messages", [])
        for m in reversed(user_messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    seed_question = content
                elif isinstance(content, list):
                    seed_question = " ".join(
                        item["data"] for item in content if item.get("type") == "text"
                    )
                break

        # Telemetry: trace + session duration
        trace_id = str(uuid.uuid4())
        session_start = time.perf_counter()
        telemetry_records: list[dict] = []
        batch_persisted = False

        # SSE buffering for the inline retry gate.
        buffered_plan_event = None
        buffered_inline_event = None
        repair_planner_count = 0
        node_timestamps: Dict[str, str] = {}

        # Initialize edge telemetry collector for this session
        set_edge_telemetry_thread_id(thread_id)

        def _record_timestamps(ev: dict):
            """Capture wall-clock time for each node in an updates event."""
            now = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for name, payload in ev.items():
                if isinstance(payload, dict):
                    node_timestamps[name] = now

        def _strip_telemetry(ev: dict) -> dict:
            return {
                k: {kk: vv for kk, vv in v.items() if kk != "execution_telemetry"}
                if isinstance(v, dict)
                else v
                for k, v in ev.items()
            }

        def _flush_buffered():
            nonlocal buffered_plan_event, buffered_inline_event
            chunks = []
            if buffered_plan_event is not None:
                chunks.append(
                    f"data: {json.dumps(_strip_telemetry(buffered_plan_event), default=_json_default)}\n\n"
                )
                buffered_plan_event = None
            if buffered_inline_event is not None:
                chunks.append(
                    f"data: {json.dumps(_strip_telemetry(buffered_inline_event), default=_json_default)}\n\n"
                )
                buffered_inline_event = None
            return chunks

        try:
            for mode, event in self.graph.stream(
                input_data,
                config=config,
                stream_mode=["updates", "custom"],
            ):
                if mode == "custom":
                    yield f"data: {json.dumps({'event': 'custom', **event})}\n\n"
                    continue

                _record_timestamps(event)

                # Collect execution telemetry from node outputs
                for node_key, node_payload in event.items():
                    if (
                        isinstance(node_payload, dict)
                        and "execution_telemetry" in node_payload
                    ):
                        # L4.2: Inject state fingerprint into telemetry metadata
                        fingerprint = state_fingerprint(node_payload)
                        for rec in node_payload["execution_telemetry"]:
                            if rec.get("metadata") is None:
                                rec["metadata"] = {}
                            rec["metadata"]["output_state_hash"] = fingerprint
                        telemetry_records.extend(node_payload["execution_telemetry"])

                # L2-SSE01: Named event — IntentClassified
                if "IntentRouter" in event:
                    ir_state = event["IntentRouter"] or {}
                    yield f"data: {json.dumps({'event': 'IntentClassified', 'intent_type': ir_state.get('route', ''), 'equipment_type': ir_state.get('equipment_type', '')})}\n\n"

                # L2-SSE01b: UnsafeMethodGate verdict (emitted if gate fires)
                if "UnsafeMethodGate" in event:
                    umg_state = event["UnsafeMethodGate"] or {}
                    yield f"data: {json.dumps({'event': 'UnsafeMethodGate', 'verdict': 'unsafe', 'corrected_intent': umg_state.get('route', ''), 'safety_warning': umg_state.get('safety_warning', '')})}\n\n"

                # L2-SSE02: Named event — RetrievalComplete
                if "KnowledgeRetriever" in event:
                    kr_state = event["KnowledgeRetriever"] or {}
                    kr_chunks = kr_state.get("retrieved_chunk_ids", [])
                    kr_telem = kr_state.get("execution_telemetry", [{}])
                    kr_latency = kr_telem[0].get("latency_ms", 0) if kr_telem else 0
                    yield f"data: {json.dumps({'event': 'RetrievalComplete', 'chunk_count': len(kr_chunks), 'retrieval_latency_ms': kr_latency})}\n\n"

                # L2-SSE03: Named event — SafetyExtractor
                if "SafetyExtractor" in event:
                    sx_state = event["SafetyExtractor"] or {}
                    sx_protocols = sx_state.get("safety_protocols", [])
                    sx_metadata = sx_state.get("safety_extraction_metadata", {})
                    yield f"data: {json.dumps({'event': 'SafetyExtractor', 'safety_protocols': sx_protocols if isinstance(sx_protocols, list) else [], 'extraction_metadata': sx_metadata}, default=str)}\n\n"

                if "RepairPlanner" in event:
                    repair_planner_count += 1
                    # A retry produced a new plan — discard the prior buffered
                    # plan/eval so the user only ever sees the winning attempt.
                    buffered_plan_event = event
                    buffered_inline_event = None
                    if repair_planner_count == 1:
                        msg = "Verifying repair plan quality..."
                    else:
                        msg = "Re-verifying repair plan..."
                    yield f"data: {json.dumps({'event': 'VERIFYING', 'message': msg})}\n\n"
                    continue

                if "InlineEvaluator" in event:
                    ie_state = event["InlineEvaluator"] or {}
                    # L2-SSE05: Named event — EvaluationComplete
                    ie_eval = ie_state.get("inline_eval") or {}
                    yield f"data: {json.dumps({'event': 'EvaluationComplete', 'badge_color': ie_eval.get('badge', ''), 'faithfulness': ie_eval.get('scores', {}).get('faithfulness'), 'answer_relevance': ie_eval.get('scores', {}).get('answer_relevance'), 'per_step_verdicts': ie_eval.get('per_step_verdicts', [])})}\n\n"
                    # Default True is fail-open: a malformed eval payload must
                    # not strand the buffered plan in limbo.
                    passed = bool(ie_state.get("inline_eval_passed", True))
                    buffered_inline_event = event
                    # Final iff the eval passed OR we've already retried once.
                    # On retry-needed (failed + count==1) the workflow loops
                    # back to RepairPlanner and a new plan event will arrive,
                    # which will overwrite buffered_inline_event above.
                    is_final = passed or repair_planner_count >= 2
                    if is_final:
                        for chunk in _flush_buffered():
                            yield chunk
                    continue

                yield f"data: {json.dumps(_strip_telemetry(event), default=_json_default)}\n\n"

            # Defensive flush: if the graph terminated with anything still
            # buffered (e.g. eval node bypassed), emit it before AGENT_DONE so
            # the UI doesn't get stuck on the VERIFYING spinner.
            for chunk in _flush_buffered():
                yield chunk

            # C-5: Post-flush safety-step alignment (off the critical plan path).
            # Runs after the user already sees the plan + badge; emitted as a
            # follow-up SSE event so the frontend can render per-step safety
            # rules incrementally.
            try:
                snapshot_pre = self.graph.get_state(config)
                _vals = (
                    dict(snapshot_pre.values)
                    if snapshot_pre and snapshot_pre.values
                    else {}
                )
                _sp = _vals.get("safety_protocols")
                _steps = _vals.get("repair_steps", [])
                if isinstance(_sp, dict) and _sp.get("rules") and _steps:
                    from pipeline.safety_alignment import align_safety_to_steps

                    alignment, align_tel = align_safety_to_steps(
                        safety_rules=_sp["rules"],
                        repair_steps=_steps,
                    )
                    if alignment:
                        yield f"data: {json.dumps({'event': 'SafetyStepAlignment', 'alignment': {str(k): v for k, v in alignment.items()}}, default=_json_default)}\n\n"
                    if align_tel:
                        telemetry_records.append(align_tel)
            except Exception:
                logger.debug("Safety-step alignment skipped", exc_info=True)

            # Unblock UI immediately so the badge / CTA can render.
            session_duration_ms = int((time.perf_counter() - session_start) * 1000)
            yield f"data: {json.dumps({'event': 'AGENT_DONE', 'session_duration_ms': session_duration_ms, 'trace_id': trace_id})}\n\n"

            # Read the final, persisted state from the PostgresSaver checkpoint
            # so we can dispatch shadow evaluation.
            try:
                snapshot = self.graph.get_state(config)
                final_state: Dict[str, Any] = (
                    dict(snapshot.values) if snapshot and snapshot.values else {}
                )
            except Exception as e:
                logger.warning(f"Checkpoint read failed: {e}")
                final_state = {}
            final_state["_seed_question"] = seed_question
            final_state["_thread_id"] = thread_id

            # Dispatch shadow evaluation asynchronously (Cloud Tasks in cloud,
            # BackgroundTask in local). Frontend polls for results.
            try:
                from api.routers.evaluation import dispatch_shadow_eval

                dispatch_shadow_eval(final_state, trace_id)
            except Exception as e:
                logger.warning(f"Shadow eval dispatch failed: {e}")

            # Batch-persist execution telemetry (edge decisions + shadow eval)
            try:
                telemetry_records.extend(collect_edge_telemetry(thread_id))
                # Grab response_id from final state for linking
                final_response_id = final_state.get("response_id")
                _persist_execution_logs(
                    trace_id, thread_id, final_response_id, telemetry_records
                )
                batch_persisted = True
            except Exception as telem_err:
                logger.warning(f"Telemetry batch persist failed: {telem_err}")

            # R2: Persist terminal state + node timestamps for analytics.
            try:
                route = final_state.get("route", "")
                deterministic = final_state.get("deterministic_override")
                if (
                    route
                    in (
                        "troubleshoot",
                        "direct_replacement",
                        "replace_part",
                    )
                    or deterministic
                ):
                    terminal = "COMPLETED"
                else:
                    terminal = "IN_PROGRESS"
                from core.crud import update_thread_terminal_state

                update_thread_terminal_state(thread_id, terminal, node_timestamps)
            except Exception as ts_err:
                logger.debug(f"Failed to update thread terminal state: {ts_err}")

            yield f"data: {json.dumps({'event': 'COMPLETE'})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream_graph: {e}")
            yield f"data: {json.dumps({'event': 'ERROR', 'error': str(e)})}\n\n"
        finally:
            # Crash recovery: persist partial telemetry if batch wasn't already written
            if telemetry_records and not batch_persisted:
                try:
                    telemetry_records.extend(collect_edge_telemetry(thread_id))
                    _persist_execution_logs(
                        trace_id, thread_id, None, telemetry_records
                    )
                except Exception:
                    logger.error(
                        "Crash-recovery telemetry persist failed", exc_info=True
                    )

    # resume_after_gate removed: the HumanGatekeeper interrupt was replaced by
    # the inline retry gate + flagged_for_review admin alert. Red badges no
    # longer pause the graph waiting on a /diagnose/resume POST.

    @with_db_retry
    def get_all_threads(self) -> list:
        with self.db_pool.connection() as conn:
            records = conn.execute(
                "SELECT thread_id, max(checkpoint_id) FROM checkpoints GROUP BY thread_id ORDER BY max(checkpoint_id) DESC"
            ).fetchall()

        threads = []
        for row in records:
            t_id = str(row[0])
            c_id = str(row[1])

            date_str = datetime.datetime.now().strftime("%d/%m/%Y")
            try:
                u = uuid.UUID(c_id)
                t = ((u.int >> 80) << 12) | ((u.int >> 64) & 0x0FFF)
                t_sec = (t - 0x01B21DD213814000) / 10000000.0
                date_str = datetime.datetime.fromtimestamp(t_sec).strftime("%d/%m/%Y")
            except Exception:
                pass

            config = {"configurable": {"thread_id": t_id}}
            try:
                snapshot = self.graph.get_state(config)
                state = snapshot.values if snapshot and snapshot.values else {}
            except Exception:
                state = {}

            device_id = state.get("device_id", "Unknown Device")
            messages = state.get("messages", [])
            snippet = "No messages"
            if messages and isinstance(messages, list) and len(messages) > 0:
                first_msg = messages[0]
                if isinstance(first_msg, dict):
                    content = first_msg.get("content")
                    if isinstance(content, str):
                        snippet = content[:50] + "..." if len(content) > 50 else content
                    elif isinstance(content, list):
                        snippet = "Multimodal diagnostic session..."

            threads.append(
                {
                    "thread_id": t_id,
                    "device_id": str(device_id),
                    "date": str(date_str),
                    "snippet": str(snippet),
                }
            )
        return threads

    @with_db_retry
    def get_thread_history(self, thread_id: str) -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        try:
            terminal_state = get_thread_terminal_state(thread_id)
        except Exception:
            terminal_state = "IN_PROGRESS"
        empty = {
            "device_id": "Unknown",
            "messages": [],
            "inline_evals": {},
            "shadow_evals": {},
            "terminal_state": terminal_state,
            "initial_user_message": None,
        }
        try:
            snapshot = self.graph.get_state(config)
            if not snapshot or not snapshot.values:
                return empty
            state = snapshot.values
        except Exception:
            return empty

        inline_evals, shadow_evals = self._load_evals_for_thread(thread_id)
        messages = state.get("messages", [])

        return {
            "device_id": state.get("device_id", "Unknown"),
            "messages": messages,
            "is_plan_safe": state.get("is_plan_safe"),
            "inline_evals": inline_evals,
            "shadow_evals": shadow_evals,
            "terminal_state": terminal_state,
            "initial_user_message": _extract_initial_user_message(messages),
        }

    @staticmethod
    def _load_evals_for_thread(thread_id: str) -> tuple[dict, dict]:
        """Build {response_id: InlineEvalEvent} and {response_id: ShadowEvaluation}
        maps from the persisted evaluation_metrics rows so history replay can
        render the same RAGAS panels the live stream did."""
        inline_evals: Dict[str, Any] = {}
        shadow_evals: Dict[str, Any] = {}
        try:
            rows = get_evaluations_for_thread(thread_id)
        except Exception as e:
            logger.warning(f"get_evaluations_for_thread failed for {thread_id}: {e}")
            return inline_evals, shadow_evals

        for r in rows:
            rid = r["response_id"]
            sync_reasons = r.get("sync_reasons") or {}
            badge = r.get("badge") or "red"
            # Legacy rows may carry "gray"; the live UI maps that to red.
            if badge not in ("green", "yellow", "red"):
                badge = "red"
            inline_evals[rid] = {
                "scores": {
                    "faithfulness": r.get("sync_faithfulness"),
                    "answer_relevance": r.get("sync_answer_relevance"),
                },
                "reasons": {
                    "faithfulness": sync_reasons.get("faithfulness", ""),
                    "answer_relevance": sync_reasons.get("answer_relevance", ""),
                },
                "badge": badge,
                "quality_badge": r.get("quality_badge") or badge,
                "safety_badge": r.get("safety_badge"),
                "duration_ms": r.get("sync_duration_ms"),
                "error": None,
                "response_id": rid,
                "step_verdicts": r.get("step_verdicts"),
            }

            ctx_score = r.get("async_context_relevance")
            comp_score = r.get("async_completeness")
            if ctx_score is None and comp_score is None:
                continue
            evidence = r.get("async_evidence") or {}
            ctx_meta = (
                evidence.get("context_relevance")
                if isinstance(evidence, dict)
                else None
            )
            comp_meta = (
                evidence.get("completeness") if isinstance(evidence, dict) else None
            )
            shadow: Dict[str, Any] = {
                "response_id": rid,
                "duration_ms": r.get("async_duration_ms"),
                "completed_at": r.get("async_completed_at"),
            }
            if ctx_score is not None:
                shadow["context_relevance"] = {
                    "score": ctx_score,
                    "reason": (ctx_meta or {}).get("reason", "")
                    if isinstance(ctx_meta, dict)
                    else "",
                }
            if comp_score is not None:
                shadow["completeness"] = {
                    "score": comp_score,
                    "reason": (comp_meta or {}).get("reason", "")
                    if isinstance(comp_meta, dict)
                    else "",
                }
            shadow_evals[rid] = shadow

        return inline_evals, shadow_evals


# Instantiate Singleton
agent_manager = AgentManager()
