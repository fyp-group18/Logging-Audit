from typing import Literal, Optional, List, Any, Dict, Union

from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime


class TextContent(BaseModel):
    type: str = "text"
    data: str


class ImageContent(BaseModel):
    type: str = "image"
    data: str


class TimelineContent(BaseModel):
    type: str = "timeline"
    events: List[dict]
    response_id: Optional[str] = None


ChatMessageContent = Union[str, List[Union[TextContent, ImageContent]], TimelineContent]


class ChatMessage(BaseModel):
    role: str
    content: ChatMessageContent


class DiagnoseRequest(BaseModel):
    device_id: str
    message: str
    thread_id: Optional[str] = Field(
        None, description="Optional. If provided, resumes the LangGraph thread."
    )
    images: Optional[List[str]] = Field(
        default_factory=list, description="List of base64 encoded images"
    )
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list, description="Legacy. No longer required."
    )


class EscalateRequest(BaseModel):
    thread_id: str
    reason: Optional[str] = None
    response_id: Optional[str] = None


class DocumentMetadata(BaseModel):
    intuitive_name: str
    original_filename: str
    upload_date: str
    doc_type: str
    filepath: str
    status: str = "READY"  # NEW field with default for backwards compatibility
    compatible_models: List[str] = []  # NEW: Default to empty list
    equipment_category: str = "GENERAL"  # NEW: Default to GENERAL


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]


class RenameRequest(BaseModel):
    filepath: str
    new_name: str
    doc_type: str


class DeviceModelCreate(BaseModel):
    name: str


class DeviceModelRename(BaseModel):
    new_name: str


class ThreadListResponse(BaseModel):
    threads: List["ThreadSummary"]  # Forward reference - defined below after UserUpdate
    total: int = 0
    page: int = 1
    per_page: int = 20


class ThreadHistoryResponse(BaseModel):
    device_id: str
    messages: List[dict]
    is_plan_safe: Optional[bool] = None
    # {response_id: InlineEvalEvent} — populated from evaluation_metrics so
    # history replay can render the RAGAS panel under each assistant message.
    inline_evals: Dict[str, Any] = Field(default_factory=dict)
    # {response_id: ShadowEvaluation} — context_relevance + completeness.
    shadow_evals: Dict[str, Any] = Field(default_factory=dict)
    # Refresh-recovery: IN_PROGRESS / COMPLETED / ABANDONED. If not COMPLETED
    # the frontend auto-reruns using initial_user_message against a new thread.
    terminal_state: str = "IN_PROGRESS"
    initial_user_message: Optional[Dict[str, Any]] = None


class DocumentMetadataUpdate(BaseModel):
    filepath: str
    model_ids: Optional[List[str]] = None  # For manuals
    category: Optional[str] = None  # For safety docs


class ReportMetadata(BaseModel):
    thread_id: str
    device_id: str
    generated_at: str
    file_size_bytes: Optional[int] = None


class ReportSummary(BaseModel):
    id: int
    thread_id: str
    device_id: str
    object_key: str
    generated_at: str
    user_email: Optional[str] = None


class ReportListResponse(BaseModel):
    reports: List[ReportSummary]


class OrphanedModelsResponse(BaseModel):
    models: List[str]


class EquipmentCategoryItem(BaseModel):
    value: str
    label: str


class CategoriesResponse(BaseModel):
    categories: List[EquipmentCategoryItem]


# --- Auth Schemas ---
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    role: Optional[str] = None
    level: Optional[str] = None


class AuthUser(BaseModel):
    id: str
    email: str
    role: str
    level: Optional[str] = None


class LoginResponse(BaseModel):
    user: AuthUser


# --- User Schemas ---
class UserBase(BaseModel):
    email: str
    role: str  # Should be 'ADMIN' or 'TECHNICIAN'
    level: Optional[str] = None  # 'JUNIOR' or 'SENIOR', only for TECHNICIAN


class UserCreate(UserBase):
    password: str


class UserInDB(UserBase):
    id: UUID

    class Config:
        from_attributes = True


class UserPublic(BaseModel):
    id: UUID
    email: str
    role: str
    level: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    email: str = Field(..., example="technician@example.com")
    role: str = Field(..., example="TECHNICIAN")
    level: Optional[str] = Field(None, example="JUNIOR")


class ThreadSummary(BaseModel):
    thread_id: str
    device_id: str
    date: str
    snippet: str
    user_email: Optional[str] = None
    flagged_for_review: bool = False
    badge: Optional[str] = None  # composite: green | yellow | red | gray | null
    quality_badge: Optional[str] = None  # green | yellow | red | gray | null
    safety_badge: Optional[str] = None  # green | yellow | red | null


# Resolve forward reference for ThreadListResponse
ThreadListResponse.model_rebuild()


# Signed URL upload schemas
MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB


class UploadUrlRequest(BaseModel):
    filename: str
    content_type: str = "application/pdf"
    doc_type: str = "manual"
    file_size: int  # File size in bytes, validated before generating signed URL


class UploadUrlResponse(BaseModel):
    signed_url: str
    object_key: str
    content_type: str
    expires_in_seconds: int


class ConfirmUploadRequest(BaseModel):
    object_key: str
    doc_type: str
    custom_title: Optional[str] = None


class ConfirmUploadResponse(BaseModel):
    status: str
    document_id: int
    message: str


class ViewUrlRequest(BaseModel):
    object_key: str


class ViewUrlResponse(BaseModel):
    signed_url: str
    expires_in_seconds: int


# --- Feedback Schemas ---
class ResponseFeedbackCreate(BaseModel):
    response_id: str
    thread_id: str
    feedback_type: Literal["THUMBS_UP", "THUMBS_DOWN"] = Field(
        ..., description="THUMBS_UP or THUMBS_DOWN"
    )
    correction_text: Optional[str] = None
    correction_category: Optional[
        Literal[
            "HALLUCINATION",
            "OUTDATED_SOURCE",
            "MISSING_STEP",
            "WRONG_VALUE",
            "WRONG_SEQUENCE",
            "OTHER",
        ]
    ] = None
    satisfaction_rating: Optional[int] = Field(None, ge=1, le=5)
    # L3-E01: Audit enrichment fields
    operator_role: Optional[str] = None
    time_on_feedback_ms: Optional[int] = Field(None, ge=0)


class StepFeedbackCreate(BaseModel):
    response_id: str
    thread_id: str
    step_index: int = Field(..., ge=0)
    action: Optional[Literal["ACCEPT", "SKIP", "MODIFY", "INCORRECT"]] = None
    correction_text: Optional[str] = None
    # L3-E02: Step-level audit enrichment
    correction_type: Optional[
        Literal[
            "factual_error",
            "ordering_error",
            "missing_safety_information",
            "wrong_tool_part",
            "irrelevant_step",
        ]
    ] = None
    severity: Optional[Literal["minor", "moderate", "critical"]] = None
    time_on_step_ms: Optional[int] = Field(None, ge=0)


class SourceReportCreate(BaseModel):
    chunk_id: str
    response_id: str
    thread_id: str
    reason: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


class StepFeedbackOut(BaseModel):
    step_index: int
    action: Optional[str] = None
    correction_text: Optional[str] = None


class ResponseFeedbackOut(BaseModel):
    feedback_type: Optional[str] = None
    correction_text: Optional[str] = None
    correction_category: Optional[str] = None
    satisfaction_rating: Optional[int] = None
    flagged_steps: List[StepFeedbackOut] = []


class OutcomeRequest(BaseModel):
    thread_id: str
    resolved: bool


class RecentThreadForDevice(BaseModel):
    thread_id: str
    device_id: str
    date: str
    snippet: str


class DeviceHistoryDigest(BaseModel):
    digest: str
    recurring_issues: List[Dict[str, Any]] = []
    session_count: int = 0


# --- Admin Feedback Schemas ---
class RetractRequest(BaseModel):
    reason: str


class AntiExampleCreate(BaseModel):
    symptom_pattern: str
    incorrect_procedure: str
    correct_alternative: Optional[str] = None
    severity: Literal["MINOR", "MAJOR", "SAFETY_CRITICAL"] = "MINOR"
    equipment_category: Literal[
        "HEAVY_INDUSTRIAL", "CONSUMER_ELECTRONICS", "IT_NETWORKING", "GENERAL"
    ] = "GENERAL"
    device_id: Optional[str] = None


class AntiExampleOut(BaseModel):
    id: str
    symptom_pattern: str
    incorrect_procedure: str
    correct_alternative: Optional[str] = None
    severity: str
    equipment_category: str
    device_id: Optional[str] = None
    is_active: bool
    superseded_by: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DeterministicRuleCreate(BaseModel):
    device_id: Optional[str] = None
    equipment_category: Literal[
        "HEAVY_INDUSTRIAL", "CONSUMER_ELECTRONICS", "IT_NETWORKING", "GENERAL"
    ] = "GENERAL"
    trigger_keywords: List[str] = Field(..., min_length=1)
    override_response: str
    override_safety_protocols: Optional[str] = None
    priority: int = 0
    source: Literal["SAFETY_RECALL", "AUDIT_FINDING", "ENGINEERING_ORDER", "MANUAL"] = (
        "MANUAL"
    )
    expires_at: Optional[datetime] = None
    review_date: Optional[datetime] = None


class DeterministicRuleOut(BaseModel):
    id: str
    device_id: Optional[str] = None
    equipment_category: str
    trigger_keywords: List[str]
    override_response: str
    priority: int
    source: str
    is_active: bool
    approved_by: Optional[str] = None
    expires_at: Optional[datetime] = None
    review_date: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PromptAmendmentCreate(BaseModel):
    target_prompt: str
    equipment_category: Optional[str] = None
    amendment_text: str
    reason: str


class PromptAmendmentOut(BaseModel):
    id: str
    target_prompt: str
    equipment_category: Optional[str] = None
    amendment_text: str
    reason: str
    is_active: bool
    approved_by: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class BlacklistRequest(BaseModel):
    reason: str


class BlacklistedChunkOut(BaseModel):
    chunk_id: str
    document_name: str
    text_preview: str
    reason: str
    blacklisted_at: Optional[datetime] = None


class AuditLogEntry(BaseModel):
    id: str
    event_type: str
    target_type: str
    target_id: str
    performed_by: str
    reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int


class EscalationFlagCreate(BaseModel):
    thread_id: str
    response_id: Optional[str] = None
    reason: Optional[str] = None
    assigned_reviewer: Optional[str] = None


class EscalationFlagOut(BaseModel):
    id: str
    thread_id: str
    response_id: Optional[str] = None
    flagged_by: str
    reason: Optional[str] = None
    assigned_reviewer: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewVerdictCreate(BaseModel):
    thread_id: str
    response_id: str
    node_name: str
    step_index: Optional[int] = None
    verdict: Literal["CORRECT", "PARTIAL", "INCORRECT"]
    correct_procedure: Optional[str] = None
    error_category: Optional[
        Literal[
            "HALLUCINATION",
            "OUTDATED_SOURCE",
            "MISSING_STEP",
            "WRONG_VALUE",
            "WRONG_SEQUENCE",
            "OTHER",
        ]
    ] = None
    notes: Optional[str] = None


class ReviewVerdictOut(BaseModel):
    id: str
    thread_id: str
    response_id: str
    node_name: str
    step_index: Optional[int] = None
    verdict: str
    correct_procedure: Optional[str] = None
    error_category: Optional[str] = None
    reviewer_id: str
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ReviewQueueItem(BaseModel):
    thread_id: str
    device_id: str
    date: str
    review_status: str
    is_escalated: bool = False
    risk_tier: int = 0
    user_email: Optional[str] = None
    terminal_state: str = "IN_PROGRESS"
    badge: Optional[str] = None
    quality_badge: Optional[str] = None
    safety_badge: Optional[str] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None


class ReviewQueueResponse(BaseModel):
    items: List[ReviewQueueItem]
    total: int
    page: int
    page_size: int


class BulkApproveRequest(BaseModel):
    thread_ids: List[str]


class SessionReplayResponse(BaseModel):
    thread_history: Dict[str, Any]
    escalations: List[EscalationFlagOut] = []
    verdicts: List[ReviewVerdictOut] = []
    execution_logs: List[Dict[str, Any]] = []
    execution_summary: Dict[str, Any] = {}
    chunk_attribution: List[Dict[str, Any]] = []
    # Pipeline context extracted from LangGraph checkpoint state
    diagnostic_context: Dict[str, Any] = {}
    retrieval_context: Dict[str, Any] = {}
    safety_context: Dict[str, Any] = {}
    evaluation_metrics: List[Dict[str, Any]] = []


class CheckpointInfo(BaseModel):
    checkpoint_id: str
    node_name: str
    timestamp: str


# --- Async Review Queue Schemas ---
class ReviewTaskFlagRequest(BaseModel):
    """Manual flag from any authenticated technician on a response."""

    thread_id: str
    response_id: Optional[str] = None
    reason: Optional[str] = Field(
        None,
        max_length=2000,
        description="Optional context for the flag. Empty reason is allowed (P3 task).",
    )


class ReviewTaskOut(BaseModel):
    id: str
    thread_id: str
    response_id: Optional[str] = None
    flagged_by: str
    source: Literal["AUTO_SAFETY", "AUTO_EVAL", "MANUAL_FLAG"]
    priority: Literal["P0", "P1", "P2", "P3"]
    status: Literal["NEW", "CLAIMED", "IN_REVIEW", "RESOLVED", "SNOOZED"]
    trigger_reason: Optional[str] = None
    payload_snapshot: Dict[str, Any] = {}
    claimed_by: Optional[str] = None
    claimed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    snoozed_until: Optional[datetime] = None
    reason: Optional[str] = None
    created_at: datetime


class ReviewTaskQueueResponse(BaseModel):
    items: List[ReviewTaskOut]
    total: int
    page: int
    page_size: int


class ReviewTaskSnoozeRequest(BaseModel):
    snooze_for: Literal["4h", "1d", "1w"] = Field(
        ..., description="How long to defer this task before it returns to the queue."
    )


class ReviewTaskVerdictRequest(BaseModel):
    """Verdict submitted by an expert that resolves the review task and writes
    to one of the memory tables (anti_examples / preference_pairs / rules) or
    blacklists offending chunks."""

    verdict_type: Literal[
        "APPROVE",
        "EDIT",
        "REJECT",
        "ADD_RULE",
        "BLACKLIST_CHUNK",
        "REPLACE_MANUAL",
    ]
    # EDIT: corrected response that becomes a preference_pair
    corrected_text: Optional[str] = None
    # REJECT / ADD_RULE: explanation that becomes anti_example.incorrect_procedure
    # or rule.override_response
    reason: Optional[str] = None
    # REJECT only: severity of the anti-example
    severity: Optional[Literal["MINOR", "MAJOR", "SAFETY_CRITICAL"]] = None
    # ADD_RULE only: trigger keywords for the deterministic rule
    trigger_keywords: Optional[List[str]] = None
    # BLACKLIST_CHUNK only: chunk IDs the expert wants flagged out of retrieval
    chunk_ids: Optional[List[str]] = None
    # REPLACE_MANUAL only: id of the document the new upload replaces (set by
    # the /replace-manual endpoint, not the verdict POST itself)
    replacement_document_id: Optional[int] = None


class ReviewTaskChunkOut(BaseModel):
    """Citation chunk surfaced in the task detail panel — feeds the
    BLACKLIST_CHUNK selection list."""

    chunk_id: str
    document_id: int
    document_name: str
    document_filepath: str
    pages: Optional[str] = None
    text_preview: str
    blacklisted: bool


class NotificationOut(BaseModel):
    id: str
    kind: Literal[
        "TASK_ASSIGNED", "TASK_REMINDER", "TASK_RESOLVED", "TASK_SNOOZE_EXPIRED"
    ]
    title: str
    body: Optional[str] = None
    link: Optional[str] = None
    task_id: Optional[str] = None
    read_at: Optional[datetime] = None
    created_at: datetime


class NotificationsMarkReadRequest(BaseModel):
    notification_ids: List[str]


# --- Thread Evaluation Detail Schemas ---
class EvaluationMetricOut(BaseModel):
    id: int
    response_id: str
    badge: Optional[str] = None
    quality_badge: Optional[str] = None
    safety_badge: Optional[str] = None
    attempt_number: Optional[int] = None
    override: bool = False
    override_reason: Optional[str] = None
    sync_faithfulness: Optional[float] = None
    sync_answer_relevance: Optional[float] = None
    sync_context_relevance: Optional[float] = None
    sync_completeness: Optional[float] = None
    sync_reasons: Dict[str, Any] = {}
    sync_duration_ms: Optional[int] = None
    async_faithfulness: Optional[float] = None
    async_answer_relevance: Optional[float] = None
    async_context_relevance: Optional[float] = None
    async_completeness: Optional[float] = None
    async_evidence: Optional[Dict[str, Any]] = None
    async_duration_ms: Optional[int] = None
    step_verdicts: Optional[List[Dict[str, Any]]] = None
    safety_confidence: Optional[float] = None
    created_at: datetime


class EscalationStatusOut(BaseModel):
    id: str
    source: str
    priority: str
    status: str
    trigger_reason: Optional[str] = None
    claimed_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    created_at: datetime


class ThreadStepFeedbackOut(BaseModel):
    response_id: str
    step_index: int
    action: Optional[str] = None
    correction_text: Optional[str] = None


class ThreadResponseFeedbackOut(BaseModel):
    response_id: str
    feedback_type: Optional[str] = None
    correction_text: Optional[str] = None
    correction_category: Optional[str] = None
    satisfaction_rating: Optional[int] = None
    created_at: datetime


class ThreadEvaluationResponse(BaseModel):
    evaluations: List[EvaluationMetricOut]
    response_feedbacks: List[ThreadResponseFeedbackOut]
    step_feedbacks: List[ThreadStepFeedbackOut]
    review_status: Optional[EscalationStatusOut] = None


# --- Trace / "How the AI Thinks" Debug View ---


class TraceChunkDetail(BaseModel):
    chunk_id: str
    text_preview: Optional[str] = None
    level: int = 0
    document_id: Optional[int] = None
    document_name: Optional[str] = None
    pages: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_header: Optional[str] = None
    similarity_score: Optional[float] = None
    reranker_score: Optional[int] = None
    grounding_label: Optional[str] = None
    step_index: Optional[int] = None
    source_chunk_ids: Optional[List[str]] = None
    has_safety_content: bool = False


class TraceQueryResult(BaseModel):
    query: str
    chunks: List[TraceChunkDetail] = []


class TraceQueryProcessing(BaseModel):
    user_query: Optional[str] = None
    search_queries: List[str] = []
    per_query_results: List[TraceQueryResult] = []


class TraceRetrievalPipeline(BaseModel):
    total_chunks_retrieved: int = 0
    chunks_after_filtering: int = 0
    filtering_threshold: Optional[float] = None
    filtered_out_chunks: List[TraceChunkDetail] = []
    reranked_chunks: List[TraceChunkDetail] = []
    reranker_model: Optional[str] = None
    chunk_allocation: Dict[str, List[str]] = {}


class TraceIntentRouting(BaseModel):
    intent: Optional[str] = None
    routing_rationale: Optional[str] = None
    unsafe_method_gate: Optional[Dict[str, Any]] = None
    deterministic_rule_match: Optional[Dict[str, Any]] = None
    node_sequence: List[str] = []


class TraceNodeExecution(BaseModel):
    node_name: str
    model: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TraceEvaluationScores(BaseModel):
    sync_faithfulness: Optional[float] = None
    sync_answer_relevance: Optional[float] = None
    sync_context_relevance: Optional[float] = None
    sync_completeness: Optional[float] = None
    async_faithfulness: Optional[float] = None
    async_answer_relevance: Optional[float] = None
    async_context_relevance: Optional[float] = None
    async_completeness: Optional[float] = None
    async_safety_coverage: Optional[float] = None
    safety_survival: Optional[Dict[str, Any]] = None
    badge: Optional[str] = None
    quality_badge: Optional[str] = None
    safety_badge: Optional[str] = None
    step_verdicts: Optional[List[Dict[str, Any]]] = None
    thresholds_snapshot: Optional[Dict[str, Any]] = None


class TraceFollowUp(BaseModel):
    parent_thread_id: Optional[str] = None
    child_thread_ids: List[str] = []


class TraceRaptorNode(BaseModel):
    chunk_id: str
    level: int = 0
    text_preview: Optional[str] = None
    source_chunk_ids: Optional[List[str]] = None
    was_retrieved: bool = False
    pages: Optional[str] = None
    section_header: Optional[str] = None


class TraceL1Metrics(BaseModel):
    leaf_coverage_d: Optional[float] = None
    leaf_count: int = 0
    summary_count: int = 0
    total_count: int = 0
    max_depth: int = 0
    kb_reconstructability: Optional[float] = None
    chunks_with_provenance: int = 0
    chunks_total_retrieved: int = 0
    manifest_valid: Optional[bool] = None
    manifest_entry_count: int = 0
    manifest_first_broken_at: Optional[int] = None
    ingestion_timestamp: Optional[str] = None
    embedding_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    clusters_per_level: List[Dict[str, Any]] = []


class TraceRaptorTree(BaseModel):
    document_id: int
    document_name: Optional[str] = None
    nodes: List[TraceRaptorNode] = []
    l1_metrics: Optional[TraceL1Metrics] = None


class TraceFeedback(BaseModel):
    response_feedbacks: List[Dict[str, Any]] = []
    step_feedbacks: List[Dict[str, Any]] = []


class TraceCrossLayerMetrics(BaseModel):
    provenance_total_steps: int = 0
    provenance_traceable_steps: int = 0
    provenance_score: Optional[float] = None
    safety_provenance_total: int = 0
    safety_provenance_complete: int = 0
    safety_provenance_score: Optional[float] = None
    has_safety_extractor: bool = False
    evaluator_agreement: Optional[List[Dict[str, Any]]] = None


class TraceConformance(BaseModel):
    conforming: bool = True
    variant_label: Optional[str] = None
    deviating_edge: Optional[Dict[str, str]] = None


class TraceAuditOverhead(BaseModel):
    total_execution_ms: Optional[int] = None
    audit_overhead_ms: Optional[int] = None
    audit_overhead_pct: Optional[float] = None
    instrumented: bool = False


class TraceResponse(BaseModel):
    thread_id: str
    response_id: str
    device_id: Optional[str] = None
    query_processing: TraceQueryProcessing = TraceQueryProcessing()
    retrieval_pipeline: TraceRetrievalPipeline = TraceRetrievalPipeline()
    intent_routing: TraceIntentRouting = TraceIntentRouting()
    agent_reasoning: List[TraceNodeExecution] = []
    evaluation_scores: TraceEvaluationScores = TraceEvaluationScores()
    cross_layer_metrics: TraceCrossLayerMetrics = TraceCrossLayerMetrics()
    follow_up: TraceFollowUp = TraceFollowUp()
    raptor_trees: List[TraceRaptorTree] = []
    trace_conformance: TraceConformance = TraceConformance()
    audit_overhead: TraceAuditOverhead = TraceAuditOverhead()
    feedback: TraceFeedback = TraceFeedback()
