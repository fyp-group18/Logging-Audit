import enum
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    ForeignKey,
    Enum,
    func,
    CheckConstraint,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR, UUID
from sqlalchemy.orm import relationship
from core.database import Base


# --- Auth Enums ---
class UserRole(enum.Enum):
    ADMIN = "ADMIN"
    TECHNICIAN = "TECHNICIAN"


class TechnicianLevel(enum.Enum):
    JUNIOR = "JUNIOR"
    SENIOR = "SENIOR"


# --- User Model ---
class User(Base):
    __tablename__ = "users"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    level = Column(Enum(TechnicianLevel), nullable=True)  # Only for technicians
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            "(role = 'ADMIN' AND level IS NULL) OR (role = 'TECHNICIAN' AND level IS NOT NULL)",
            name="check_user_role_level",
        ),
    )


class DocumentStatus(enum.Enum):
    PENDING = "PENDING"
    PARSING_VISION = "PARSING_VISION"
    BUILDING_TREE = "BUILDING_TREE"
    READY = "READY"
    FAILED = "FAILED"


class EquipmentCategory(enum.Enum):
    HEAVY_INDUSTRIAL = "HEAVY_INDUSTRIAL"
    CONSUMER_ELECTRONICS = "CONSUMER_ELECTRONICS"
    IT_NETWORKING = "IT_NETWORKING"
    GENERAL = "GENERAL"


class ThreadTerminalState(enum.Enum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    ABANDONED = "ABANDONED"


class ReviewStatus(enum.Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    REVIEWED = "REVIEWED"
    BULK_APPROVED = "BULK_APPROVED"


class VerdictEnum(enum.Enum):
    CORRECT = "CORRECT"
    PARTIAL = "PARTIAL"
    INCORRECT = "INCORRECT"


class ErrorCategory(enum.Enum):
    HALLUCINATION = "HALLUCINATION"
    OUTDATED_SOURCE = "OUTDATED_SOURCE"
    MISSING_STEP = "MISSING_STEP"
    WRONG_VALUE = "WRONG_VALUE"
    WRONG_SEQUENCE = "WRONG_SEQUENCE"
    OTHER = "OTHER"


# --- Async Review Queue Enums (replaces sync interrupt_gates flow) ---
class ReviewTaskSource(enum.Enum):
    AUTO_SAFETY = "AUTO_SAFETY"  # Safety risk category detected during evaluation
    AUTO_EVAL = "AUTO_EVAL"  # InlineEvaluator produced red-after-retry
    MANUAL_FLAG = "MANUAL_FLAG"  # Junior or expert clicked "Flag for review"
    EXPERT_FEEDBACK = "EXPERT_FEEDBACK"  # Expert 👎 with correction text


class ReviewTaskPriority(enum.Enum):
    P0 = "P0"  # safety-critical, < 4h
    P1 = "P1"  # high, < 24h
    P2 = "P2"  # medium, < 3d
    P3 = "P3"  # best-effort


class ReviewTaskStatus(enum.Enum):
    NEW = "NEW"
    CLAIMED = "CLAIMED"
    IN_REVIEW = "IN_REVIEW"
    RESOLVED = "RESOLVED"
    SNOOZED = "SNOOZED"


class NotificationKind(enum.Enum):
    TASK_ASSIGNED = "TASK_ASSIGNED"
    TASK_REMINDER = "TASK_REMINDER"
    TASK_RESOLVED = "TASK_RESOLVED"
    TASK_SNOOZE_EXPIRED = "TASK_SNOOZE_EXPIRED"


class DeviceModel(Base):
    __tablename__ = "device_models"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)


class DiagnosticReport(Base):
    __tablename__ = "diagnostic_reports"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True, nullable=False)
    device_id = Column(String, index=True, nullable=False)
    object_key = Column(String, unique=True, nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    generated_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)


class DiagnosticThread(Base):
    __tablename__ = "diagnostic_threads"
    id = Column(String, primary_key=True, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        index=True,
        nullable=True,
    )
    device_id = Column(String, nullable=False)
    terminal_state = Column(
        Enum(ThreadTerminalState),
        nullable=False,
        default=ThreadTerminalState.IN_PROGRESS,
    )
    node_timestamps = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    outcome_resolved = Column(Boolean, nullable=True)
    outcome_recorded_at = Column(DateTime(timezone=True), nullable=True)
    # HITL review columns (Workstream H)
    review_status = Column(
        Enum(ReviewStatus),
        nullable=False,
        default=ReviewStatus.PENDING_REVIEW,
        server_default=text("'PENDING_REVIEW'::reviewstatus"),
    )
    reviewed_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    is_escalated = Column(Boolean, nullable=False, server_default="false")
    risk_tier = Column(Integer, nullable=True, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    # Set when InlineEvaluator returns a red badge after retry exhaustion.
    # Surfaces in the admin history dashboard for expert review.
    flagged_for_review = Column(
        Boolean, nullable=False, default=False, server_default="false"
    )

    user = relationship("User", foreign_keys=[user_id])
    reviewer = relationship("User", foreign_keys=[reviewed_by])


class OAuthRefreshToken(Base):
    __tablename__ = "oauth_refresh_tokens"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    token_hash = Column(String, unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User")


class DocumentMultimodal(Base):
    __tablename__ = "documents_multimodal"
    id = Column(Integer, primary_key=True, index=True)
    original_filename = Column(String, index=True)
    intuitive_name = Column(String)
    filepath = Column(String, unique=True, index=True)
    doc_type = Column(String, default="manual", index=True)
    upload_date = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    file_hash = Column(String, unique=True, index=True, nullable=True)
    status = Column(
        Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False
    )
    compatible_models = Column(ARRAY(String), default=list, nullable=False)
    equipment_category = Column(
        Enum(EquipmentCategory), default=EquipmentCategory.GENERAL, nullable=False
    )
    doc_format = Column(String, nullable=True)

    chunks = relationship(
        "DocumentChunkMultimodal",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class DocumentChunkMultimodal(Base):
    __tablename__ = "document_chunks_multimodal"
    id = Column(String, primary_key=True)
    document_id = Column(
        Integer, ForeignKey("documents_multimodal.id", ondelete="CASCADE")
    )
    text = Column(Text)
    embedding = Column(Vector(3072))
    level = Column(Integer)
    pages = Column(String)
    # Numeric page bounds used for correct ordering and for procedural
    # retrieval. `pages` is kept as the human-readable label.
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    # Leaf order within a document (0-based). NULL for summary nodes.
    sequence_index = Column(Integer, nullable=True)
    # Procedure/section name from PDF headings (e.g., "Display Replacement").
    # NULL for docs ingested before this column was added.
    section_header = Column(Text, nullable=True)
    # Self-referencing FK: table-row children point to their full-table parent.
    parent_id = Column(
        String,
        ForeignKey("document_chunks_multimodal.id", ondelete="CASCADE"),
        nullable=True,
    )
    # ATA chapter/section identifier extracted from section_header (e.g. "6.3.10").
    section_code = Column(String, nullable=True)
    images = Column(JSONB, default=list)
    tsv = Column(TSVECTOR, nullable=True)

    # L1-E04: Safety content tagging
    has_safety_content = Column(Boolean, nullable=False, server_default="false")
    safety_signal_words = Column(JSONB, nullable=True)
    # L1-E03: SHA-256 hash of the chunk text for deduplication and integrity checks.
    chunk_text_hash = Column(String(64), nullable=True)
    # RAPTOR: IDs of child chunks this summary was built from. NULL for leaf chunks.
    source_chunk_ids = Column(JSONB, nullable=True, default=None)

    blacklisted = Column(Boolean, nullable=False, server_default="false")
    blacklisted_reason = Column(Text, nullable=True)
    blacklisted_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    blacklisted_at = Column(DateTime(timezone=True), nullable=True)

    document = relationship("DocumentMultimodal", back_populates="chunks")


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    thread_id = Column(
        String,
        ForeignKey("diagnostic_threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    response_id = Column(String, nullable=False, index=True)
    device_id = Column(String, nullable=False, index=True)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    badge = Column(String, nullable=False, index=True)
    quality_badge = Column(String, nullable=True, index=True)
    safety_badge = Column(String, nullable=True, index=True)
    # 1 = first inline attempt, 2 = retry. Multiple rows per response_id are
    # valid (the original attempt is preserved for audit even when the retry
    # supersedes it).
    attempt_number = Column(Integer, nullable=False, default=1, server_default="1")
    override = Column(Boolean, nullable=False, default=False)
    override_reason = Column(Text, nullable=True)

    sync_faithfulness = Column(Float, nullable=True)
    sync_answer_relevance = Column(Float, nullable=True)
    sync_context_relevance = Column(Float, nullable=True)
    sync_completeness = Column(Float, nullable=True)
    sync_reasons = Column(JSONB, nullable=False, default=dict)
    sync_duration_ms = Column(Integer, nullable=True)

    async_faithfulness = Column(Float, nullable=True)
    async_answer_relevance = Column(Float, nullable=True)
    async_context_relevance = Column(Float, nullable=True)
    async_completeness = Column(Float, nullable=True)
    async_evidence = Column(JSONB, nullable=True)
    async_duration_ms = Column(Integer, nullable=True)
    async_completed_at = Column(DateTime(timezone=True), nullable=True)
    # L2-EA09: Safety coverage score from shadow evaluator
    async_safety_coverage = Column(Float, nullable=True)

    thresholds_snapshot = Column(JSONB, nullable=False)
    step_verdicts = Column(JSONB, nullable=True)
    safety_survival = Column(JSONB, nullable=True)
    safety_confidence = Column(Float, nullable=True)  # deprecated — always null
    badge_source = Column(String(10), nullable=False, server_default="inline")
    shadow_step_verdicts = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    user = relationship("User")
    thread = relationship("DiagnosticThread")


class AgentExecutionLog(Base):
    __tablename__ = "agent_execution_logs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    trace_id = Column(String(36), nullable=True, index=True)
    thread_id = Column(String, nullable=True, index=True)
    document_id = Column(Integer, nullable=True, index=True)
    response_id = Column(String, nullable=True)
    operation_type = Column(String(20), nullable=False)
    node_name = Column(String, nullable=False)
    model = Column(String, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    started_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    record_hash = Column(String(64), nullable=True, index=True)
    chain_id = Column(String(36), nullable=True, index=True)


class EvaluationConfig(Base):
    __tablename__ = "evaluation_config"
    id = Column(SmallInteger, primary_key=True, default=1)
    thresholds = Column(JSONB, nullable=False)
    recommendations = Column(JSONB, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    updated_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    __table_args__ = (CheckConstraint("id = 1", name="check_eval_config_singleton"),)


class EvaluationConfigHistory(Base):
    __tablename__ = "evaluation_config_history"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    thresholds = Column(JSONB, nullable=False)
    changed_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    changed_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    change_reason = Column(Text, nullable=True)
    source = Column(String, nullable=False)


# --- Feedback Enums (Workstream R) ---
class FeedbackType(enum.Enum):
    THUMBS_UP = "THUMBS_UP"
    THUMBS_DOWN = "THUMBS_DOWN"


class StepFeedbackAction(str, enum.Enum):
    ACCEPT = "ACCEPT"
    SKIP = "SKIP"
    MODIFY = "MODIFY"
    INCORRECT = "INCORRECT"


class AntiExampleSeverity(enum.Enum):
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    SAFETY_CRITICAL = "SAFETY_CRITICAL"


class AntiExampleSource(enum.Enum):
    AUTO_REVIEW_VERDICT = "AUTO_REVIEW_VERDICT"
    MANUAL = "MANUAL"


class PreferencePairSource(enum.Enum):
    GATE_EDIT = "GATE_EDIT"
    REVIEW_VERDICT = "REVIEW_VERDICT"


class RuleSource(enum.Enum):
    SAFETY_RECALL = "SAFETY_RECALL"
    AUDIT_FINDING = "AUDIT_FINDING"
    ENGINEERING_ORDER = "ENGINEERING_ORDER"
    MANUAL = "MANUAL"


# --- Feedback Models ---
class ResponseFeedback(Base):
    __tablename__ = "response_feedback"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    response_id = Column(String, nullable=False, index=True)
    thread_id = Column(
        String,
        ForeignKey("diagnostic_threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    device_id = Column(String, nullable=False, index=True)
    equipment_category = Column(
        Enum(EquipmentCategory), nullable=False, default=EquipmentCategory.GENERAL
    )
    feedback_type = Column(Enum(FeedbackType), nullable=False)
    correction_text = Column(Text, nullable=True)
    correction_category = Column(String(20), nullable=True)
    satisfaction_rating = Column(SmallInteger, nullable=True)
    weight = Column(Float, nullable=False, default=1.0)
    # L3-E01: Audit enrichment fields
    operator_role = Column(String(20), nullable=True)
    time_on_feedback_ms = Column(Integer, nullable=True)
    retracted = Column(Boolean, nullable=False, server_default="false")
    retracted_reason = Column(Text, nullable=True)
    retracted_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    retracted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("response_id", "user_id", name="uq_resp_fb_response_user"),
    )


class StepFeedback(Base):
    __tablename__ = "step_feedback"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    response_id = Column(String, nullable=False, index=True)
    step_index = Column(Integer, nullable=False)
    thread_id = Column(
        String, ForeignKey("diagnostic_threads.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    device_id = Column(String, nullable=False)
    equipment_category = Column(
        Enum(EquipmentCategory), nullable=False, default=EquipmentCategory.GENERAL
    )
    action = Column(String(20), nullable=True)
    correction_text = Column(Text, nullable=True)
    # L3-E02: Step-level audit enrichment
    correction_type = Column(String(30), nullable=True)
    severity = Column(String(10), nullable=True)
    time_on_step_ms = Column(Integer, nullable=True)
    weight = Column(Float, nullable=False, default=1.0)
    retracted = Column(Boolean, nullable=False, server_default="false")
    retracted_reason = Column(Text, nullable=True)
    retracted_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    retracted_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "response_id", "step_index", "user_id", name="uq_step_fb_response_step_user"
        ),
    )


class SourceReport(Base):
    __tablename__ = "source_report"
    __table_args__ = (
        UniqueConstraint(
            "chunk_id", "response_id", "user_id", name="uq_src_rpt_chunk_response_user"
        ),
    )
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    chunk_id = Column(
        String,
        ForeignKey("document_chunks_multimodal.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    response_id = Column(String, nullable=False)
    thread_id = Column(
        String, ForeignKey("diagnostic_threads.id", ondelete="CASCADE"), nullable=False
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ResponseChunkLink(Base):
    __tablename__ = "response_chunk_link"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    response_id = Column(String, nullable=False, index=True)
    chunk_id = Column(
        String,
        ForeignKey("document_chunks_multimodal.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_index = Column(Integer, nullable=True)
    grounding_label = Column(String(20), nullable=True)
    similarity_score = Column(Float, nullable=True)
    reranker_score = Column(Integer, nullable=True)


class ChunkFeedbackCache(Base):
    __tablename__ = "chunk_feedback_cache"
    chunk_id = Column(
        String,
        ForeignKey("document_chunks_multimodal.id", ondelete="CASCADE"),
        primary_key=True,
    )
    positive_count = Column(Integer, nullable=False, default=0)
    negative_count = Column(Integer, nullable=False, default=0)
    weighted_positive = Column(Float, nullable=False, default=0.0)
    weighted_negative = Column(Float, nullable=False, default=0.0)
    net_score = Column(Float, nullable=False, default=0.0)
    last_recomputed = Column(DateTime(timezone=True), server_default=func.now())


# --- Preference Pairs ---
class PreferencePair(Base):
    __tablename__ = "preference_pair"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    source_type = Column(Enum(PreferencePairSource), nullable=False)
    source_id = Column(String, nullable=False)
    thread_id = Column(
        String, ForeignKey("diagnostic_threads.id", ondelete="SET NULL"), nullable=True
    )
    device_id = Column(String, nullable=False)
    equipment_category = Column(
        Enum(EquipmentCategory), nullable=False, default=EquipmentCategory.GENERAL
    )
    symptom_context = Column(Text, nullable=False)
    original_output = Column(Text, nullable=False)
    preferred_output = Column(Text, nullable=False)
    context_embedding = Column(Vector(3072), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("source_type", "source_id", name="uq_pref_pair_source"),
    )


# --- Prompt Amendments ---
class PromptAmendment(Base):
    __tablename__ = "prompt_amendment"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    target_prompt = Column(String, nullable=False, index=True)
    equipment_category = Column(Enum(EquipmentCategory), nullable=True)
    amendment_text = Column(Text, nullable=False)
    reason = Column(Text, nullable=False)
    is_active = Column(Boolean, nullable=False, server_default="false")
    approved_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    deactivated_at = Column(DateTime(timezone=True), nullable=True)
    pre_amendment_avg_score = Column(Float, nullable=True)
    post_amendment_avg_score = Column(Float, nullable=True)


# --- Anti-Examples ---
class AntiExample(Base):
    __tablename__ = "anti_example"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    symptom_pattern = Column(Text, nullable=False)
    symptom_embedding = Column(Vector(3072), nullable=True)
    incorrect_procedure = Column(Text, nullable=False)
    correct_alternative = Column(Text, nullable=True)
    severity = Column(
        Enum(AntiExampleSeverity), nullable=False, default=AntiExampleSeverity.MINOR
    )
    equipment_category = Column(
        Enum(EquipmentCategory), nullable=False, default=EquipmentCategory.GENERAL
    )
    device_id = Column(String, nullable=True)
    source_type = Column(
        Enum(AntiExampleSource), nullable=False, default=AntiExampleSource.MANUAL
    )
    source_id = Column(String, nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")
    requires_second_approval = Column(Boolean, nullable=False, server_default="false")
    approved_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    second_approved_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    superseded_by = Column(
        UUID(as_uuid=True),
        ForeignKey("anti_example.id", ondelete="SET NULL"),
        nullable=True,
    )
    superseded_at = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# --- Deterministic Rules ---
class DeterministicRule(Base):
    __tablename__ = "deterministic_rule"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    device_id = Column(String, nullable=True)
    equipment_category = Column(
        Enum(EquipmentCategory), nullable=False, default=EquipmentCategory.GENERAL
    )
    trigger_keywords = Column(ARRAY(String), nullable=False)
    override_response = Column(Text, nullable=False)
    override_safety_protocols = Column(Text, nullable=True)
    priority = Column(Integer, nullable=False, default=0)
    source = Column(Enum(RuleSource), nullable=False, default=RuleSource.MANUAL)
    is_active = Column(Boolean, nullable=False, server_default="false")
    approved_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    approved_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    review_date = Column(DateTime(timezone=True), nullable=True)
    created_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# --- Unlearning Audit Log ---
class UnlearningAuditLog(Base):
    __tablename__ = "unlearning_audit_log"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    event_type = Column(String, nullable=False, index=True)
    target_type = Column(String, nullable=False, index=True)
    target_id = Column(String, nullable=False)
    performed_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    reason = Column(Text, nullable=True)
    # `metadata` is reserved by SQLAlchemy declarative; map Python attr to "metadata" DB column.
    event_metadata = Column("metadata", JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class GateType(enum.Enum):
    RISK = "RISK"


class GateStatus(enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    OVERRIDDEN = "OVERRIDDEN"
    EDITED = "EDITED"


class InterruptGate(Base):
    __tablename__ = "interrupt_gates"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    thread_id = Column(
        String,
        ForeignKey("diagnostic_threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    response_id = Column(String, nullable=True)
    gate_type = Column(Enum(GateType), nullable=False)
    status = Column(
        Enum(GateStatus), nullable=False, server_default=text("'PENDING'::gatestatus")
    )
    gate_payload = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    resolution_payload = Column(JSONB, nullable=True)
    resolved_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class EscalationFlag(Base):
    """Unified async review-task table.

    Receives auto-flags from safety evaluation (AUTO_SAFETY) and InlineEvaluator
    (AUTO_EVAL), manual flags from technicians (MANUAL_FLAG), and expert feedback
    (EXPERT_FEEDBACK). Rows surface work to admin/expert dashboards.
    """

    __tablename__ = "escalation_flags"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    thread_id = Column(
        String,
        ForeignKey("diagnostic_threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    response_id = Column(String, nullable=True)
    flagged_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    assigned_reviewer = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # --- Async review queue extensions ---
    source = Column(
        Enum(ReviewTaskSource),
        nullable=False,
        default=ReviewTaskSource.MANUAL_FLAG,
        server_default=text("'MANUAL_FLAG'::reviewtasksource"),
    )
    priority = Column(
        Enum(ReviewTaskPriority),
        nullable=False,
        default=ReviewTaskPriority.P2,
        server_default=text("'P2'::reviewtaskpriority"),
    )
    status = Column(
        Enum(ReviewTaskStatus),
        nullable=False,
        default=ReviewTaskStatus.NEW,
        server_default=text("'NEW'::reviewtaskstatus"),
    )
    trigger_reason = Column(Text, nullable=True)
    payload_snapshot = Column(JSONB, nullable=False, server_default=text("'{}'::jsonb"))
    claimed_by = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    claimed_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    snoozed_until = Column(DateTime(timezone=True), nullable=True)


class Notification(Base):
    __tablename__ = "notifications"
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind = Column(Enum(NotificationKind), nullable=False)
    title = Column(Text, nullable=False)
    body = Column(Text, nullable=True)
    link = Column(Text, nullable=True)
    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("escalation_flags.id", ondelete="CASCADE"),
        nullable=True,
    )
    read_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ReviewVerdict(Base):
    __tablename__ = "review_verdicts"
    __table_args__ = (
        Index(
            "uq_verdict_per_step_reviewer",
            "thread_id",
            "response_id",
            "node_name",
            text("COALESCE(step_index, -1)"),
            "reviewer_id",
            unique=True,
        ),
    )
    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    thread_id = Column(
        String,
        ForeignKey("diagnostic_threads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    response_id = Column(String, nullable=False, index=True)
    node_name = Column(String, nullable=False)
    step_index = Column(Integer, nullable=True)
    verdict = Column(Enum(VerdictEnum), nullable=False)
    correct_procedure = Column(Text, nullable=True)
    error_category = Column(Enum(ErrorCategory), nullable=True)
    reviewer_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# --- Audit Logging (P3) ---


class IngestionManifest(Base):
    """Append-only hash-chain audit log for document ingestion events.

    Each entry records a single ingestion event (doc_registered, chunk_created,
    embedding_generated) with a SHA-256 hash chaining it to the previous entry.
    UPDATE/DELETE are blocked by a DB trigger (see migrate_audit_logging.py).
    """

    __tablename__ = "ingestion_manifest"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    entry_sequence = Column(Integer, nullable=False)
    entry_hash = Column(String(64), nullable=False)
    previous_hash = Column(String(64), nullable=False)
    entry_type = Column(String(30), nullable=False)
    entry_data = Column(JSONB, nullable=False)
    entry_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    document_id = Column(
        Integer,
        ForeignKey("documents_multimodal.id", ondelete="SET NULL"),
        nullable=True,
    )
