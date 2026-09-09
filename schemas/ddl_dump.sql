-- DDL dump for audit-relevant tables
-- Generated from SQLAlchemy models in the companion project

-- Enum types used by these tables
CREATE TYPE userrole AS ENUM ('ADMIN', 'TECHNICIAN');
CREATE TYPE technicianlevel AS ENUM ('JUNIOR', 'SENIOR');
CREATE TYPE documentstatus AS ENUM ('PENDING', 'PROCESSING', 'READY', 'ERROR');
CREATE TYPE equipmentcategory AS ENUM ('GENERAL', 'CONVEYOR', 'SCANNER', 'PRINTER', 'SCALE', 'SORTER');
CREATE TYPE feedbacktype AS ENUM ('THUMBS_UP', 'THUMBS_DOWN', 'CORRECTION');
CREATE TYPE threadterminalstate AS ENUM ('IN_PROGRESS', 'COMPLETED', 'ABANDONED', 'ESCALATED');
CREATE TYPE reviewstatus AS ENUM ('PENDING_REVIEW', 'REVIEWED', 'FLAGGED');


-- =============================================================================
-- USERS (referenced by FK from multiple audit tables)
-- =============================================================================
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR NOT NULL UNIQUE,
    password_hash   VARCHAR NOT NULL,
    role            userrole NOT NULL,
    level           technicianlevel,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_users_email ON users (email);


-- =============================================================================
-- L1: DOCUMENT LAYER
-- =============================================================================

CREATE TABLE documents_multimodal (
    id                  SERIAL PRIMARY KEY,
    original_filename   VARCHAR,
    intuitive_name      VARCHAR,
    filepath            VARCHAR UNIQUE,
    doc_type            VARCHAR DEFAULT 'manual',
    upload_date         TIMESTAMPTZ NOT NULL DEFAULT now(),
    file_hash           VARCHAR UNIQUE,
    status              documentstatus NOT NULL DEFAULT 'PENDING',
    compatible_models   VARCHAR[] NOT NULL DEFAULT '{}',
    equipment_category  equipmentcategory NOT NULL DEFAULT 'GENERAL',
    doc_format          VARCHAR
);
CREATE INDEX ix_documents_multimodal_id ON documents_multimodal (id);
CREATE INDEX ix_documents_multimodal_original_filename ON documents_multimodal (original_filename);
CREATE INDEX ix_documents_multimodal_filepath ON documents_multimodal (filepath);
CREATE INDEX ix_documents_multimodal_doc_type ON documents_multimodal (doc_type);
CREATE INDEX ix_documents_multimodal_file_hash ON documents_multimodal (file_hash);


CREATE TABLE document_chunks_multimodal (
    id                  VARCHAR PRIMARY KEY,
    document_id         INTEGER REFERENCES documents_multimodal(id) ON DELETE CASCADE,
    text                TEXT,
    embedding           VECTOR(3072),
    level               INTEGER,
    pages               VARCHAR,
    page_start          INTEGER,
    page_end            INTEGER,
    sequence_index      INTEGER,
    section_header      TEXT,
    parent_id           VARCHAR REFERENCES document_chunks_multimodal(id) ON DELETE CASCADE,
    section_code        VARCHAR,
    images              JSONB DEFAULT '[]',
    tsv                 TSVECTOR,

    -- Safety tagging (L1-E05)
    has_safety_content  BOOLEAN NOT NULL DEFAULT false,
    safety_signal_words JSONB,
    chunk_text_hash     VARCHAR(64),

    -- RAPTOR lineage
    source_chunk_ids    JSONB DEFAULT NULL,

    -- Blacklisting (R7)
    blacklisted         BOOLEAN NOT NULL DEFAULT false,
    blacklisted_reason  TEXT,
    blacklisted_by      UUID REFERENCES users(id) ON DELETE SET NULL,
    blacklisted_at      TIMESTAMPTZ
);
CREATE INDEX ix_chunks_document_id ON document_chunks_multimodal (document_id);
CREATE INDEX ix_chunks_level ON document_chunks_multimodal (level);
CREATE INDEX ix_chunks_parent_id ON document_chunks_multimodal (parent_id);
CREATE INDEX ix_chunks_section_code ON document_chunks_multimodal (section_code);
CREATE INDEX ix_chunks_safety ON document_chunks_multimodal (has_safety_content) WHERE has_safety_content = true;


CREATE TABLE ingestion_manifest (
    id              BIGSERIAL PRIMARY KEY,
    entry_sequence  INTEGER NOT NULL,
    entry_hash      VARCHAR(64) NOT NULL,
    previous_hash   VARCHAR(64) NOT NULL,
    entry_type      VARCHAR(30) NOT NULL,
    entry_data      JSONB NOT NULL,
    entry_timestamp TIMESTAMPTZ DEFAULT now(),
    document_id     INTEGER REFERENCES documents_multimodal(id) ON DELETE SET NULL
);
CREATE INDEX ix_manifest_document_id ON ingestion_manifest (document_id);
CREATE INDEX ix_manifest_entry_type ON ingestion_manifest (entry_type);

-- Immutability trigger: block UPDATE and DELETE on ingestion_manifest
CREATE OR REPLACE FUNCTION trg_manifest_immutable_fn() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'ingestion_manifest is append-only: % not permitted', TG_OP;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_manifest_immutable
    BEFORE UPDATE OR DELETE ON ingestion_manifest
    FOR EACH ROW EXECUTE FUNCTION trg_manifest_immutable_fn();


-- =============================================================================
-- L2: EXECUTION LAYER
-- =============================================================================

CREATE TABLE diagnostic_threads (
    id                  VARCHAR PRIMARY KEY,
    user_id             UUID REFERENCES users(id) ON DELETE SET NULL,
    device_id           VARCHAR NOT NULL,
    terminal_state      threadterminalstate NOT NULL DEFAULT 'IN_PROGRESS',
    node_timestamps     JSONB NOT NULL DEFAULT '{}'::jsonb,
    outcome_resolved    BOOLEAN,
    outcome_recorded_at TIMESTAMPTZ,
    review_status       reviewstatus NOT NULL DEFAULT 'PENDING_REVIEW'::reviewstatus,
    reviewed_by         UUID REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at         TIMESTAMPTZ,
    is_escalated        BOOLEAN NOT NULL DEFAULT false,
    risk_tier           INTEGER DEFAULT 0,
    created_at          TIMESTAMPTZ DEFAULT now(),
    updated_at          TIMESTAMPTZ DEFAULT now(),
    flagged_for_review  BOOLEAN NOT NULL DEFAULT false
);
CREATE INDEX ix_diagnostic_threads_id ON diagnostic_threads (id);
CREATE INDEX ix_diagnostic_threads_user_id ON diagnostic_threads (user_id);
CREATE INDEX ix_diagnostic_threads_device_id ON diagnostic_threads (device_id);
CREATE INDEX ix_diagnostic_threads_terminal_state ON diagnostic_threads (terminal_state);
CREATE INDEX ix_diagnostic_threads_review_status ON diagnostic_threads (review_status);


CREATE TABLE agent_execution_logs (
    id              BIGSERIAL PRIMARY KEY,
    trace_id        VARCHAR(36),
    thread_id       VARCHAR,
    document_id     INTEGER,
    response_id     VARCHAR,
    operation_type  VARCHAR(20) NOT NULL,
    node_name       VARCHAR NOT NULL,
    model           VARCHAR,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    latency_ms      INTEGER,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ,
    error           TEXT,
    metadata        JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    record_hash     VARCHAR(64),
    chain_id        VARCHAR(36)
);
CREATE INDEX ix_ael_trace_id ON agent_execution_logs (trace_id);
CREATE INDEX ix_ael_thread_id ON agent_execution_logs (thread_id);
CREATE INDEX ix_ael_document_id ON agent_execution_logs (document_id);
CREATE INDEX ix_ael_operation_type_created ON agent_execution_logs (operation_type, created_at DESC);
CREATE INDEX ix_ael_record_hash ON agent_execution_logs (record_hash);
CREATE INDEX ix_ael_chain_id ON agent_execution_logs (chain_id);
CREATE INDEX ix_ael_trace_started ON agent_execution_logs (trace_id, started_at);


CREATE TABLE evaluation_metrics (
    id                      BIGSERIAL PRIMARY KEY,
    thread_id               VARCHAR NOT NULL REFERENCES diagnostic_threads(id) ON DELETE CASCADE,
    response_id             VARCHAR NOT NULL,
    device_id               VARCHAR NOT NULL,
    user_id                 UUID REFERENCES users(id) ON DELETE SET NULL,
    badge                   VARCHAR NOT NULL,
    quality_badge           VARCHAR,
    safety_badge            VARCHAR,
    attempt_number          INTEGER NOT NULL DEFAULT 1,
    override                BOOLEAN NOT NULL DEFAULT false,
    override_reason         TEXT,

    -- Inline (sync) RAGAS scores
    sync_faithfulness       DOUBLE PRECISION,
    sync_answer_relevance   DOUBLE PRECISION,
    sync_context_relevance  DOUBLE PRECISION,
    sync_completeness       DOUBLE PRECISION,
    sync_reasons            JSONB NOT NULL DEFAULT '{}',
    sync_duration_ms        INTEGER,

    -- Shadow (async) RAGAS scores
    async_faithfulness      DOUBLE PRECISION,
    async_answer_relevance  DOUBLE PRECISION,
    async_context_relevance DOUBLE PRECISION,
    async_completeness      DOUBLE PRECISION,
    async_evidence          JSONB,
    async_duration_ms       INTEGER,
    async_completed_at      TIMESTAMPTZ,
    async_safety_coverage   DOUBLE PRECISION,

    -- Evaluation metadata
    thresholds_snapshot     JSONB NOT NULL,
    step_verdicts           JSONB,
    safety_survival         JSONB,
    safety_confidence       DOUBLE PRECISION,   -- deprecated, always null
    badge_source            VARCHAR(10) NOT NULL DEFAULT 'inline',
    shadow_step_verdicts    JSONB,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_eval_metrics_thread_id ON evaluation_metrics (thread_id);
CREATE INDEX ix_eval_metrics_response_id ON evaluation_metrics (response_id);
CREATE INDEX ix_eval_metrics_device_id ON evaluation_metrics (device_id);
CREATE INDEX ix_eval_metrics_user_id ON evaluation_metrics (user_id);
CREATE INDEX ix_eval_metrics_badge ON evaluation_metrics (badge);
CREATE INDEX ix_eval_metrics_quality_badge ON evaluation_metrics (quality_badge);
CREATE INDEX ix_eval_metrics_safety_badge ON evaluation_metrics (safety_badge);


CREATE TABLE evaluation_config (
    id              SMALLINT PRIMARY KEY DEFAULT 1,
    thresholds      JSONB NOT NULL,
    recommendations JSONB,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by      UUID REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT check_eval_config_singleton CHECK (id = 1)
);


CREATE TABLE response_chunk_link (
    id              BIGSERIAL PRIMARY KEY,
    response_id     VARCHAR NOT NULL,
    chunk_id        VARCHAR NOT NULL REFERENCES document_chunks_multimodal(id) ON DELETE CASCADE,
    step_index      INTEGER,
    grounding_label VARCHAR(20),
    similarity_score DOUBLE PRECISION,
    reranker_score  INTEGER
);
CREATE INDEX ix_rcl_response_id ON response_chunk_link (response_id);
CREATE INDEX ix_rcl_chunk_id ON response_chunk_link (chunk_id);
CREATE INDEX ix_rcl_response_step ON response_chunk_link (response_id, step_index);


-- =============================================================================
-- L3: FEEDBACK LAYER
-- =============================================================================

CREATE TABLE response_feedback (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id         VARCHAR NOT NULL,
    thread_id           VARCHAR NOT NULL REFERENCES diagnostic_threads(id) ON DELETE CASCADE,
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id           VARCHAR NOT NULL,
    equipment_category  equipmentcategory NOT NULL DEFAULT 'GENERAL',
    feedback_type       feedbacktype NOT NULL,
    correction_text     TEXT,
    correction_category VARCHAR(20),
    satisfaction_rating SMALLINT,
    weight              DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    operator_role       VARCHAR(20),
    time_on_feedback_ms INTEGER,
    retracted           BOOLEAN NOT NULL DEFAULT false,
    retracted_reason    TEXT,
    retracted_by        UUID REFERENCES users(id) ON DELETE SET NULL,
    retracted_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_resp_fb_response_user UNIQUE (response_id, user_id)
);
CREATE INDEX ix_resp_fb_response_id ON response_feedback (response_id);
CREATE INDEX ix_resp_fb_thread_id ON response_feedback (thread_id);
CREATE INDEX ix_resp_fb_user_id ON response_feedback (user_id);
CREATE INDEX ix_resp_fb_device_id ON response_feedback (device_id);


CREATE TABLE step_feedback (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id         VARCHAR NOT NULL,
    step_index          INTEGER NOT NULL,
    thread_id           VARCHAR NOT NULL REFERENCES diagnostic_threads(id) ON DELETE CASCADE,
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_id           VARCHAR NOT NULL,
    equipment_category  equipmentcategory NOT NULL DEFAULT 'GENERAL',
    action              VARCHAR(20),
    correction_text     TEXT,
    correction_type     VARCHAR(30),
    severity            VARCHAR(10),
    time_on_step_ms     INTEGER,
    weight              DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    retracted           BOOLEAN NOT NULL DEFAULT false,
    retracted_reason    TEXT,
    retracted_by        UUID REFERENCES users(id) ON DELETE SET NULL,
    retracted_at        TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_step_fb_response_step_user UNIQUE (response_id, step_index, user_id)
);
CREATE INDEX ix_step_fb_response_id ON step_feedback (response_id);
CREATE INDEX ix_step_fb_response_step ON step_feedback (response_id, step_index);
CREATE INDEX ix_step_fb_thread_id ON step_feedback (thread_id);
