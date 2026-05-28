#!/usr/bin/env bash
# ============================================================================
# Restore evaluation dataset to a local Postgres instance.
#
# Prerequisites:
#   - Docker container "hitl_postgres" running pgvector/pgvector:pg18
#   - gunzip available on PATH
#
# Usage (from repo root):
#   bash schemas/data/restore.sh [container_name] [dbname]
#
# Defaults: container=hitl_postgres, db=hitldss
# ============================================================================
set -euo pipefail

CONTAINER="${1:-hitl_postgres}"
DBNAME="${2:-hitldss}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_DIR="/tmp/eval_restore"

psql_run() { docker exec -i "$CONTAINER" psql -U postgres -d "$DBNAME" "$@"; }

echo "==> Creating schema (idempotent) ..."
psql_run -v ON_ERROR_STOP=1 <<'DDL'
CREATE EXTENSION IF NOT EXISTS vector;

DO $$ BEGIN CREATE TYPE documentstatus AS ENUM ('PENDING', 'PROCESSING', 'READY', 'ERROR'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE equipmentcategory AS ENUM ('HEAVY_INDUSTRIAL', 'CONSUMER_ELECTRONICS', 'IT_NETWORKING', 'GENERAL'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE TABLE IF NOT EXISTS documents_multimodal (
    id                 SERIAL PRIMARY KEY,
    original_filename  VARCHAR,
    intuitive_name     VARCHAR,
    filepath           VARCHAR UNIQUE,
    doc_type           VARCHAR,
    upload_date        TIMESTAMPTZ NOT NULL DEFAULT now(),
    file_hash          VARCHAR UNIQUE,
    status             documentstatus NOT NULL DEFAULT 'PENDING',
    compatible_models  VARCHAR[] NOT NULL DEFAULT '{}',
    equipment_category equipmentcategory NOT NULL DEFAULT 'GENERAL',
    doc_format         VARCHAR
);

CREATE TABLE IF NOT EXISTS document_chunks_multimodal (
    id                 VARCHAR PRIMARY KEY,
    document_id        INTEGER REFERENCES documents_multimodal(id),
    text               TEXT,
    level              INTEGER,
    pages              VARCHAR,
    images             JSONB,
    embedding          VECTOR(3072),
    page_start         INTEGER,
    page_end           INTEGER,
    sequence_index     INTEGER,
    section_header     TEXT,
    tsv                TSVECTOR,
    blacklisted        BOOLEAN NOT NULL DEFAULT false,
    blacklisted_reason TEXT,
    blacklisted_by     UUID,
    blacklisted_at     TIMESTAMPTZ,
    parent_id          VARCHAR,
    section_code       VARCHAR
);

CREATE TABLE IF NOT EXISTS evaluation_config (
    id              INTEGER PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    thresholds      JSONB,
    recommendations JSONB,
    updated_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS ingestion_manifest (
    id              BIGSERIAL PRIMARY KEY,
    entry_sequence  BIGINT NOT NULL,
    entry_hash      VARCHAR(64) NOT NULL,
    previous_hash   VARCHAR(64) NOT NULL,
    entry_type      VARCHAR NOT NULL,
    entry_data      JSONB NOT NULL,
    entry_timestamp TIMESTAMPTZ NOT NULL DEFAULT now(),
    document_id     INTEGER REFERENCES documents_multimodal(id)
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_level ON document_chunks_multimodal (document_id, level);
CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON document_chunks_multimodal (parent_id) WHERE parent_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_section_code ON document_chunks_multimodal (section_code) WHERE section_code IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_chunks_page_start ON document_chunks_multimodal (page_start);
DDL

echo "==> Copying compressed data into container ..."
docker exec "$CONTAINER" mkdir -p "$REMOTE_DIR"
for f in documents_multimodal.csv.gz document_chunks_multimodal.csv.gz evaluation_config.csv.gz; do
    docker cp "$SCRIPT_DIR/$f" "$CONTAINER:$REMOTE_DIR/$f"
done

echo "==> Decompressing inside container ..."
docker exec "$CONTAINER" bash -c "cd $REMOTE_DIR && gunzip -f *.gz"

echo "==> Clearing existing rows for document_id IN (87, 88) ..."
psql_run -c "DELETE FROM document_chunks_multimodal WHERE document_id IN (87, 88)" 2>/dev/null || true
psql_run -c "DELETE FROM documents_multimodal WHERE id IN (87, 88)" 2>/dev/null || true
psql_run -c "DELETE FROM evaluation_config WHERE id = 1" 2>/dev/null || true

echo "==> Loading documents_multimodal (2 rows) ..."
psql_run -v ON_ERROR_STOP=1 -c "\COPY documents_multimodal FROM '$REMOTE_DIR/documents_multimodal.csv' WITH (FORMAT csv, HEADER)"

echo "==> Loading document_chunks_multimodal (5659 rows — takes a moment) ..."
psql_run -v ON_ERROR_STOP=1 -c "\COPY document_chunks_multimodal FROM '$REMOTE_DIR/document_chunks_multimodal.csv' WITH (FORMAT csv, HEADER)"

echo "==> Loading evaluation_config (1 row) ..."
psql_run -v ON_ERROR_STOP=1 -c "\COPY evaluation_config FROM '$REMOTE_DIR/evaluation_config.csv' WITH (FORMAT csv, HEADER)"

echo "==> Building HNSW index (may take 30-60s) ..."
psql_run -v ON_ERROR_STOP=1 -c "
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
    ON document_chunks_multimodal
    USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops)
    WITH (m=16, ef_construction=64);
"

echo "==> Cleaning up container temp files ..."
docker exec "$CONTAINER" rm -rf "$REMOTE_DIR"

echo "==> Backfilling ingestion manifest (hash-chain audit trail) ..."
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
if [ -f "$SCRIPT_REPO_ROOT/.venv/bin/python" ]; then
    PYTHON="$SCRIPT_REPO_ROOT/.venv/bin/python"
fi
DATABASE_URL="${DATABASE_URL:-postgresql://postgres:postgrespassword@localhost:5432/$DBNAME}" \
    "$PYTHON" "$SCRIPT_REPO_ROOT/schemas/data/backfill_manifest.py"

echo ""
echo "==> Verifying row counts ..."
psql_run -c "
SELECT 'documents_multimodal' AS table_name, count(*) FROM documents_multimodal WHERE id IN (87, 88)
UNION ALL
SELECT 'document_chunks_multimodal', count(*) FROM document_chunks_multimodal WHERE document_id IN (87, 88)
UNION ALL
SELECT 'evaluation_config', count(*) FROM evaluation_config WHERE id = 1
UNION ALL
SELECT 'ingestion_manifest', count(*) FROM ingestion_manifest WHERE document_id IN (87, 88);
"

echo ""
echo "Done. Evaluation dataset restored."
