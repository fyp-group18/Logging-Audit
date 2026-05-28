# Evaluation Reproduction Guide

Evaluation scripts and metric computation code for:

> **Cross-Layer Provenance for Agentic RAG: Preserving Identity Continuity Across Ingestion, Execution, and Feedback**
> Zhi An Ling, Chi Yung Siew, Miko May Lee Chang

Validated on a 12-node LangGraph diagnostic decision-support system for aircraft corrective maintenance.

## Scope and Limitations

This repository provides **methodological transparency**, not push-button reproducibility. Every script that produced a reported number is included and inspectable. However, the scripts require:

- A PostgreSQL 15+ database with the audit schema (`schemas/ddl_dump.sql`) and session data. The two source documents are publicly available in the [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval) and can be ingested via the companion application
- The [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval) (pinned to commit [`27723ac`](https://github.com/fyp-group18/aircraft-maintenance-rag-eval/tree/27723ac8e9bef27ba61e8e2360c2b8a322a135ad))

Corpus generation (`run_eval_queries.py`) requires a live instance of the companion application to generate the session corpus. All other scripts — provenance metrics and failure injection — operate directly against the database.

## Prerequisites

- Python 3.11+
- PostgreSQL 15+ with `pgvector`, populated by the companion application
- `psycopg[binary]` (v3) — installed via `requirements.txt`
- Environment variable: `DATABASE_URL=postgresql://user:password@host:5432/dbname`

```bash
pip install -r requirements.txt
```

## Repository Structure

```
├── audit/                          # Metric computation (§Evaluation)
│   ├── l1_metrics.py               #   Leaf Coverage@d, KB Reconstructability@t
│   ├── l2_metrics.py               #   L2 response-chunk link metrics
│   ├── l3_metrics.py               #   L3 feedback chain metrics
│   ├── cross_layer_metrics.py      #   Provenance Completeness, Safety Provenance
│   ├── c3_eval_metrics.py          #   C3 cross-layer evaluation metrics
│   ├── manifest.py                 #   SHA-256 hash-chain append/verify
│   └── process_mining.py           #   Token-replay fitness
│
├── core/                           # Database and application layer
│   ├── database.py                 #   SessionLocal(), with_db_retry, lazy engine
│   ├── models.py                   #   SQLAlchemy ORM models
│   ├── crud.py                     #   Database queries (semantic search, chunk retrieval)
│   ├── config.py                   #   Configuration from environment variables
│   └── ...                         #   hash_chain, eval_config, security, storage, utils
│
├── api/                            # FastAPI REST layer
│   ├── agent_manager.py            #   LangGraph agent lifecycle
│   ├── routers/                    #   diagnostics, evaluation, trace endpoints
│   └── schemas.py                  #   Request/response models
│
├── pipeline/                       # LangGraph pipeline (12-node diagnostic workflow)
│   ├── nodes.py                    #   Node implementations
│   ├── workflow.py                 #   Graph construction
│   ├── state.py                    #   State schema
│   ├── eval_logic.py               #   Inline evaluation logic
│   ├── safety_judge_gate.py        #   Safety classification gate
│   └── telemetry.py                #   Execution telemetry
│
├── modules/                        # Retrieval modules (embeddings, reranking, MMR)
│
├── prompts/                        # System prompts for the pipeline
│
├── eval/
│   ├── generation/
│   │   ├── eval_queries.json       #   30-query evaluation corpus
│   │   └── eval_queries_v2.json    #   150-query extended corpus
│   ├── scripts/                    #   Evaluation pipeline (see sections below)
│   ├── shared/                     #   Shared metric/stats utilities
│   └── results/
│       └── metrics/                #   Aggregated results (no raw data)
│
├── schemas/
│   ├── ddl_dump.sql                #   PostgreSQL DDL for 10 audit tables
│   └── data/                       #   Seed data + restore script (see below)
│
├── datasets/
│   └── aircraft-maintenance-rag-eval  # Evaluation dataset (git submodule)
│
└── tests/                          # Unit tests
```

## Application Layer

The `core/`, `api/`, `pipeline/`, `modules/`, and `prompts/` directories contain the LangGraph diagnostic decision-support system described in the paper. This code is included for reproducibility — corpus generation executes queries against this application to produce the session corpus.

- `app.py` — FastAPI entry point (`uvicorn app:app`)
- `core/database.py` — `SessionLocal()`, `with_db_retry`; `DATABASE_URL` is read lazily from the environment, requiring the `psycopg` (v3) driver
- `pipeline/workflow.py` — constructs the 12-node LangGraph workflow

## Corpus Generation

Produces the 150-session corpus. All metrics are computed from this corpus.

**Paper reference:** §III-A Evaluation Environment

```bash
# Query selection (deterministic, SEED=42)
python eval/scripts/select_eval_queries.py

# Execute against the live companion application
python eval/scripts/run_eval_queries.py
```

## Provenance Metrics

**Paper reference:** §IV-A Metrics, §IV-B Provenance Reconstruction (Table III)

All modules require `DATABASE_URL` at runtime (not at import time).

```bash
export DATABASE_URL=postgresql://user:password@host:5432/dbname

# L1: Leaf Coverage@d, KB Reconstructability@t, Manifest Integrity
python -m audit.l1_metrics

# Cross-layer: Provenance Completeness, Safety Provenance
python -m audit.cross_layer_metrics

# Process mining: Token-replay fitness
python -m audit.process_mining

# All metrics (provenance, feedback chain, failure injection,
# identity triple, overhead)
python -m eval.scripts.compute_metrics
```

### Reported results (Table III)

| Metric | Value | Source module |
|---|---|---|
| Provenance Completeness | 100.00% (399/399) | `cross_layer_metrics.py` |
| Safety Provenance | 100.00% (157/157) | `cross_layer_metrics.py` |
| KB Reconstructability@t | 1.0000 (399/399) | `l1_metrics.py` |
| Leaf Coverage@d | 0.9311 ± 0.0018 | `l1_metrics.py` |
| Manifest integrity | 2/2 valid | `manifest.py` |
| Process mining fitness | 1.0000 (27/27) | `process_mining.py` |

## Failure Injection Protocol

Adversarial integrity testing of the provenance architecture. 12 sessions (6 injected, 6 control), 4 failure types: chunk deletion, score perturbation, log gap, manifest tamper.

**Paper reference:** §IV-C Failure Injection

### Protocol

1. **Select sessions** from the evaluation corpus:
   ```bash
   python eval/scripts/select_injection_sessions.py
   ```

2. **Randomize failures** — assign types, generate SHA-256 sealed manifest:
   ```bash
   python eval/scripts/randomize_failures.py
   ```

3. **Apply injections** — each mutation applied and reverted individually:
   ```bash
   python eval/scripts/run_failure_injection.py
   ```

4. **Blinded diagnosis** — evaluator classifies sessions from exported artifacts only:
   ```bash
   python eval/scripts/run_diagnostics.py
   ```

5. **Unblind and score**:
   ```bash
   python eval/scripts/score_and_unblind.py
   ```

6. **Verify reverts**:
   ```bash
   python eval/scripts/verify_reverts.py
   ```

### Reported results

- Blinded accuracy: 0.83 (10/12), F1 = 0.83, 95% Clopper-Pearson CI [0.52, 0.97]
- 3/4 failure types detected: chunk deletion (2/2), score perturbation (1/1), log gap (2/2)
- Manifest tamper (0/1 correct localization): document-level hash chain blast radius prevents per-session attribution; per-chunk hashing proposed as architectural fix

### Integrity disclosure

The injection designer and the blinded evaluator belong to the same research team. Blinding was procedural (randomized session order, sealed manifest), not organizational. Documented as threat (1) in the paper.

## Human Evaluation

One evaluator assessed all 150 responses across four dimensions (correctness, completeness, safety, overall) and 2,707 individual repair steps. The evaluation harness (`human_evaluate.py`) presented each response with retrieved chunks, inline evaluation verdicts, and a recommendation; the evaluator confirmed or overrode each verdict. As a single-evaluator design, no inter-rater reliability is reported; this is documented as a limitation in the paper.

Evaluation data (response-level and step-level verdicts) and the computed metrics are available in `eval/results/metrics/`. Raw evaluation sheets containing production identifiers and copyrighted response text are excluded from this repository.

## Seed Data

`schemas/data/` contains structural metadata for the two evaluation documents and their chunks. Text content and embedding vectors have been stripped from `document_chunks_multimodal.csv.gz` for copyright reasons — the source documents are proprietary aircraft maintenance manuals. The retained columns (chunk ID, document ID, level, page range, section header, parent ID) are sufficient to verify provenance chain integrity and reproduce metric computation against a populated database.

To restore the seed data into a local PostgreSQL instance:

```bash
bash schemas/data/restore.sh          # default: container=hitl_postgres, db=hitldss
```

To obtain the full chunk text, ingest the source documents using the included pipeline (`app.py`) with the publicly available [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval).

## Database Schema

`schemas/ddl_dump.sql` contains the DDL for 10 audit tables. Cross-layer FK chains:

- **L2→L1:** `response_chunk_link.chunk_id` → `document_chunks_multimodal.document_id` → `ingestion_manifest.document_id`
- **L3→L2:** `step_feedback.response_id` → `evaluation_metrics.response_id` → `response_chunk_link.response_id`

## Citation

```bibtex
@inproceedings{ling2026crosslayer,
  title     = {Cross-Layer Provenance for Agentic RAG: Preserving Identity Continuity Across Ingestion, Execution, and Feedback},
  author    = {Ling, Zhi An and Siew, Chi Yung and Chang, Miko May Lee},
  booktitle = {TODO},
  year      = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
