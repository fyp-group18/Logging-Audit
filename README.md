# Evaluation Reproduction Guide

Evaluation scripts and metric computation code for:

> **Cross-Layer Provenance for Agentic RAG: Preserving Identity Continuity Across Ingestion, Execution, and Feedback**
> Zhi An Ling, Joanne Wan Chuin Jong, Alfi Diyana Ngu Binti Mohd Faris Ngu, Miko May Lee Chang
> IEEE MCAIT 2026

Validated on a 12-node LangGraph diagnostic decision-support system for aircraft corrective maintenance.

## Scope and Limitations

This repository provides **methodological transparency**, not push-button reproducibility. Every script that produced a reported number is included and inspectable. However, the scripts require:

- A PostgreSQL 15+ database with the audit schema (`schemas/ddl_dump.sql`) and session data. The two source documents are publicly available in the [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval) and can be ingested via the companion application
- The [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval) (pinned to commit [`27723ac`](https://github.com/fyp-group18/aircraft-maintenance-rag-eval/tree/27723ac8e9bef27ba61e8e2360c2b8a322a135ad))

Corpus generation (`run_eval_queries.py`) requires a live instance of the companion application to generate the session corpus. All other scripts — provenance metrics and failure injection — operate directly against the database.

**Artifacts that are not redistributed here.** The raw run log, the human-evaluation sheets, and the sealed failure-injection artifacts carry production identifiers and verbatim text from copyrighted maintenance manuals. What ships instead is the computed metrics under `eval/results/metrics/` and a redacted run log (see [Evaluation Corpus](#evaluation-corpus)). Two consequences worth stating plainly:

- `compute_metrics.py` cannot run end to end from a clean clone: it reads the raw `eval/results/eval_run_v2.json`, which is not included.
- The Human–LLM concordance figure requires `eval/results/human_eval/evaluator1_responses.csv`, which is not included; that block reports `SKIPPED` without it.

## Prerequisites

- Python 3.11+ — the reported results were produced with Python 3.14
- PostgreSQL 15+ with `pgvector`, populated by the companion application
- `psycopg[binary]` (v3) — installed via `requirements.txt`

Environment variables:

| Variable | Required by | Purpose |
|---|---|---|
| `DATABASE_URL` | `audit/*`, `eval/scripts/compute_metrics.py`, the failure-injection scripts | `postgresql://user:password@host:5432/dbname` |
| `GOOGLE_CLOUD_PROJECT` | the application layer (`core/config.py`), and therefore the client-backed tests | Vertex AI project for Gemini and embeddings. Tests that need it skip when it is unset. |

```bash
pip install -r requirements.txt
```

Versions in `requirements.txt` are pinned to the environment that produced the reported results.

## Repository Structure

```
├── audit/                          # Metric computation (§IV-A, §IV-B)
│   ├── l1_metrics.py               #   Leaf Coverage@d, KB Reconstructability@t
│   ├── l2_metrics.py               #   L2 response-chunk link metrics
│   ├── l3_metrics.py               #   L3 feedback chain metrics
│   ├── cross_layer_metrics.py      #   L2→L1 and L3→L2 FK integrity
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
│   │   ├── eval_queries.json       #   30-query pilot corpus
│   │   ├── eval_queries_coverage.md#   Coverage report for the pilot corpus
│   │   └── eval_queries_v2.json    #   150-query corpus used in the paper
│   ├── scripts/                    #   Evaluation pipeline (see sections below)
│   ├── shared/                     #   Shared metric/stats utilities
│   └── results/
│       ├── eval_run_v2_redacted.json #  Redacted run log (150 sessions)
│       └── metrics/                #   Computed metrics (no raw data)
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

## Evaluation Corpus

**Paper reference:** §III-A Evaluation Environment (Table II)

The corpus used in the paper is `eval/generation/eval_queries_v2.json` — 150 queries, **shipped as data**. It has no generator script in this repository. It combines symptom queries derived from the manuals' troubleshooting tables through the public evaluation dataset (technician-style observations generated per fault entry, verified by multiple models, and adjudicated by hand) with queries authored by the team to exercise each routing path: deterministic-rule exits, follow-ups tied to a parent session, unsafe-method requests and unsafe follow-ups, part-replacement requests, and out-of-domain probes.

The per-variant counts in Table II are the **observed** trace variants of the executed run, not a designed stratification; 148 of 150 sessions matched their intended route.

`eval/scripts/select_eval_queries.py` is the **30-query pilot selector**. It produces `eval/generation/eval_queries.json`, not the 150-query corpus. Its selection input (`combined_eval_dataset.json`) is derived from the public evaluation dataset repository and is not redistributed here, so the selection itself cannot be re-run from a clean clone; the pilot corpus it produced is shipped instead. Run without `--force`, the script regenerates `eval_queries_coverage.md` from that shipped corpus, which reproduces the committed file byte for byte:

```bash
python -m eval.scripts.select_eval_queries
```

```bash
# Execute the corpus against the live companion application
python -m eval.scripts.run_eval_queries \
    --api-url http://localhost:8000 \
    --queries-file eval/generation/eval_queries_v2.json \
    --output eval/results/eval_run_v2.json
```

### Redacted run log

`eval/results/eval_run_v2_redacted.json` is the redistributable form of the run log: `response_id`, `thread_id` and `trace_id` are replaced by stable pseudonyms assigned in first-appearance order, so thread reuse by follow-up queries is preserved; every other field is verbatim. The run log holds no response text and no SSE payloads. Regenerate it from a raw run log with:

```bash
python -m eval.scripts.redact_run_log \
    --input eval/results/eval_run_v2.json \
    --output eval/results/eval_run_v2_redacted.json
```

Table II reproduces from the redacted file:

```bash
python -c "
import json, collections
d = json.load(open('eval/results/eval_run_v2_redacted.json'))
print(collections.Counter(r['actual_trace_variant'] for r in d).most_common())
print('total', len(d), '| all PASS:', all(r['status'] == 'PASS' for r in d))
print('expected == actual:', sum(1 for r in d if r['expected_trace_variant'] == r['actual_trace_variant']), '/', len(d))
"
```

which prints 52 / 34 / 21 / 19 / 10 / 10 / 4 across the seven variants, 150 sessions all `PASS`, and 148/150 expected-versus-actual agreement.

## Provenance Metrics

**Paper reference:** §IV-A Metrics, §IV-B Provenance Reconstruction (Table III)

All modules require `DATABASE_URL` at runtime (not at import time).

```bash
export DATABASE_URL=postgresql://user:password@host:5432/dbname

# L1: Leaf Coverage@d, KB Reconstructability@t, manifest integrity
#     (defaults to the two evaluation documents, ids 87 and 88)
python -m audit.l1_metrics

# Cross-layer: L2→L1 and L3→L2 FK integrity
python -m audit.cross_layer_metrics

# Process mining: token-replay fitness and precision
python -m audit.process_mining

# All metrics (provenance, feedback chain, failure injection,
# identity triple, overhead). Requires the raw run log, which is not
# redistributed here — see "Scope and Limitations".
python -m eval.scripts.compute_metrics
```

Provenance Completeness and Safety Provenance are corpus-level metrics computed by `compute_metrics.py`; `audit/cross_layer_metrics.py` provides the per-thread and FK-integrity computations they build on.

### Reported results (Table III)

Source: `eval/results/metrics/provenance_reconstruction.json`.

| Metric | Value |
|---|---|
| Provenance Completeness | 100% (11,004/11,004) |
| Safety Provenance | 100% (3,668/3,668) |
| KB-Recon@*t* (SC10000AMM) | 0.9602 |
| KB-Recon@*t* (AS-AMM-01-000) | 0.9878 |
| LC@*d* (SC10000AMM, *d*=2) | 0.8550 (1,114/1,303) |
| LC@*d* (AS-AMM-01-000, *d*=2) | 0.8813 (3,839/4,356) |
| Process mining fitness | 0.999 (262/264 traces conforming) |

Notes on how to read this table:

- **KB-Recon@*t* is a count-based upper bound**, not an exact reconstruction rate. The ingestion manifest records chunk *counts* rather than individual chunk identifiers, so the metric compares counts at time *t* against the current chunk set; both values carry `"value_type": "upper_bound"` in the metrics file.
- Document ids in the metrics file are `87` = AS-AMM-01-000 (Aerospool WT9 Dynamic) and `88` = SC10000AMM (CubCrafters CC11-100 Sport Cub), per `schemas/data/documents_multimodal.csv.gz`.
- Resolved links by variant: troubleshoot full 4,179, replace part and follow-up variants 4,857, unsafe redirect 1,968. Variants without retrieval-response linkage are validated through execution-level process conformance instead.

## Failure Injection Protocol

Adversarial integrity testing of the provenance architecture. 12 sessions (6 injected, 6 control), 4 failure types: chunk deletion, score perturbation, log gap, manifest tamper.

**Paper reference:** §IV-C Failure Injection: Adversarial Integrity Testing (Fig. 2, Fig. 3)

### Session selection

The 12 sessions were selected to cover all routes, under these eligibility thresholds: at least 3 retrieved chunks, at least 5 executed nodes, `PASS` status, both intent types represented, at least 4 safety-tagged sessions, and both retrieval-bearing variants represented. The constants are in `eval/scripts/select_injection_sessions.py`.

`select_injection_sessions.py` operates on the **30-query pilot corpus**, in which the trace variant is an integer; the 150-query corpus labels variants with strings. The script is documented and marked as pilot-only for that reason.

### Protocol

1. **Select sessions** from the pilot corpus:
   ```bash
   python -m eval.scripts.select_injection_sessions --run-log <pilot run log> \
       --queries eval/generation/eval_queries.json \
       --output eval/results/failure_injection/failure_injection_sessions.json
   ```

2. **Randomize failures** — assign types, generate SHA-256 sealed manifest:
   ```bash
   python -m eval.scripts.randomize_failures
   ```

3. **Apply injections** — each mutation applied and reverted individually:
   ```bash
   python -m eval.scripts.run_failure_injection
   ```

4. **Blinded diagnosis** — evaluator classifies sessions from exported artifacts only:
   ```bash
   python -m eval.scripts.run_diagnostics
   ```

5. **Unblind and score**:
   ```bash
   python -m eval.scripts.score_and_unblind
   ```

6. **Verify reverts**:
   ```bash
   python -m eval.scripts.verify_reverts
   ```

### Reported results

Source: `eval/results/metrics/failure_injection.json`.

- Blinded accuracy 0.83 (10/12); precision, recall and F1 all 0.83; 95% Clopper–Pearson CI [0.52, 0.98]
- 3/4 failure types detected with perfect recall: chunk deletion (2/2), score perturbation (1/1), log gap (2/2)
- Manifest tamper (0/1 correct localization): the tampered document was referenced by 8 of 12 sessions, all producing identical `MANIFEST INVALID` signals. The document-level hash chain creates a blast radius that prevents per-session attribution; per-chunk integrity hashing is proposed as the architectural fix

With n = 12 the interval is wide. The F1 should be read as evidence that provenance artifacts expose detectable signals for three of four failure types, not as a population-level detection rate.

### Integrity disclosure

The injection designer and the blinded evaluator were different members of the same research team. Blinding was procedural: the assignment was randomised with a non-reproducible seed and sealed by SHA-256 hash before diagnosis; the evaluator received only the exported provenance artifacts.

## Human Evaluation

**Paper reference:** §IV-D Layer 3 Feedback Chain Validation

Source: `eval/results/metrics/feedback_chain.json`.

One evaluator assessed all 106 response-generating sessions across four dimensions (correctness, completeness, safety, overall) and 2,771 individual repair steps. Step verdicts were accept (2,462), skip (154), incorrect (150) and modify (5), giving 155 corrections traceable through the provenance chain.

Feedback-chain resolution:

| Chain | Result |
|---|---|
| L2→L1 (response → chunk → manifest) | 5,461/5,461 valid |
| L3→L2 (step feedback → evaluation record) | 1,789/1,790 (99.94%), one out-of-range step index |

### Human–LLM concordance

Human–LLM concordance across 101 paired response-level assessments was **80.2% (81/101)**:

| | LLM positive | LLM negative |
|---|---|---|
| **Human positive** | 65 | 8 |
| **Human negative** | 12 | 16 |

Correctness-versus-faithfulness agreement was 92.9% (91/98), with 6 false-faithful cases where a high faithfulness score did not correspond to human-judged correctness.

Concordance is computed against **the badge shown to the evaluator at evaluation time** — the `badge` and `sync_faithfulness` columns of the evaluator sheet — not against the run log. The two disagree on 9 of the 101 compared responses, because the run log was rewritten by later reruns; scoring against it would compare the human verdict with an inline verdict the evaluator never saw. Badges map green → positive and yellow/red → negative; the 5 responses whose badge is `gray` (no inline verdict) have no LLM side and are excluded, giving 101 pairs from 106 assessed responses.

The evaluation harness presented each response with retrieved chunks, inline evaluation verdicts, and a pre-computed recommendation; the evaluator confirmed or overrode each recommendation. The concordance figure should therefore be read as an override-adjusted agreement rate, not as blind agreement. As a single-evaluator design, no inter-rater reliability is reported; this is documented as a limitation in the paper.

Recomputing the concordance requires `eval/results/human_eval/evaluator1_responses.csv`, which is not redistributed (it contains verbatim response text and production identifiers). `eval/scripts/human_eval_import.py` documents the full column schema of both evaluator sheets.

## Seed Data

`schemas/data/` contains structural metadata for the two evaluation documents and their chunks. Text content and embedding vectors have been stripped from `document_chunks_multimodal.csv.gz` for copyright reasons — the source documents are proprietary aircraft maintenance manuals. The retained columns (chunk ID, document ID, level, page range, section header, parent ID) are sufficient to verify provenance chain integrity and reproduce metric computation against a populated database.

To restore the seed data into a local PostgreSQL instance:

```bash
bash schemas/data/restore.sh          # default: container=hitl_postgres, db=hitldss
```

The script's `DATABASE_URL` default carries a `CHANGEME` placeholder password; export `DATABASE_URL` to match your own instance before running it.

To obtain the full chunk text, ingest the source documents using the included pipeline (`app.py`) with the publicly available [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval).

## Database Schema

**Paper reference:** §III-B Cross-Layer Provenance Design (Fig. 1)

`schemas/ddl_dump.sql` contains the DDL for 10 audit tables. Cross-layer FK chains:

- **L2→L1:** `response_chunk_link.chunk_id` → `document_chunks_multimodal.document_id` → `ingestion_manifest.document_id`
- **L3→L2:** `step_feedback.response_id` → `evaluation_metrics.response_id` → `response_chunk_link.response_id`

## Tests

```bash
pytest tests
```

Tests that construct a FastAPI client skip when `GOOGLE_CLOUD_PROJECT` is unset, since the application configuration cannot be imported without it. The metric and utility tests run without any environment configuration.

## Cite this work

```bibtex
@inproceedings{ling2026crosslayer,
  title     = {Cross-Layer Provenance for Agentic RAG: Preserving Identity
               Continuity Across Ingestion, Execution, and Feedback},
  author    = {Ling, Zhi An and Jong, Joanne Wan Chuin and
               Ngu, Alfi Diyana Ngu Binti Mohd Faris and Chang, Miko May Lee},
  booktitle = {2026 IEEE International Conference on Modern Computing and
               Artificial Intelligence Technologies (MCAIT)},
  year      = {2026}
  % doi: add once IEEE Xplore assigns one
}
```

Machine-readable metadata is in [`CITATION.cff`](CITATION.cff).

## License

MIT — see [LICENSE](LICENSE).
