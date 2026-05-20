# Evaluation Reproduction Guide

Evaluation scripts and metric computation code for:

> **Cross-Layer Provenance for Agentic RAG: Preserving Identity Continuity Across Ingestion, Execution, and Feedback**
> Zhi An Ling, Chi Yung Siew, Miko May Lee Chang

Validated on a 12-node LangGraph diagnostic decision-support system for aircraft corrective maintenance.

## Scope and Limitations

This repository provides **methodological transparency**, not push-button reproducibility. Every script that produced a reported number is included and inspectable. However, the scripts require:

- A PostgreSQL 15+ database with the audit schema (`schemas/ddl_dump.sql`) and session data. The two source documents are publicly available in the [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval) and can be ingested via the companion application
- The [evaluation dataset](https://github.com/fyp-group18/aircraft-maintenance-rag-eval) (pinned to commit [`27723ac`](https://github.com/fyp-group18/aircraft-maintenance-rag-eval/tree/27723ac8e9bef27ba61e8e2360c2b8a322a135ad))

Experiment 1 (`run_eval_queries.py`) requires a live instance of the companion application to generate the session corpus. All other scripts — metric computation (Experiment 2) and failure injection (Experiment 3) — operate directly against the database.

## Prerequisites

- Python 3.11+
- Access to a PostgreSQL 15+ database (with `pgvector`) populated by the companion application
- Environment variable: `DATABASE_URL=postgresql://user:password@host:5432/dbname`

```bash
pip install -r requirements.txt
```

## Database Pipeline

The `core/` directory contains the minimal database connection layer extracted from the companion application. It provides:

- `SessionLocal()` — returns a new SQLAlchemy session; the engine is created lazily on first call
- `with_db_retry` — retry decorator for transient DB errors (Neon cold starts)

`DATABASE_URL` is read from the environment at first use, not at import time — modules can be imported, linted, and tested without a live database. The engine requires the `psycopg` (v3) driver.

This is the only component extracted from the companion project. No application logic is included.

## Repository Structure

```
├── core/                           # Database pipeline (extracted from companion project)
│   ├── __init__.py
│   └── database.py                 #   SessionLocal(), with_db_retry, lazy engine
│
├── audit/                          # Metric computation (§Evaluation)
│   ├── l1_metrics.py               #   Leaf Coverage@d, KB Reconstructability@t
│   ├── cross_layer_metrics.py      #   Provenance Completeness, Safety Provenance
│   ├── manifest.py                 #   SHA-256 hash-chain append/verify
│   └── process_mining.py           #   Token-replay fitness
│
├── eval/
│   ├── queries/
│   │   └── eval_queries.json       # 30 evaluation queries
│   └── scripts/
│       ├── select_eval_queries.py   # Deterministic query selection (SEED=42)
│       ├── run_eval_queries.py      # Submit queries via SSE
│       ├── select_injection_sessions.py
│       ├── randomize_failures.py    # Failure assignment + SHA-256 manifest
│       ├── apply_injection.py       # Apply/revert a single DB mutation
│       ├── run_failure_injection.py # Orchestrate all injections
│       ├── run_diagnostics.py       # Blinded diagnostic observation collection
│       ├── score_and_unblind.py     # Unblind + confusion matrix + F1
│       └── verify_reverts.py        # Post-experiment revert verification
│
└── schemas/
    └── ddl_dump.sql                # Full PostgreSQL DDL for 10 audit tables
```

## Experiment 1: Evaluation Query Execution

Produces the 30-session corpus. All metrics are computed from this corpus.

**Paper reference:** §III-A Evaluation Environment

```bash
# Query selection (deterministic, SEED=42)
python eval/scripts/select_eval_queries.py

# Execute against the live application (only step that requires the running system)
python eval/scripts/run_eval_queries.py --email <user> --password <password>
```

## Experiment 2: Metric Computation

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

## Experiment 3: Failure Injection Protocol

Adversarial integrity testing of the provenance architecture. 12 sessions (6 injected, 6 control), 4 failure types: chunk deletion, score perturbation, log gap, manifest tamper.

**Paper reference:** §IV-C Failure Injection

### Protocol

1. **Select sessions** from the 30-query corpus:
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

4. **Blinded diagnosis** — second researcher classifies sessions from exported artifacts only:
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
