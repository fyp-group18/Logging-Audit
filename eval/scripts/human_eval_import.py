#!/usr/bin/env python3
"""Human evaluation import — import evaluator verdicts into L3 tables and validate.

Run AFTER the evaluator has completed their assessment.

Reads (none of these sheets are redistributed in this repository — they carry
verbatim response text from copyrighted manuals and production identifiers):

  eval/results/human_eval/evaluator{N}_responses.csv — one row per response:
      query_id            corpus query identifier, e.g. "EQ-011"
      response_id         L2 response UUID the verdict attaches to
      variant             observed trace variant label
      device_id           aircraft model the query was posed against
      query_text          the query as executed
      badge               inline evaluator badge (green | yellow | red | gray)
      sync_faithfulness   inline faithfulness score in [0, 1], may be empty
      chunks_retrieved    number of chunks retrieved for the response
      safety_expected     whether the query was expected to surface safety content
      notes               free text
      rec_correctness     pre-computed recommendation shown to the evaluator
      rec_completeness    pre-computed recommendation shown to the evaluator
      rec_safety          pre-computed recommendation shown to the evaluator
      rec_overall         pre-computed recommendation shown to the evaluator
      rec_reasoning       why the recommendation was made
      correctness         EVALUATOR VERDICT: correct | partial | incorrect
      completeness        EVALUATOR VERDICT: complete | partial | incomplete
      safety_assessment   EVALUATOR VERDICT: present | partial | missing | n_a
      overall_verdict     EVALUATOR VERDICT: THUMBS_UP | THUMBS_DOWN
      correction_text     free-text correction, for negative verdicts

  eval/results/human_eval/evaluator{N}_steps.csv — one row per repair step:
      query_id            corpus query identifier
      response_id         L2 response UUID
      step_index          0-based index of the step within the repair plan
      step_content        the step text as generated
      rec_verdict         pre-computed recommendation shown to the evaluator
      rec_reasoning       why the recommendation was made
      inline_faithful     inline evaluator faithfulness signal for the step
      inline_grounding    inline grounding label (verbatim | paraphrased | ungrounded)
      verdict             EVALUATOR VERDICT: ACCEPT | SKIP | MODIFY | INCORRECT
      correction_text     free-text correction, required for MODIFY and INCORRECT
      severity            severity tag for the correction

  eval/results/human_eval/evaluation_data.json — full response context, used
      only for thread_id / device_id lookup.

Writes to DB:
  step_feedback   — per-step verdicts
  response_feedback — response-level verdicts

Requires:
  DATABASE_URL env var
  Evaluator user UUIDs in the users table (or creates them)
"""

import csv
import json
import os
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psycopg

ROOT = Path(__file__).resolve().parents[2]
HUMAN_EVAL_DIR = ROOT / "eval" / "results" / "human_eval"

# Evaluator UUIDs — deterministic so re-runs are idempotent
EVALUATOR_UUIDS = {
    1: uuid.UUID("00000000-0000-4000-a001-000000000001"),
    2: uuid.UUID("00000000-0000-4000-a002-000000000002"),
}
EVALUATOR_NAMES = {1: "eval_annotator_1", 2: "eval_annotator_2"}
EVAL_DEVICE_ID = "eval-workstation"


def cohens_kappa(labels1: list[str], labels2: list[str]) -> float:
    """Compute Cohen's kappa without sklearn.

    Uses the standard formula: κ = (p_o - p_e) / (1 - p_e)
    where p_o = observed agreement, p_e = expected agreement by chance.
    """
    assert len(labels1) == len(labels2), "Label lists must be same length"
    n = len(labels1)
    if n == 0:
        return 0.0

    categories = sorted(set(labels1) | set(labels2))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Build confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(labels1, labels2):
        matrix[cat_idx[a], cat_idx[b]] += 1

    p_o = np.trace(matrix) / n  # observed agreement

    # Expected agreement: sum of (row_i_total * col_i_total) / n^2
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    p_e = (row_sums * col_sums).sum() / (n * n)

    if p_e == 1.0:
        return 1.0  # perfect agreement trivially
    return float((p_o - p_e) / (1 - p_e))


def ensure_evaluator_users(conn):
    """Create synthetic evaluator users if they don't exist."""
    cur = conn.cursor()
    for eval_num in [1, 2]:
        uid = EVALUATOR_UUIDS[eval_num]
        name = EVALUATOR_NAMES[eval_num]
        cur.execute("SELECT id FROM users WHERE id = %s", (str(uid),))
        if cur.fetchone() is None:
            cur.execute(
                """
                INSERT INTO users (id, username, role, created_at)
                VALUES (%s, %s, 'evaluator', NOW())
                ON CONFLICT (id) DO NOTHING
                """,
                (str(uid), name),
            )
            print(f"  Created evaluator user: {name} ({uid})")
        else:
            print(f"  Evaluator user exists: {name} ({uid})")
    conn.commit()


def load_evaluation_context() -> dict[str, dict]:
    """Load evaluation_data.json keyed by response_id for thread_id/device_id lookup."""
    path = HUMAN_EVAL_DIR / "evaluation_data.json"
    with open(path) as f:
        data = json.load(f)
    return {d["response_id"]: d for d in data}


def load_step_verdicts(filepath: Path) -> dict[tuple[str, int], dict]:
    """Load step-level verdicts keyed by (response_id, step_index)."""
    verdicts = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = (row.get("verdict") or "").strip().upper()
            if not verdict:
                continue  # skip unevaluated rows
            key = (row["response_id"], int(row["step_index"]))
            verdicts[key] = {
                "action": verdict,  # ACCEPT / SKIP / MODIFY / INCORRECT
                "correction_text": row.get("correction_text", "").strip() or None,
                "severity": row.get("severity", "").strip() or None,
            }
    return verdicts


def load_response_verdicts(filepath: Path) -> dict[str, dict]:
    """Load response-level verdicts keyed by response_id."""
    verdicts = {}
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            overall = (row.get("overall_verdict") or "").strip().upper()
            if not overall:
                continue
            verdicts[row["response_id"]] = {
                "feedback_type": overall,  # THUMBS_UP / THUMBS_DOWN
                "correctness": row.get("correctness", "").strip(),
                "completeness": row.get("completeness", "").strip(),
                "safety_assessment": row.get("safety_assessment", "").strip(),
                "correction_text": row.get("correction_text", "").strip() or None,
            }
    return verdicts


def import_step_feedback(conn, context: dict[str, dict]):
    """Import step verdicts from both evaluators into step_feedback."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0
    corrections = 0

    for eval_num in [1, 2]:
        filepath = HUMAN_EVAL_DIR / f"evaluator{eval_num}_steps.csv"
        if not filepath.exists():
            print(f"  WARNING: {filepath.name} not found, skipping evaluator {eval_num}")
            continue

        verdicts = load_step_verdicts(filepath)
        user_id = str(EVALUATOR_UUIDS[eval_num])
        print(f"  Evaluator {eval_num}: {len(verdicts)} step verdicts loaded")

        for (response_id, step_index), v in verdicts.items():
            ctx = context.get(response_id)
            if not ctx:
                print(f"    SKIP: response_id={response_id} not in evaluation context")
                skipped += 1
                continue

            thread_id = ctx["thread_id"]
            device_id = ctx.get("device_id", EVAL_DEVICE_ID)

            cur.execute(
                """
                INSERT INTO step_feedback (
                    response_id, step_index, thread_id, user_id,
                    device_id, equipment_category,
                    action, correction_text, severity, weight, created_at
                ) VALUES (%s, %s, %s, %s, %s, 'GENERAL', %s, %s, %s, 1.0, NOW())
                ON CONFLICT (response_id, step_index, user_id) DO UPDATE SET
                    action = EXCLUDED.action,
                    correction_text = EXCLUDED.correction_text,
                    severity = EXCLUDED.severity
                """,
                (
                    response_id,
                    step_index,
                    thread_id,
                    user_id,
                    device_id,
                    v["action"],
                    v["correction_text"],
                    v["severity"],
                ),
            )
            inserted += 1
            if v["action"] in ("MODIFY", "INCORRECT"):
                corrections += 1

    conn.commit()
    return inserted, corrections, skipped


def import_response_feedback(conn, context: dict[str, dict]):
    """Import response-level verdicts from both evaluators into response_feedback."""
    cur = conn.cursor()
    inserted = 0
    skipped = 0

    for eval_num in [1, 2]:
        filepath = HUMAN_EVAL_DIR / f"evaluator{eval_num}_responses.csv"
        if not filepath.exists():
            print(f"  WARNING: {filepath.name} not found, skipping evaluator {eval_num}")
            continue

        verdicts = load_response_verdicts(filepath)
        user_id = str(EVALUATOR_UUIDS[eval_num])
        print(f"  Evaluator {eval_num}: {len(verdicts)} response verdicts loaded")

        for response_id, v in verdicts.items():
            ctx = context.get(response_id)
            if not ctx:
                print(f"    SKIP: response_id={response_id} not in evaluation context")
                skipped += 1
                continue

            thread_id = ctx["thread_id"]
            device_id = ctx.get("device_id", EVAL_DEVICE_ID)

            # Build correction text with structured assessment
            parts = []
            if v["correctness"]:
                parts.append(f"correctness={v['correctness']}")
            if v["completeness"]:
                parts.append(f"completeness={v['completeness']}")
            if v["safety_assessment"]:
                parts.append(f"safety={v['safety_assessment']}")
            if v["correction_text"]:
                parts.append(v["correction_text"])
            correction = "; ".join(parts) if parts else None

            cur.execute(
                """
                INSERT INTO response_feedback (
                    response_id, thread_id, user_id,
                    device_id, equipment_category,
                    feedback_type, correction_text, weight, created_at
                ) VALUES (%s, %s, %s, %s, 'GENERAL', %s, %s, 1.0, NOW())
                ON CONFLICT (response_id, user_id) DO UPDATE SET
                    feedback_type = EXCLUDED.feedback_type,
                    correction_text = EXCLUDED.correction_text
                """,
                (
                    response_id,
                    thread_id,
                    user_id,
                    device_id,
                    v["feedback_type"],
                    correction,
                ),
            )
            inserted += 1

    conn.commit()
    return inserted, skipped


def validate_fk_chains(conn):
    """Validate L3→L2→L1 FK chains using existing audit functions."""
    print("\n=== FK CHAIN VALIDATION ===")

    # Use existing audit functions
    sys.path.insert(0, str(ROOT))
    from audit.cross_layer_metrics import (
        validate_l2_to_l1_fk_integrity,
        validate_l3_to_l2_fk_integrity,
    )

    l3_l2 = validate_l3_to_l2_fk_integrity()
    print(f"  L3→L2 (step_feedback → evaluation_metrics):")
    print(f"    Integrity: {l3_l2['value']}")
    if l3_l2.get("metadata"):
        for k, v in l3_l2["metadata"].items():
            print(f"    {k}: {v}")

    l2_l1 = validate_l2_to_l1_fk_integrity()
    print(f"  L2→L1 (response_chunk_link → document_chunks_multimodal):")
    print(f"    Integrity: {l2_l1['value']}")
    if l2_l1.get("metadata"):
        for k, v in l2_l1["metadata"].items():
            print(f"    {k}: {v}")

    # Sanity count: step_feedback records with matching evaluation_metrics
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            COUNT(*) AS total_sf,
            COUNT(em.response_id) AS has_eval_metric,
            COUNT(DISTINCT sf.response_id) AS distinct_responses
        FROM step_feedback sf
        LEFT JOIN evaluation_metrics em ON sf.response_id = em.response_id
        WHERE sf.retracted = false
        """
    )
    row = cur.fetchone()
    print(f"\n  Sanity counts:")
    print(f"    Total step_feedback (non-retracted): {row[0]}")
    print(f"    With evaluation_metrics match: {row[1]}")
    print(f"    Distinct responses: {row[2]}")

    return l3_l2, l2_l1


def compute_agreement(context: dict[str, dict]):
    """Compute inter-annotator agreement (Cohen's κ) at response and step levels."""
    print("\n=== INTER-ANNOTATOR AGREEMENT ===")

    # Response-level
    resp1_path = HUMAN_EVAL_DIR / "evaluator1_responses.csv"
    resp2_path = HUMAN_EVAL_DIR / "evaluator2_responses.csv"

    if not resp1_path.exists() or not resp2_path.exists():
        print("  SKIP: evaluator response files not found")
        return None, None

    rv1 = load_response_verdicts(resp1_path)
    rv2 = load_response_verdicts(resp2_path)

    common_responses = sorted(set(rv1.keys()) & set(rv2.keys()))
    if len(common_responses) < 2:
        print(f"  SKIP: only {len(common_responses)} common responses (need ≥2)")
        return None, None

    # Correctness agreement
    labels1 = [rv1[rid]["correctness"] for rid in common_responses if rv1[rid]["correctness"] and rv2[rid]["correctness"]]
    labels2 = [rv2[rid]["correctness"] for rid in common_responses if rv1[rid]["correctness"] and rv2[rid]["correctness"]]

    if len(labels1) >= 2:
        kappa_resp = cohens_kappa(labels1, labels2)
        print(f"  Response-level correctness (n={len(labels1)}):")
        print(f"    Cohen's κ = {kappa_resp:.4f}")
        print(f"    Evaluator 1: {dict(Counter(labels1))}")
        print(f"    Evaluator 2: {dict(Counter(labels2))}")
        _interpret_kappa(kappa_resp)
    else:
        kappa_resp = None
        print(f"  Response-level: insufficient paired correctness labels")

    # Step-level verdict agreement
    sv1 = load_step_verdicts(HUMAN_EVAL_DIR / "evaluator1_steps.csv")
    sv2 = load_step_verdicts(HUMAN_EVAL_DIR / "evaluator2_steps.csv")

    common_steps = sorted(set(sv1.keys()) & set(sv2.keys()))
    step_labels1 = [sv1[k]["action"] for k in common_steps]
    step_labels2 = [sv2[k]["action"] for k in common_steps]

    if len(step_labels1) >= 2:
        kappa_step = cohens_kappa(step_labels1, step_labels2)
        print(f"\n  Step-level verdict (n={len(step_labels1)}):")
        print(f"    Cohen's κ = {kappa_step:.4f}")
        print(f"    Evaluator 1: {dict(Counter(step_labels1))}")
        print(f"    Evaluator 2: {dict(Counter(step_labels2))}")
        _interpret_kappa(kappa_step)
    else:
        kappa_step = None
        print(f"  Step-level: insufficient paired verdicts")

    return kappa_resp, kappa_step


def _interpret_kappa(kappa: float):
    if kappa > 0.8:
        print(f"    → Almost perfect agreement")
    elif kappa > 0.6:
        print(f"    → Substantial agreement")
    elif kappa > 0.4:
        print(f"    → Moderate agreement")
    elif kappa > 0.2:
        print(f"    → Fair agreement")
    else:
        print(f"    → Poor agreement — investigate disagreements")


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    # Check evaluator files exist
    required = [
        HUMAN_EVAL_DIR / "evaluator1_steps.csv",
        HUMAN_EVAL_DIR / "evaluator2_steps.csv",
        HUMAN_EVAL_DIR / "evaluation_data.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        print("ERROR: Missing required files:", file=sys.stderr)
        for p in missing:
            print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
        sys.exit(1)

    print("Loading evaluation context...")
    context = load_evaluation_context()
    print(f"  {len(context)} responses in context")

    conn = psycopg.connect(db_url)
    try:
        # Ensure evaluator users exist
        print("\nEnsuring evaluator users...")
        ensure_evaluator_users(conn)

        # Import step-level feedback
        print("\nImporting step-level feedback...")
        step_inserted, step_corrections, step_skipped = import_step_feedback(conn, context)
        print(f"  Inserted: {step_inserted}")
        print(f"  Corrections (MODIFY/INCORRECT): {step_corrections}")
        print(f"  Skipped (no context): {step_skipped}")

        # Import response-level feedback
        print("\nImporting response-level feedback...")
        resp_inserted, resp_skipped = import_response_feedback(conn, context)
        print(f"  Inserted: {resp_inserted}")
        print(f"  Skipped: {resp_skipped}")

        # Validate FK chains
        l3_l2, l2_l1 = validate_fk_chains(conn)

        # Compute inter-annotator agreement
        kappa_resp, kappa_step = compute_agreement(context)

    finally:
        conn.close()

    # Summary
    print(f"\n{'='*50}")
    print(f"  HUMAN EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Responses in context: {len(context)}")
    print(f"  step_feedback records: {step_inserted}")
    print(f"  response_feedback records: {resp_inserted}")
    print(f"  Step corrections: {step_corrections}")
    print(f"  L3→L2 integrity: {l3_l2['value'] if l3_l2 else 'N/A'}")
    print(f"  L2→L1 integrity: {l2_l1['value'] if l2_l1 else 'N/A'}")
    if kappa_resp is not None:
        print(f"  Cohen's κ (response correctness): {kappa_resp:.4f}")
    if kappa_step is not None:
        print(f"  Cohen's κ (step verdict): {kappa_step:.4f}")
    print()
    resp_with_corrections = set()
    for eval_num in [1, 2]:
        filepath = HUMAN_EVAL_DIR / f"evaluator{eval_num}_steps.csv"
        if filepath.exists():
            sv = load_step_verdicts(filepath)
            for (rid, _), v in sv.items():
                if v["action"] in ("MODIFY", "INCORRECT"):
                    resp_with_corrections.add(rid)
    print(f"  Responses with ≥1 correction: {len(resp_with_corrections)}")


if __name__ == "__main__":
    main()
