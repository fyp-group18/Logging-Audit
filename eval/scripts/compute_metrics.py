#!/usr/bin/env python3
"""Metrics computation — provenance, feedback chain, failure injection,
identity triple, and overhead from the 150-query eval corpus + human evaluation.

Usage:
    cd backend && source .venv/bin/activate
    python eval/scripts/compute_metrics.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

DB_URL = os.environ["DATABASE_URL"]

BASE = Path(__file__).resolve().parent.parent
RESULTS_PATH = BASE / "results" / "eval_run_v2.json"
HUMAN_EVAL_DIR = BASE / "results" / "human_eval"
FI_DIR = BASE / "results" / "failure_injection" / "experiment_run_001"
OUT_DIR = BASE / "results" / "metrics"


def load_results() -> list[dict]:
    with open(RESULTS_PATH) as f:
        return json.load(f)


def save_json(data: dict, filename: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════
# PROVENANCE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════

def provenance_reconstruction(conn, results: list[dict]) -> dict:
    print("=" * 60)
    print("PROVENANCE RECONSTRUCTION")
    print("=" * 60)

    cur = conn.cursor()

    eval_response_ids = [
        r["response_id"] for r in results
        if r.get("response_id") and r["response_id"] != "timeout"
    ]
    response_variant = {}
    for r in results:
        rid = r.get("response_id")
        if rid and rid != "timeout":
            response_variant[rid] = (
                r.get("actual_trace_variant") or r.get("expected_trace_variant", "unknown")
            )

    # ── 1A. Provenance Completeness (overall) ─────────────────
    # Use EXISTS for manifest check to avoid row multiplication
    cur.execute("""
        SELECT
            rcl.response_id,
            rcl.chunk_id,
            dcm.id AS chunk_exists,
            dcm.document_id,
            EXISTS (
                SELECT 1 FROM ingestion_manifest im
                WHERE im.document_id = dcm.document_id
            ) AS manifest_exists
        FROM response_chunk_link rcl
        LEFT JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id
        WHERE rcl.response_id = ANY(%s)
    """, (eval_response_ids,))

    rows = cur.fetchall()
    total_links = len(rows)
    resolved = sum(1 for r in rows if r[2] is not None and r[4] is True)
    broken_chunk = sum(1 for r in rows if r[2] is None)
    broken_manifest = sum(1 for r in rows if r[2] is not None and r[4] is not True)

    print(f"\n1A. Provenance Completeness (eval corpus)")
    print(f"  Total response-chunk links: {total_links}")
    print(f"  Resolved to manifest: {resolved}")
    print(f"  Broken (chunk missing): {broken_chunk}")
    print(f"  Broken (manifest missing): {broken_manifest}")
    pct = 100 * resolved / total_links if total_links else 0
    print(f"  Completeness: {resolved}/{total_links} = {pct:.4f}%")

    # ── 1B. Per-Variant Completeness ──────────────────────────
    print(f"\n1B. Per-Variant Provenance Completeness")

    variant_links: dict[str, dict] = defaultdict(lambda: {"total": 0, "resolved": 0})
    for row in rows:
        variant = response_variant.get(row[0], "unknown")
        variant_links[variant]["total"] += 1
        if row[2] is not None and row[4] is True:
            variant_links[variant]["resolved"] += 1

    print(f"  {'Variant':<35} {'Resolved':>10} {'Total':>8} {'%':>8}")
    print(f"  {'-'*35} {'-'*10} {'-'*8} {'-'*8}")
    for variant in sorted(variant_links.keys()):
        v = variant_links[variant]
        vpct = 100 * v["resolved"] / v["total"] if v["total"] > 0 else 0
        print(f"  {variant:<35} {v['resolved']:>10} {v['total']:>8} {vpct:>7.2f}%")

    # ── 1C. Safety Provenance ─────────────────────────────────
    print(f"\n1C. Safety Provenance")

    safety_response_ids = [
        r["response_id"] for r in results
        if r.get("safety_found") and r.get("response_id") and r["response_id"] != "timeout"
    ]

    safety_total = 0
    safety_resolved = 0
    if safety_response_ids:
        cur.execute("""
            SELECT
                rcl.response_id,
                COUNT(*) AS total,
                SUM(CASE WHEN EXISTS (
                    SELECT 1 FROM ingestion_manifest im
                    WHERE im.document_id = dcm.document_id
                ) THEN 1 ELSE 0 END) AS resolved
            FROM response_chunk_link rcl
            LEFT JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id
            WHERE rcl.response_id = ANY(%s)
            GROUP BY rcl.response_id
        """, (safety_response_ids,))

        for row in cur.fetchall():
            safety_total += row[1]
            safety_resolved += row[2]

        spct = 100 * safety_resolved / safety_total if safety_total else 0
        print(f"  Safety-flagged responses: {len(safety_response_ids)}")
        print(f"  Safety chunk links: {safety_total}")
        print(f"  Resolved to manifest: {safety_resolved}")
        print(f"  Safety Provenance: {safety_resolved}/{safety_total} = {spct:.4f}%")
    else:
        print(f"  No safety-flagged responses found")

    # ── 1D. KB-Recon@t ────────────────────────────────────────
    print(f"\n1D. KB Reconstructability @ t")

    # Get distinct document_ids referenced by eval corpus
    cur.execute("""
        SELECT DISTINCT dcm.document_id
        FROM response_chunk_link rcl
        JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id
        WHERE rcl.response_id = ANY(%s)
          AND dcm.document_id IS NOT NULL
    """, (eval_response_ids,))
    doc_ids = [row[0] for row in cur.fetchall()]
    print(f"  Documents referenced in eval corpus: {doc_ids}")

    sys.path.insert(0, str(BASE.parent))
    from audit.l1_metrics import compute_kb_reconstructability

    kb_results = {}
    for doc_id in doc_ids:
        try:
            # Use far-future timestamp to include all manifest entries
            # (backfilled data has all entries within the same second)
            kb = compute_kb_reconstructability(doc_id, as_of_timestamp="2099-01-01T00:00:00Z")
            kb_results[doc_id] = kb
            print(f"  doc={doc_id}: KB-Recon@t = {kb['value']}"
                  f" ({kb['metadata'].get('approximation', kb['metadata'].get('note', ''))})")
        except Exception as e:
            print(f"  doc={doc_id}: ERROR — {e}")
            kb_results[doc_id] = {"value": None, "error": str(e)}

    # ── 1E. Leaf Coverage@d ───────────────────────────────────
    print(f"\n1E. Leaf Coverage @ d")

    from audit.l1_metrics import compute_leaf_coverage

    lc_results = {}
    for doc_id in doc_ids:
        try:
            lc = compute_leaf_coverage(doc_id)
            lc_results[doc_id] = lc
            meta = lc.get("metadata", {})
            print(f"  doc={doc_id}: LC@d = {lc['value']}"
                  f" (leaves={meta.get('leaf_count')}, "
                  f"reachable={meta.get('reachable_leaves')}, "
                  f"depth={meta.get('max_depth')})")
        except Exception as e:
            print(f"  doc={doc_id}: ERROR — {e}")
            lc_results[doc_id] = {"value": None, "error": str(e)}

    # ── 1F. Process Mining Fitness ────────────────────────────
    print(f"\n1F. Process Mining Fitness")

    from audit.process_mining import compute_process_fitness

    try:
        fitness = compute_process_fitness()
        print(f"  Fitness: {fitness['value']}")
        meta = fitness.get("metadata", {})
        print(f"  Traces: {meta.get('total_traces')}, "
              f"Conforming: {meta.get('conforming_traces')}, "
              f"Deviating: {meta.get('deviating_traces')}")
    except Exception as e:
        print(f"  ERROR: {e}")
        fitness = {"value": None, "error": str(e)}

    # ── Save ──────────────────────────────────────────────────
    exp1 = {
        "provenance_completeness": {
            "resolved": resolved, "total": total_links,
            "broken_chunk": broken_chunk, "broken_manifest": broken_manifest,
            "percentage": resolved / total_links if total_links else None,
        },
        "per_variant": {v: dict(d) for v, d in variant_links.items()},
        "safety_provenance": {
            "resolved": safety_resolved, "total": safety_total,
            "responses": len(safety_response_ids),
            "percentage": safety_resolved / safety_total if safety_total else None,
        },
        "kb_reconstructability": {str(k): v for k, v in kb_results.items()},
        "leaf_coverage": {str(k): v for k, v in lc_results.items()},
        "process_mining_fitness": fitness,
    }
    save_json(exp1, "provenance_reconstruction.json")
    return exp1


# ═══════════════════════════════════════════════════════════════
# FEEDBACK CHAIN VALIDATION
# ═══════════════════════════════════════════════════════════════

def feedback_chain_validation(conn, results: list[dict]) -> dict:
    print("\n" + "=" * 60)
    print("FEEDBACK CHAIN VALIDATION")
    print("=" * 60)

    cur = conn.cursor()

    # ── Feedback counts ───────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM step_feedback WHERE retracted = FALSE")
    total_feedback = cur.fetchone()[0]
    print(f"\nTotal step_feedback records (non-retracted): {total_feedback}")

    cur.execute(
        "SELECT COUNT(DISTINCT response_id) FROM step_feedback WHERE retracted = FALSE"
    )
    responses_with_feedback = cur.fetchone()[0]
    print(f"Responses with feedback: {responses_with_feedback}")

    cur.execute("""
        SELECT action, COUNT(*)
        FROM step_feedback
        WHERE retracted = FALSE
        GROUP BY action
    """)
    action_dist = dict(cur.fetchall())
    print(f"Action distribution: {action_dist}")

    corrections = sum(
        v for k, v in action_dist.items() if k in ("MODIFY", "INCORRECT")
    )
    print(f"Corrections (MODIFY + INCORRECT): {corrections}")

    # ── L3→L2→L1 chain validation via audit functions ─────────
    print(f"\nFK Integrity (via audit functions):")

    from audit.cross_layer_metrics import (
        validate_l2_to_l1_fk_integrity,
        validate_l3_to_l2_fk_integrity,
    )

    l2_to_l1 = validate_l2_to_l1_fk_integrity()
    print(f"  L2→L1 (response_chunk_link → chunks): {l2_to_l1['value']}")
    print(f"    {l2_to_l1['metadata']}")

    l3_to_l2 = validate_l3_to_l2_fk_integrity()
    print(f"  L3→L2 (step_feedback → evaluation_metrics): {l3_to_l2['value']}")
    print(f"    {l3_to_l2['metadata']}")

    # ── Full chain: feedback → chunks → documents → manifest ─
    print(f"\nL3→L2→L1 Full Chain Resolution:")

    cur.execute("""
        SELECT
            sf.id AS feedback_id,
            sf.response_id,
            sf.step_index,
            sf.action,
            em.response_id AS em_link,
            BOOL_OR(rcl.response_id IS NOT NULL) AS has_chunk_link,
            BOOL_OR(dcm.document_id IS NOT NULL) AS has_doc,
            BOOL_OR(EXISTS (
                SELECT 1 FROM ingestion_manifest im
                WHERE im.document_id = dcm.document_id
            )) AS has_manifest
        FROM step_feedback sf
        LEFT JOIN evaluation_metrics em ON sf.response_id = em.response_id
        LEFT JOIN response_chunk_link rcl ON sf.response_id = rcl.response_id
        LEFT JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id
        WHERE sf.retracted = FALSE
        GROUP BY sf.id, sf.response_id, sf.step_index, sf.action, em.response_id
    """)

    chain_rows = cur.fetchall()
    chain_total = len(chain_rows)
    chain_full = sum(1 for r in chain_rows if r[4] and r[5] and r[6] and r[7])
    chain_no_em = sum(1 for r in chain_rows if r[4] is None)
    chain_no_rcl = sum(1 for r in chain_rows if r[4] and not r[5])
    chain_no_doc = sum(1 for r in chain_rows if r[5] and not r[6])
    chain_no_manifest = sum(1 for r in chain_rows if r[6] and not r[7])

    print(f"  Feedback records: {chain_total}")
    print(f"  Full chain (feedback→eval→chunk→doc→manifest): {chain_full}")
    print(f"  Broken at feedback→eval_metrics: {chain_no_em}")
    print(f"  Broken at eval→response_chunk_link: {chain_no_rcl}")
    print(f"  Broken at chunk→document: {chain_no_doc}")
    print(f"  Broken at document→manifest: {chain_no_manifest}")

    # ── Documents traced from corrections ─────────────────────
    cur.execute("""
        SELECT DISTINCT dcm.document_id
        FROM step_feedback sf
        JOIN response_chunk_link rcl ON sf.response_id = rcl.response_id
        JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id
        WHERE sf.retracted = FALSE
          AND sf.action IN ('MODIFY', 'INCORRECT')
    """)
    traced_docs = [row[0] for row in cur.fetchall()]
    print(f"\n  Documents traced from corrections: {len(traced_docs)} — {traced_docs}")

    # ── Human-vs-LLM Concordance ──────────────────────────────
    concordance = _compute_human_vs_llm_concordance(results)

    # ── Save ──────────────────────────────────────────────────
    exp2 = {
        "total_feedback": total_feedback,
        "responses_with_feedback": responses_with_feedback,
        "action_distribution": action_dist,
        "corrections": corrections,
        "l2_to_l1_fk_integrity": l2_to_l1,
        "l3_to_l2_fk_integrity": l3_to_l2,
        "full_chain": {
            "total": chain_total,
            "resolved": chain_full,
            "broken_eval": chain_no_em,
            "broken_rcl": chain_no_rcl,
            "broken_doc": chain_no_doc,
            "broken_manifest": chain_no_manifest,
        },
        "documents_traced_from_corrections": traced_docs,
        "human_vs_llm_concordance": concordance,
    }
    save_json(exp2, "feedback_chain.json")
    return exp2


def _compute_human_vs_llm_concordance(results: list[dict]) -> dict | None:
    """Compare Evaluator 1 human verdicts against InlineEvaluator badge/scores."""
    print(f"\nHuman-vs-LLM Concordance (Evaluator 1 vs InlineEvaluator):")

    e1_path = HUMAN_EVAL_DIR / "evaluator1_responses.csv"
    if not e1_path.exists():
        print(f"  SKIPPED — {e1_path} not found")
        return None

    with open(e1_path) as f:
        e1_rows = {r["query_id"]: r for r in csv.DictReader(f)}

    results_map = {r["query_id"]: r for r in results}

    # Map human overall_verdict to binary
    # Map LLM badge to binary: green → THUMBS_UP, yellow/red → THUMBS_DOWN
    badge_to_binary = {"green": "THUMBS_UP", "yellow": "THUMBS_DOWN", "red": "THUMBS_DOWN"}

    pairs = []
    for qid, e1 in e1_rows.items():
        human_verdict = e1.get("overall_verdict")
        r = results_map.get(qid)
        if not r or not human_verdict:
            continue
        badge = r.get("badge")
        if badge not in badge_to_binary:
            continue
        llm_verdict = badge_to_binary[badge]
        pairs.append({
            "query_id": qid,
            "human": human_verdict,
            "llm_badge": badge,
            "llm_binary": llm_verdict,
            "agree": human_verdict == llm_verdict,
            "faithfulness": r.get("sync_faithfulness"),
            "relevance": r.get("sync_answer_relevance"),
        })

    if not pairs:
        print(f"  No comparable pairs found")
        return None

    n = len(pairs)
    agree = sum(1 for p in pairs if p["agree"])
    disagree = n - agree

    print(f"  Comparable pairs: {n}")
    print(f"  Agreement: {agree}/{n} = {100 * agree / n:.1f}%")
    print(f"  Disagreement: {disagree}/{n}")

    # Confusion matrix: rows = human, cols = LLM
    cm = defaultdict(int)
    for p in pairs:
        cm[(p["human"], p["llm_binary"])] += 1

    print(f"\n  Confusion matrix (Human × LLM):")
    print(f"  {'':>20} {'LLM_UP':>10} {'LLM_DOWN':>10}")
    for hv in ["THUMBS_UP", "THUMBS_DOWN"]:
        up = cm.get((hv, "THUMBS_UP"), 0)
        down = cm.get((hv, "THUMBS_DOWN"), 0)
        print(f"  {'Human_' + hv.split('_')[1]:>20} {up:>10} {down:>10}")

    # Disagreement analysis: which cases does human say DOWN but LLM says UP?
    human_down_llm_up = [p for p in pairs if p["human"] == "THUMBS_DOWN" and p["llm_binary"] == "THUMBS_UP"]
    human_up_llm_down = [p for p in pairs if p["human"] == "THUMBS_UP" and p["llm_binary"] == "THUMBS_DOWN"]

    print(f"\n  Human DOWN, LLM UP (LLM too lenient): {len(human_down_llm_up)}")
    for p in human_down_llm_up[:5]:
        print(f"    {p['query_id']}: badge={p['llm_badge']}, "
              f"faith={p['faithfulness']}, rel={p['relevance']}")

    print(f"  Human UP, LLM DOWN (LLM too strict): {len(human_up_llm_down)}")
    for p in human_up_llm_down[:5]:
        print(f"    {p['query_id']}: badge={p['llm_badge']}, "
              f"faith={p['faithfulness']}, rel={p['relevance']}")

    # Per-dimension concordance: correctness vs faithfulness threshold
    correct_faith = []
    for qid, e1 in e1_rows.items():
        r = results_map.get(qid)
        if not r or not e1.get("correctness"):
            continue
        faith = r.get("sync_faithfulness")
        if faith is None:
            continue
        correct_faith.append({
            "query_id": qid,
            "human_correctness": e1["correctness"],
            "faithfulness": faith,
            # correct/partial → faithful (≥0.7), incorrect → unfaithful (<0.7)
            "llm_faithful": faith >= 0.7,
            "human_correct": e1["correctness"] in ("correct", "partial"),
        })

    if correct_faith:
        dim_agree = sum(
            1 for p in correct_faith
            if p["llm_faithful"] == p["human_correct"]
        )
        print(f"\n  Correctness-vs-Faithfulness concordance:")
        print(f"    Pairs: {len(correct_faith)}")
        print(f"    Agreement: {dim_agree}/{len(correct_faith)} "
              f"= {100 * dim_agree / len(correct_faith):.1f}%")

        # Cases where faithfulness ≥ 0.7 but human says incorrect
        faith_high_incorrect = [
            p for p in correct_faith
            if p["llm_faithful"] and not p["human_correct"]
        ]
        print(f"    High faithfulness but human=incorrect: {len(faith_high_incorrect)}")
        for p in faith_high_incorrect[:5]:
            print(f"      {p['query_id']}: faithfulness={p['faithfulness']}")

    return {
        "n": n,
        "agreement": agree,
        "agreement_rate": agree / n,
        "confusion_matrix": {f"{h}_{l}": cm.get((h, l), 0)
                             for h in ["THUMBS_UP", "THUMBS_DOWN"]
                             for l in ["THUMBS_UP", "THUMBS_DOWN"]},
        "human_down_llm_up": len(human_down_llm_up),
        "human_up_llm_down": len(human_up_llm_down),
        "correctness_faithfulness": {
            "n": len(correct_faith),
            "agreement": dim_agree if correct_faith else None,
            "agreement_rate": dim_agree / len(correct_faith) if correct_faith else None,
            "high_faith_incorrect": len(faith_high_incorrect) if correct_faith else None,
        } if correct_faith else None,
    }


# ═══════════════════════════════════════════════════════════════
# FAILURE INJECTION
# ═══════════════════════════════════════════════════════════════

def failure_injection() -> dict:
    print("\n" + "=" * 60)
    print("FAILURE INJECTION")
    print("=" * 60)

    figure_path = FI_DIR / "figure_data.json"
    if not figure_path.exists():
        print(f"  ERROR: {figure_path} not found")
        return {"status": "MISSING", "error": str(figure_path)}

    with open(figure_path) as f:
        fi_data = json.load(f)

    summary = fi_data["summary"]

    print(f"\n  Experiment: {summary['experiment']}")
    print(f"  Date: {summary['date']}")
    print(f"  Sessions: {summary['total_sessions']} "
          f"({summary['injected']} injected, {summary['control']} control)")
    print(f"  Failure types: {summary['failure_types_tested']}")

    for label, key in [("Unblinded (ground truth)", "unblinded_results"),
                       ("Blinded (blinded evaluator)", "blinded_results")]:
        r = summary[key]
        print(f"\n  {label}:")
        print(f"    TP={r['TP']} TN={r['TN']} FP={r['FP']} FN={r['FN']}")
        print(f"    Accuracy={r['accuracy']:.4f}  Precision={r['precision']:.4f}  "
              f"Recall={r['recall']:.4f}  F1={r['f1']:.4f}")

        # Clopper-Pearson 95% CI for accuracy
        n = r["TP"] + r["TN"] + r["FP"] + r["FN"]
        k = r["TP"] + r["TN"]
        ci_low, ci_high = _clopper_pearson(k, n)
        print(f"    Clopper-Pearson 95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

    print(f"\n  Per-type recall (blinded):")
    for ftype, recall_str in summary["per_type_recall_blinded"].items():
        print(f"    {ftype}: {recall_str}")

    print(f"\n  Key finding: {summary['key_finding']}")

    # Build structured output
    blinded = summary["blinded_results"]
    unblinded = summary["unblinded_results"]
    n_b = blinded["TP"] + blinded["TN"] + blinded["FP"] + blinded["FN"]
    n_u = unblinded["TP"] + unblinded["TN"] + unblinded["FP"] + unblinded["FN"]

    exp3 = {
        "sessions": summary["total_sessions"],
        "injected": summary["injected"],
        "control": summary["control"],
        "failure_types": summary["failure_types_tested"],
        "blinded": {
            **{k: blinded[k] for k in ["TP", "TN", "FP", "FN",
                                        "accuracy", "precision", "recall", "f1"]},
            "clopper_pearson_95ci": list(_clopper_pearson(
                blinded["TP"] + blinded["TN"], n_b)),
        },
        "unblinded": {
            **{k: unblinded[k] for k in ["TP", "TN", "FP", "FN",
                                          "accuracy", "precision", "recall", "f1"]},
            "clopper_pearson_95ci": list(_clopper_pearson(
                unblinded["TP"] + unblinded["TN"], n_u)),
        },
        "per_type_recall_blinded": summary["per_type_recall_blinded"],
        "per_session": summary["per_session_results"],
        "key_finding": summary["key_finding"],
    }
    save_json(exp3, "failure_injection.json")
    return exp3


def _clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson exact binomial confidence interval."""
    from scipy.stats import beta as beta_dist
    if k == 0:
        lo = 0.0
    else:
        lo = beta_dist.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = beta_dist.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


# ═══════════════════════════════════════════════════════════════
# IDENTITY TRIPLE VALIDATION
# ═══════════════════════════════════════════════════════════════

def identity_triple_validation(conn, results: list[dict]) -> dict:
    print("\n" + "=" * 60)
    print("IDENTITY TRIPLE VALIDATION")
    print("=" * 60)

    cur = conn.cursor()

    # Check for natural safety retries in the 150-query corpus
    cur.execute("""
        SELECT trace_id, COUNT(DISTINCT chain_id) AS chain_count
        FROM agent_execution_logs
        WHERE trace_id = ANY(%s)
        GROUP BY trace_id
        HAVING COUNT(DISTINCT chain_id) > 1
    """, ([r["trace_id"] for r in results],))
    multi_chain = cur.fetchall()

    print(f"\n  Traces with multiple chain_ids (natural retries): {len(multi_chain)}")
    for row in multi_chain:
        print(f"    trace_id={row[0]}: {row[1]} chains")

    # Select candidates for forced-threshold re-run
    candidates = [
        r for r in results
        if r.get("actual_trace_variant") == "troubleshoot_full"
        and r.get("badge") == "green"
    ][:5]

    print(f"\n  Candidates for forced-threshold re-run: {len(candidates)}")
    for c in candidates:
        print(f"    {c['query_id']} (trace={c['trace_id'][:12]}...)")

    print(f"\n  Status: PENDING forced-threshold re-run")
    print(f"  Protocol: set _SURVIVAL_THRESHOLD = 1.01 in pipeline/safety_judge_gate.py")
    print(f"  Re-run {len(candidates)} queries, then restore threshold to 0.85")
    print(f"  Report as: 'Safety retry validated under controlled trigger (threshold=1.01);")
    print(f"  not triggered at operational threshold (0.85) in 150-query corpus.'")

    exp4 = {
        "natural_retry_count": len(multi_chain),
        "natural_retries": [{"trace_id": r[0], "chain_count": r[1]} for r in multi_chain],
        "operational_threshold": 0.85,
        "forced_threshold": 1.01,
        "candidates": [r["query_id"] for r in candidates],
        "status": "PENDING",
    }
    save_json(exp4, "identity_triple.json")
    return exp4


# ═══════════════════════════════════════════════════════════════
# PROVENANCE OVERHEAD
# ═══════════════════════════════════════════════════════════════

def provenance_overhead(results: list[dict]) -> dict:
    print("\n" + "=" * 60)
    print("PROVENANCE OVERHEAD")
    print("=" * 60)

    # Check for existing overhead measurement files
    overhead_with = BASE / "results" / "overhead_with_audit.json"
    overhead_without = BASE / "results" / "overhead_no_audit.json"

    if overhead_with.exists() and overhead_without.exists():
        import statistics

        with open(overhead_with) as f:
            with_data = json.load(f)
        with open(overhead_without) as f:
            without_data = json.load(f)

        with_median = statistics.median(with_data)
        without_median = statistics.median(without_data)
        overhead_ms = with_median - without_median
        overhead_pct = (overhead_ms / without_median) * 100 if without_median else 0

        print(f"\n  With audit median: {with_median:.1f} ms")
        print(f"  Without audit median: {without_median:.1f} ms")
        print(f"  Overhead: {overhead_ms:.1f} ms ({overhead_pct:.2f}%)")

        exp5 = {
            "status": "COMPLETE",
            "with_audit_median_ms": with_median,
            "without_audit_median_ms": without_median,
            "overhead_ms": overhead_ms,
            "overhead_pct": overhead_pct,
            "n_with": len(with_data),
            "n_without": len(without_data),
        }
    else:
        # Compute baseline latency stats from eval_run_v2.json
        durations = [r["duration_ms"] for r in results if r.get("duration_ms")]
        if durations:
            import statistics
            print(f"\n  Baseline latency from eval corpus (with audit):")
            print(f"    n = {len(durations)}")
            print(f"    median = {statistics.median(durations):.0f} ms")
            print(f"    mean = {statistics.mean(durations):.0f} ms")
            print(f"    p95 = {sorted(durations)[int(0.95 * len(durations))]:.0f} ms")

        print(f"\n  Status: PENDING — no overhead measurement files found")
        print(f"  Expected: {overhead_with}")
        print(f"  Expected: {overhead_without}")
        print(f"  Protocol: run 20 queries × 3 runs each with/without AUDIT_LOGGING_ENABLED")

        exp5 = {
            "status": "PENDING",
            "baseline_latency": {
                "n": len(durations),
                "median_ms": statistics.median(durations) if durations else None,
                "mean_ms": statistics.mean(durations) if durations else None,
            } if durations else None,
        }

    save_json(exp5, "provenance_overhead.json")
    return exp5


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    results = load_results()
    print(f"Loaded {len(results)} queries from {RESULTS_PATH}\n")

    conn = psycopg.connect(DB_URL)

    try:
        exp1 = provenance_reconstruction(conn, results)
        exp2 = feedback_chain_validation(conn, results)
        exp3 = failure_injection()
        exp4 = identity_triple_validation(conn, results)
        exp5 = provenance_overhead(results)
    finally:
        conn.close()

    # ── Summary ───────────────────────────────────────────────
    pc = exp1["provenance_completeness"]
    sp = exp1["safety_provenance"]
    fc = exp2["full_chain"]

    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    print(f"""
Provenance Reconstruction
  Provenance Completeness: {pc['resolved']}/{pc['total']} ({100*(pc['percentage'] or 0):.4f}%)
  Safety Provenance: {sp['resolved']}/{sp['total']}
  KB-Recon@t: per-document (see provenance_reconstruction.json)
  LC@d: per-document (see provenance_reconstruction.json)
  Process Mining Fitness: {exp1['process_mining_fitness'].get('value')}

Feedback Chain Validation
  Total feedback records: {exp2['total_feedback']}
  Corrections: {exp2['corrections']}
  L3→L2→L1 full chain: {fc['resolved']}/{fc['total']}
  L2→L1 FK integrity: {exp2['l2_to_l1_fk_integrity']['value']}
  L3→L2 FK integrity: {exp2['l3_to_l2_fk_integrity']['value']}

Failure Injection
  Sessions: {exp3.get('sessions', 'N/A')} ({exp3.get('injected', '?')} injected, {exp3.get('control', '?')} control)
  Blinded accuracy: {exp3.get('blinded', {}).get('accuracy', 'N/A')}
  Unblinded accuracy: {exp3.get('unblinded', {}).get('accuracy', 'N/A')}

Identity Triple Validation
  Natural retries: {exp4['natural_retry_count']}
  Status: {exp4['status']}

Provenance Overhead
  Status: {exp5['status']}
""")
    print("All results saved to eval/results/metrics/")


if __name__ == "__main__":
    main()
