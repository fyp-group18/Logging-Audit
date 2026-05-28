#!/usr/bin/env python3
"""Human evaluation — interactive handoff for CC-assisted evaluators.

Usage:
    python eval/scripts/human_evaluate.py --evaluator 1
    python eval/scripts/human_evaluate.py --evaluator 2 --resume

This script is designed to be run inside Claude Code. CC presents each response
with full evidence (chunks, inline verdicts, safety signals) and recommends
verdicts. The evaluator confirms or overrides each recommendation.

Output:
    eval/results/human_eval/evaluator{N}_responses.csv  (response-level)
    eval/results/human_eval/evaluator{N}_steps.csv      (step-level)
    eval/results/human_eval/evaluator{N}_progress.json  (resume checkpoint)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HUMAN_EVAL_DIR = ROOT / "eval" / "results" / "human_eval"

# ── Verdict mapping from inline eval to human-facing recommendations ──

GROUNDING_TO_STEP_VERDICT = {
    # inline grounding_label → recommended step verdict
    "verbatim": "ACCEPT",
    "paraphrased": "ACCEPT",     # paraphrased but faithful → likely accept
    "ungrounded": "INCORRECT",   # not grounded in source → flag for review
}


def load_data():
    with open(HUMAN_EVAL_DIR / "evaluation_data.json") as f:
        return json.load(f)


def load_progress(evaluator: int) -> dict:
    path = HUMAN_EVAL_DIR / f"evaluator{evaluator}_progress.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"completed_responses": [], "completed_steps": {}}


def save_progress(evaluator: int, progress: dict):
    path = HUMAN_EVAL_DIR / f"evaluator{evaluator}_progress.json"
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def format_chunk_evidence(entry: dict) -> str:
    """Format retrieved chunks as readable evidence block."""
    chunks = entry.get("retrieved_chunks", [])
    if not chunks:
        return "  (no chunks retrieved)"

    # Deduplicate by chunk_id
    seen = set()
    unique = []
    for c in chunks:
        if c["chunk_id"] not in seen:
            seen.add(c["chunk_id"])
            unique.append(c)

    lines = []
    for i, c in enumerate(unique):
        safety_tag = " [SAFETY]" if c.get("has_safety_content") else ""
        section = c.get("section_header") or "—"
        lines.append(f"  Chunk {i+1} (doc={c['document_id']}, step_idx={c.get('step_index', '—')}){safety_tag}")
        lines.append(f"    Section: {section}")
        content = (c.get("content_preview") or "").strip().replace("\n", "\n    ")
        lines.append(f"    Content: {content[:400]}")
        lines.append("")
    return "\n".join(lines)


def format_step_verdict_evidence(entry: dict, step_idx: int) -> str:
    """Format inline eval's verdict for a specific step."""
    verdicts = entry.get("step_verdicts") or []
    # Match by step_id (s-0, s-1, ...) or index
    for sv in verdicts:
        sid = sv.get("step_id", "")
        if sid == f"s-{step_idx}":
            faithful = sv.get("faithful", "?")
            label = sv.get("grounding_label", "?")
            reason = sv.get("reason", "")
            source = sv.get("source_chunk_ids", [])
            return (
                f"  Inline eval: faithful={faithful}, grounding={label}\n"
                f"  Reason: {reason}\n"
                f"  Source chunks: {', '.join(str(s)[:12] for s in source) if source else '—'}"
            )
    return "  Inline eval: (no verdict for this step)"


def recommend_response_verdict(entry: dict) -> dict:
    """Generate response-level verdict recommendation with reasoning."""
    reasons = []

    # ── Correctness ──
    faith = entry.get("sync_faithfulness")
    relevance = entry.get("sync_answer_relevance")
    verdicts = entry.get("step_verdicts") or []
    unfaithful = sum(1 for v in verdicts if not v.get("faithful"))
    total_steps = len(entry.get("repair_steps", []))

    if faith is not None and faith >= 0.9 and relevance is not None and relevance >= 0.9:
        correctness = "correct"
        reasons.append(f"faithfulness={faith}, relevance={relevance} (both ≥0.9)")
    elif faith is not None and faith >= 0.6:
        correctness = "partial"
        reasons.append(f"faithfulness={faith} (moderate), {unfaithful}/{len(verdicts)} steps unfaithful")
    elif faith is not None:
        correctness = "incorrect"
        reasons.append(f"faithfulness={faith} (<0.6)")
    else:
        correctness = "partial"
        reasons.append("no inline scores available — needs manual check")

    # ── Completeness ──
    chunks = entry.get("chunks_retrieved", 0)
    if chunks >= 5 and total_steps >= 3:
        completeness = "complete"
        reasons.append(f"{chunks} chunks retrieved, {total_steps} steps generated")
    elif chunks >= 2:
        completeness = "partial"
        reasons.append(f"only {chunks} chunks, {total_steps} steps — may miss context")
    else:
        completeness = "incomplete"
        reasons.append(f"minimal evidence: {chunks} chunks, {total_steps} steps")

    # ── Safety ──
    safety_expected = entry.get("safety_expected", False)
    safety_chunks = entry.get("safety_chunks", 0)
    steps_text = " ".join(entry.get("repair_steps", []))
    has_safety_language = any(
        kw in steps_text.upper()
        for kw in ["CAUTION", "WARNING", "⚠", "DANGER", "SAFETY", "HAZARD"]
    )

    if not safety_expected:
        safety = "n_a"
        reasons.append("safety not expected for this query")
    elif has_safety_language:
        safety = "present"
        reasons.append("safety language found in response steps")
    elif safety_chunks > 0:
        safety = "partial"
        reasons.append(f"{safety_chunks} safety chunks retrieved but no safety language in steps")
    else:
        safety = "missing"
        reasons.append("safety expected but no safety content in chunks or steps")

    # ── Overall ──
    if correctness == "correct" and completeness in ("complete", "partial") and safety != "missing":
        overall = "THUMBS_UP"
    elif correctness == "incorrect" or safety == "missing":
        overall = "THUMBS_DOWN"
    else:
        overall = "THUMBS_UP"  # lean accept for partial
        reasons.append("borderline — leaning THUMBS_UP, please verify")

    return {
        "correctness": correctness,
        "completeness": completeness,
        "safety_assessment": safety,
        "overall_verdict": overall,
        "reasons": reasons,
    }


def recommend_step_verdict(entry: dict, step_idx: int, step_text: str) -> dict:
    """Generate step-level verdict recommendation with reasoning."""
    verdicts = entry.get("step_verdicts") or []
    reasons = []

    # Find matching inline verdict
    matched = None
    for sv in verdicts:
        if sv.get("step_id") == f"s-{step_idx}":
            matched = sv
            break

    if matched:
        faithful = matched.get("faithful", True)
        label = matched.get("grounding_label", "verbatim")
        reason = matched.get("reason", "")

        if faithful and label in ("verbatim", "paraphrased"):
            verdict = "ACCEPT"
            reasons.append(f"faithful={faithful}, grounding={label}")
        elif not faithful:
            verdict = "INCORRECT"
            reasons.append(f"unfaithful: {reason}")
        else:
            verdict = "MODIFY"
            reasons.append(f"grounding={label}, review needed: {reason}")
    else:
        # No inline verdict — check if step text looks grounded
        has_cite = "<cite" in step_text
        if has_cite:
            verdict = "ACCEPT"
            reasons.append("no inline verdict but step has citation references")
        else:
            verdict = "MODIFY"
            reasons.append("no inline verdict and no citations — verify against chunks")

    # Flag safety steps
    safety_kws = ["CAUTION", "WARNING", "⚠", "DANGER"]
    if any(kw in step_text.upper() for kw in safety_kws):
        reasons.append("contains safety language — verify accuracy carefully")

    return {"verdict": verdict, "reasons": reasons}


def build_handoff_document(data: list[dict], evaluator: int):
    """Build the complete handoff as a structured JSON for CC to walk through."""
    progress = load_progress(evaluator)
    completed = set(progress["completed_responses"])
    remaining = [d for d in data if d["query_id"] not in completed]

    print(f"\n{'='*60}")
    print(f"  HUMAN EVALUATION — Evaluator {evaluator}")
    print(f"{'='*60}")
    print(f"  Total responses: {len(data)}")
    print(f"  Already completed: {len(completed)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"{'='*60}\n")

    handoff = []

    for entry in remaining:
        qid = entry["query_id"]
        resp_rec = recommend_response_verdict(entry)
        steps = entry.get("repair_steps", [])

        step_recs = []
        for idx, step_text in enumerate(steps):
            step_rec = recommend_step_verdict(entry, idx, step_text)
            step_recs.append({
                "step_index": idx,
                "step_content": step_text,
                "inline_evidence": format_step_verdict_evidence(entry, idx),
                "recommended_verdict": step_rec["verdict"],
                "reasons": step_rec["reasons"],
            })

        handoff.append({
            "query_id": qid,
            "response_id": entry["response_id"],
            "thread_id": entry["thread_id"],
            "device_id": entry["device_id"],
            "variant": entry["variant"],
            "query_text": entry["query_text"],
            "root_cause": entry.get("root_cause", ""),
            "badge": entry["badge"],
            "sync_faithfulness": entry["sync_faithfulness"],
            "sync_answer_relevance": entry["sync_answer_relevance"],
            "safety_expected": entry["safety_expected"],
            "notes": entry["notes"],
            "chunk_evidence": format_chunk_evidence(entry),
            "repair_steps": steps,
            "response_recommendation": resp_rec,
            "step_recommendations": step_recs,
            "total_steps": len(steps),
            "chunks_retrieved": entry["chunks_retrieved"],
        })

    # Write handoff document
    handoff_path = HUMAN_EVAL_DIR / f"evaluator{evaluator}_handoff.json"
    with open(handoff_path, "w") as f:
        json.dump(handoff, f, indent=2, default=str)
    print(f"Written: {handoff_path.relative_to(ROOT)}")

    # Also write a human-readable markdown walkthrough
    md_path = HUMAN_EVAL_DIR / f"evaluator{evaluator}_walkthrough.md"
    with open(md_path, "w") as f:
        f.write(f"# Human Evaluation Walkthrough — Evaluator {evaluator}\n\n")
        f.write(f"**Responses to evaluate:** {len(handoff)}\n")
        f.write(f"**Total steps:** {sum(h['total_steps'] for h in handoff)}\n\n")
        f.write("---\n\n")
        f.write("## Instructions\n\n")
        f.write("For each response, CC will present:\n")
        f.write("1. The query and device context\n")
        f.write("2. The root cause analysis\n")
        f.write("3. Each repair step with chunk evidence and inline eval verdicts\n")
        f.write("4. A recommended verdict with reasoning\n\n")
        f.write("Your job: **confirm or override** each recommendation.\n\n")
        f.write("### Response-level verdicts\n")
        f.write("- **correctness**: correct / partial / incorrect\n")
        f.write("- **completeness**: complete / partial / incomplete\n")
        f.write("- **safety_assessment**: present / partial / missing / n_a\n")
        f.write("- **overall_verdict**: THUMBS_UP / THUMBS_DOWN\n\n")
        f.write("### Step-level verdicts\n")
        f.write("- **ACCEPT**: Step is accurate and grounded in source material\n")
        f.write("- **SKIP**: Step is irrelevant but not harmful\n")
        f.write("- **MODIFY**: Step needs correction (provide correction_text)\n")
        f.write("- **INCORRECT**: Step is factually wrong (provide correction_text)\n\n")
        f.write("---\n\n")

        for h in handoff:
            f.write(f"## {h['query_id']} — {h['variant']}\n\n")
            f.write(f"**Device:** {h['device_id']}  \n")
            f.write(f"**Badge:** {h['badge']} | ")
            f.write(f"**Faithfulness:** {h['sync_faithfulness']} | ")
            f.write(f"**Relevance:** {h['sync_answer_relevance']}  \n")
            f.write(f"**Safety expected:** {h['safety_expected']}  \n")
            if h["notes"]:
                f.write(f"**Notes:** {h['notes']}  \n")
            f.write(f"\n### Query\n> {h['query_text']}\n\n")

            if h["root_cause"]:
                f.write(f"### Root Cause\n{h['root_cause'][:1000]}\n\n")

            # Response recommendation
            rec = h["response_recommendation"]
            f.write(f"### Response Verdict (recommended)\n")
            f.write(f"| Dimension | Recommendation | Reasoning |\n")
            f.write(f"|---|---|---|\n")
            f.write(f"| Correctness | **{rec['correctness']}** | {rec['reasons'][0] if rec['reasons'] else '—'} |\n")
            f.write(f"| Completeness | **{rec['completeness']}** | {rec['reasons'][1] if len(rec['reasons']) > 1 else '—'} |\n")
            f.write(f"| Safety | **{rec['safety_assessment']}** | {rec['reasons'][2] if len(rec['reasons']) > 2 else '—'} |\n")
            f.write(f"| **Overall** | **{rec['overall_verdict']}** | |\n\n")

            # Chunk evidence
            f.write(f"### Retrieved Chunks ({h['chunks_retrieved']})\n")
            f.write(f"```\n{h['chunk_evidence']}\n```\n\n")

            # Steps
            f.write(f"### Repair Steps ({h['total_steps']})\n\n")
            for sr in h["step_recommendations"]:
                verdict_emoji = {"ACCEPT": "+", "SKIP": "~", "MODIFY": "?", "INCORRECT": "!"}
                marker = verdict_emoji.get(sr["recommended_verdict"], "?")
                f.write(f"**[{marker}] Step {sr['step_index']}** → rec: **{sr['recommended_verdict']}**\n")
                f.write(f"> {sr['step_content'][:300]}\n\n")
                if sr["reasons"]:
                    f.write(f"  _{'; '.join(sr['reasons'])}_\n\n")
                f.write(f"{sr['inline_evidence']}\n\n")

            f.write("---\n\n")

    print(f"Written: {md_path.relative_to(ROOT)}")

    # Write pre-filled CSVs with recommendations (evaluator overrides blank columns)
    resp_path = HUMAN_EVAL_DIR / f"evaluator{evaluator}_responses.csv"
    step_path = HUMAN_EVAL_DIR / f"evaluator{evaluator}_steps.csv"

    resp_fields = [
        "query_id", "response_id", "variant", "device_id", "query_text",
        "badge", "sync_faithfulness", "chunks_retrieved", "safety_expected", "notes",
        "rec_correctness", "rec_completeness", "rec_safety", "rec_overall",
        "rec_reasoning",
        # Evaluator fills (or confirms by copying rec_* values):
        "correctness", "completeness", "safety_assessment", "overall_verdict",
        "correction_text",
    ]
    with open(resp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=resp_fields)
        writer.writeheader()
        for h in handoff:
            rec = h["response_recommendation"]
            writer.writerow({
                "query_id": h["query_id"],
                "response_id": h["response_id"],
                "variant": h["variant"],
                "device_id": h["device_id"],
                "query_text": h["query_text"][:500],
                "badge": h["badge"],
                "sync_faithfulness": h["sync_faithfulness"],
                "chunks_retrieved": h["chunks_retrieved"],
                "safety_expected": h["safety_expected"],
                "notes": h["notes"][:200],
                "rec_correctness": rec["correctness"],
                "rec_completeness": rec["completeness"],
                "rec_safety": rec["safety_assessment"],
                "rec_overall": rec["overall_verdict"],
                "rec_reasoning": "; ".join(rec["reasons"])[:500],
                "correctness": "",
                "completeness": "",
                "safety_assessment": "",
                "overall_verdict": "",
                "correction_text": "",
            })
    print(f"Written: {resp_path.relative_to(ROOT)} ({len(handoff)} rows)")

    step_fields = [
        "query_id", "response_id", "step_index", "step_content",
        "rec_verdict", "rec_reasoning", "inline_faithful", "inline_grounding",
        # Evaluator fills:
        "verdict", "correction_text", "severity",
    ]
    total_step_rows = 0
    with open(step_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields)
        writer.writeheader()
        for h in handoff:
            verdicts_map = {}
            for sv in (data_entry_by_qid(data, h["query_id"]).get("step_verdicts") or []):
                verdicts_map[sv.get("step_id", "")] = sv

            for sr in h["step_recommendations"]:
                sv = verdicts_map.get(f"s-{sr['step_index']}", {})
                writer.writerow({
                    "query_id": h["query_id"],
                    "response_id": h["response_id"],
                    "step_index": sr["step_index"],
                    "step_content": sr["step_content"][:2000],
                    "rec_verdict": sr["recommended_verdict"],
                    "rec_reasoning": "; ".join(sr["reasons"])[:300],
                    "inline_faithful": sv.get("faithful", ""),
                    "inline_grounding": sv.get("grounding_label", ""),
                    "verdict": "",
                    "correction_text": "",
                    "severity": "",
                })
                total_step_rows += 1

            # Handle 0-step entries
            if not h["step_recommendations"]:
                writer.writerow({
                    "query_id": h["query_id"],
                    "response_id": h["response_id"],
                    "step_index": 0,
                    "step_content": "(no repair steps — see response_text in evaluation_data.json)",
                    "rec_verdict": "SKIP",
                    "rec_reasoning": "no steps generated",
                    "inline_faithful": "",
                    "inline_grounding": "",
                    "verdict": "",
                    "correction_text": "",
                    "severity": "",
                })
                total_step_rows += 1

    print(f"Written: {step_path.relative_to(ROOT)} ({total_step_rows} rows)")

    # Print summary stats for CC agent to relay
    print(f"\n{'='*60}")
    print(f"  RECOMMENDATION SUMMARY")
    print(f"{'='*60}")

    from collections import Counter
    resp_recs = Counter(h["response_recommendation"]["overall_verdict"] for h in handoff)
    print(f"  Response-level overall:")
    for k, v in resp_recs.items():
        print(f"    {k}: {v}")

    step_recs = Counter()
    for h in handoff:
        for sr in h["step_recommendations"]:
            step_recs[sr["recommended_verdict"]] += 1
    print(f"  Step-level verdicts:")
    for k, v in step_recs.items():
        print(f"    {k}: {v}")

    correctness_recs = Counter(h["response_recommendation"]["correctness"] for h in handoff)
    print(f"  Correctness breakdown:")
    for k, v in correctness_recs.items():
        print(f"    {k}: {v}")

    safety_recs = Counter(h["response_recommendation"]["safety_assessment"] for h in handoff)
    print(f"  Safety assessment:")
    for k, v in safety_recs.items():
        print(f"    {k}: {v}")

    # Highlight items needing extra attention
    flagged = [h for h in handoff if h["response_recommendation"]["overall_verdict"] == "THUMBS_DOWN"]
    if flagged:
        print(f"\n  FLAGGED FOR REVIEW ({len(flagged)} responses):")
        for h in flagged:
            rec = h["response_recommendation"]
            print(f"    {h['query_id']}: correctness={rec['correctness']}, safety={rec['safety_assessment']}")

    unfaithful_steps = []
    for h in handoff:
        for sr in h["step_recommendations"]:
            if sr["recommended_verdict"] == "INCORRECT":
                unfaithful_steps.append((h["query_id"], sr["step_index"]))
    if unfaithful_steps:
        print(f"\n  UNFAITHFUL STEPS ({len(unfaithful_steps)}):")
        for qid, idx in unfaithful_steps[:20]:
            print(f"    {qid} step {idx}")
        if len(unfaithful_steps) > 20:
            print(f"    ... and {len(unfaithful_steps) - 20} more")

    print(f"\n{'='*60}")
    print(f"  NEXT STEPS")
    print(f"{'='*60}")
    print(f"  1. Open evaluator{evaluator}_walkthrough.md for full evidence")
    print(f"  2. Fill verdict columns in evaluator{evaluator}_responses.csv")
    print(f"  3. Fill verdict columns in evaluator{evaluator}_steps.csv")
    print(f"  4. For ACCEPT recommendations: confirm by copying rec_* → verdict")
    print(f"  5. For INCORRECT/MODIFY: review chunk evidence, provide correction_text")
    print(f"  6. Run human_eval_import.py after both evaluators complete")


def data_entry_by_qid(data, qid):
    for d in data:
        if d["query_id"] == qid:
            return d
    return {}


def main():
    parser = argparse.ArgumentParser(description="Human evaluation — CC-assisted")
    parser.add_argument("--evaluator", type=int, required=True, choices=[1, 2])
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()

    data = load_data()
    build_handoff_document(data, args.evaluator)


if __name__ == "__main__":
    main()
