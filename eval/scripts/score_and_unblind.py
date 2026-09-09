"""
Unblind and score the failure injection experiment.

Reconciles the injection designer's sealed assignment against the blinded evaluator's diagnosis log.
Computes detection metrics (accuracy, precision, recall, F1), per-failure-type
detection rates, type classification accuracy, and confusion matrices.

Usage:
    cd backend && uv run python -m eval.scripts.score_and_unblind \
        --assignment eval/results/failure_injection/experiment_run_001/assignments.json \
        --diagnosis eval/results/failure_injection/experiment_run_001/diagnostic_trace_log.json \
        --manifest-hash eval/results/failure_injection/experiment_run_001/manifest_sha256.txt \
        --output eval/results/failure_injection/experiment_run_001/scoring_summary.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FAILURE_TYPES = ["chunk_deletion", "score_perturbation", "log_gap", "manifest_tamper"]


def verify_manifest(assignment_path: Path, manifest_hash_path: Path) -> bool:
    """Verify SHA-256 of assignments.json matches manifest_sha256.txt."""
    expected = manifest_hash_path.read_text().strip()
    actual = hashlib.sha256(assignment_path.read_bytes()).hexdigest()
    return actual == expected


def load_ground_truth(assignment_path: Path) -> dict[str, dict]:
    """Load assignments and index by session_id."""
    with open(assignment_path) as f:
        data = json.load(f)
    return {s["session_id"]: s for s in data["sessions"]}


def load_diagnosis(diagnosis_path: Path) -> dict[str, dict]:
    """Load the blinded evaluator's diagnosis log and index by session_id."""
    with open(diagnosis_path) as f:
        data = json.load(f)
    return {s["session_id"]: s for s in data["sessions"]}


def classify(
    ground_truth: dict[str, dict], diagnosis: dict[str, dict]
) -> list[dict]:
    """Classify each session as TP/TN/FP/FN."""
    results = []
    for sid, truth in ground_truth.items():
        diag = diagnosis.get(sid)
        if diag is None:
            print(f"WARNING: {sid} missing from diagnosis log")
            continue

        is_injected = truth["group"] == "injection"
        detected = diag.get("failure_detected") is True

        if is_injected and detected:
            classification = "TP"
        elif not is_injected and not detected:
            classification = "TN"
        elif not is_injected and detected:
            classification = "FP"
        else:
            classification = "FN"

        results.append({
            "session_id": sid,
            "ground_truth_group": truth["group"],
            "ground_truth_type": truth.get("failure_type"),
            "detected": detected,
            "guessed_type": diag.get("failure_type_guess", ""),
            "confidence": diag.get("confidence", ""),
            "classification": classification,
            "time_to_diagnosis_seconds": diag.get("time_to_diagnosis_seconds"),
            "reasoning_trace": diag.get("reasoning_trace", []),
        })

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute detection and type-classification metrics."""
    tp = sum(1 for r in results if r["classification"] == "TP")
    tn = sum(1 for r in results if r["classification"] == "TN")
    fp = sum(1 for r in results if r["classification"] == "FP")
    fn = sum(1 for r in results if r["classification"] == "FN")
    total = tp + tn + fp + fn

    accuracy = (tp + tn) / total if total else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    # Per-failure-type detection rate
    type_detection: dict[str, dict] = {}
    for ftype in FAILURE_TYPES:
        injected_of_type = [r for r in results if r["ground_truth_type"] == ftype]
        detected_of_type = [r for r in injected_of_type if r["classification"] == "TP"]
        type_detection[ftype] = {
            "injected": len(injected_of_type),
            "detected": len(detected_of_type),
            "rate": len(detected_of_type) / len(injected_of_type) if injected_of_type else None,
        }

    # Type classification accuracy (among TPs: did the blinded evaluator guess the correct type?)
    tp_results = [r for r in results if r["classification"] == "TP"]
    correct_type = sum(
        1 for r in tp_results
        if r["guessed_type"] == r["ground_truth_type"]
    )
    type_accuracy = correct_type / len(tp_results) if tp_results else None

    return {
        "detection": {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_type_detection": type_detection,
        "type_classification": {
            "correct": correct_type,
            "total_tp": len(tp_results),
            "accuracy": round(type_accuracy, 4) if type_accuracy is not None else None,
        },
    }


def build_confusion_matrix_2x2(results: list[dict]) -> list[list[int]]:
    """2x2 detection confusion matrix: [[TP, FN], [FP, TN]]."""
    tp = sum(1 for r in results if r["classification"] == "TP")
    fp = sum(1 for r in results if r["classification"] == "FP")
    fn = sum(1 for r in results if r["classification"] == "FN")
    tn = sum(1 for r in results if r["classification"] == "TN")
    return [[tp, fn], [fp, tn]]


def build_type_confusion_matrix(results: list[dict]) -> dict:
    """4x4 type classification matrix for TP sessions only."""
    tp_results = [r for r in results if r["classification"] == "TP"]
    matrix: dict[str, dict[str, int]] = {
        actual: {guessed: 0 for guessed in FAILURE_TYPES + ["unknown"]}
        for actual in FAILURE_TYPES
    }
    for r in tp_results:
        actual = r["ground_truth_type"]
        guessed = r["guessed_type"] if r["guessed_type"] in FAILURE_TYPES else "unknown"
        if actual in matrix:
            matrix[actual][guessed] += 1
    return matrix


def build_cross_check_tasks(results: list[dict]) -> list[dict]:
    """Prioritized list of FN and FP sessions for post-hoc investigation."""
    tasks = []
    for r in results:
        if r["classification"] in ("FN", "FP"):
            tasks.append({
                "session_id": r["session_id"],
                "classification": r["classification"],
                "ground_truth_group": r["ground_truth_group"],
                "ground_truth_type": r["ground_truth_type"],
                "detected": r["detected"],
                "guessed_type": r["guessed_type"],
                "priority": "P0" if r["classification"] == "FN" else "P1",
                "investigation_notes": "",
            })
    return sorted(tasks, key=lambda t: t["priority"])


def format_summary(
    metrics: dict,
    confusion_2x2: list[list[int]],
    type_confusion: dict,
    results: list[dict],
) -> str:
    """Format scoring summary as plain text for the paper."""
    lines = []
    lines.append("=" * 70)
    lines.append("FAILURE INJECTION PROTOCOL — SCORING SUMMARY")
    lines.append("=" * 70)

    d = metrics["detection"]
    lines.append("")
    lines.append("Detection Metrics (binary: injected vs control)")
    lines.append("-" * 50)
    lines.append(f"  True Positives:   {d['tp']}")
    lines.append(f"  True Negatives:   {d['tn']}")
    lines.append(f"  False Positives:  {d['fp']}")
    lines.append(f"  False Negatives:  {d['fn']}")
    lines.append(f"  Accuracy:         {d['accuracy']:.4f}")
    lines.append(f"  Precision:        {d['precision']:.4f}")
    lines.append(f"  Recall:           {d['recall']:.4f}")
    lines.append(f"  F1 Score:         {d['f1']:.4f}")

    lines.append("")
    lines.append("Detection Confusion Matrix")
    lines.append("-" * 50)
    lines.append("                    Predicted")
    lines.append("                  Injected  Control")
    lines.append(f"  Actual Injected    {confusion_2x2[0][0]:>4d}     {confusion_2x2[0][1]:>4d}")
    lines.append(f"  Actual Control     {confusion_2x2[1][0]:>4d}     {confusion_2x2[1][1]:>4d}")

    lines.append("")
    lines.append("Per-Failure-Type Detection Rate")
    lines.append("-" * 50)
    for ftype, info in metrics["per_type_detection"].items():
        rate_str = f"{info['rate']:.2%}" if info["rate"] is not None else "N/A"
        lines.append(f"  {ftype:25s}  {info['detected']}/{info['injected']}  ({rate_str})")

    tc = metrics["type_classification"]
    lines.append("")
    lines.append("Type Classification Accuracy (among true positives)")
    lines.append("-" * 50)
    if tc["accuracy"] is not None:
        lines.append(f"  Correct type: {tc['correct']}/{tc['total_tp']} ({tc['accuracy']:.2%})")
    else:
        lines.append("  No true positives to evaluate.")

    lines.append("")
    lines.append("Type Classification Confusion Matrix (TP only)")
    lines.append("-" * 50)
    header = f"  {'Actual \\ Guessed':25s}" + "".join(f"{t[:10]:>12s}" for t in FAILURE_TYPES + ["unknown"])
    lines.append(header)
    for actual in FAILURE_TYPES:
        row_vals = "".join(
            f"{type_confusion[actual][g]:>12d}" for g in FAILURE_TYPES + ["unknown"]
        )
        lines.append(f"  {actual:25s}{row_vals}")

    # Per-session detail
    lines.append("")
    lines.append("Per-Session Results")
    lines.append("-" * 50)
    lines.append(f"  {'Session':10s} {'Class':5s} {'Truth':18s} {'Detected':9s} {'Guessed':18s} {'Time(s)':>8s}")
    for r in sorted(results, key=lambda x: x["session_id"]):
        gt = r["ground_truth_type"] or "control"
        det = "Y" if r["detected"] else "N"
        guessed = r["guessed_type"] or "-"
        time_s = f"{r['time_to_diagnosis_seconds']:.0f}" if r["time_to_diagnosis_seconds"] else "-"
        lines.append(
            f"  {r['session_id']:10s} {r['classification']:5s} {gt:18s} {det:9s} {guessed:18s} {time_s:>8s}"
        )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unblind and score failure injection experiment"
    )
    parser.add_argument(
        "--assignment", type=Path, required=True,
        help="Path to assignments.json (the injection designer's sealed ground truth)",
    )
    parser.add_argument(
        "--diagnosis", type=Path, required=True,
        help="Path to diagnostic_trace_log.json (the blinded evaluator's completed form)",
    )
    parser.add_argument(
        "--manifest-hash", type=Path, required=True,
        help="Path to manifest_sha256.txt",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for scoring_summary.txt",
    )
    args = parser.parse_args()

    # Step 1: Verify manifest integrity
    print("Verifying manifest integrity...", end=" ")
    if not verify_manifest(args.assignment, args.manifest_hash):
        print("FAILED")
        print("ERROR: assignments.json SHA-256 does not match manifest_sha256.txt")
        print("The assignment file may have been tampered with.")
        sys.exit(1)
    print("OK")

    # Step 2: Load data
    ground_truth = load_ground_truth(args.assignment)
    diagnosis = load_diagnosis(args.diagnosis)
    print(f"Ground truth: {len(ground_truth)} sessions")
    print(f"Diagnosis:    {len(diagnosis)} sessions")

    # Step 3: Classify
    results = classify(ground_truth, diagnosis)

    # Step 4: Compute metrics
    metrics = compute_metrics(results)
    confusion_2x2 = build_confusion_matrix_2x2(results)
    type_confusion = build_type_confusion_matrix(results)

    # Step 5: Format and write summary
    summary = format_summary(metrics, confusion_2x2, type_confusion, results)
    print(f"\n{summary}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(summary + "\n")
    print(f"\nSummary saved to {args.output}")

    # Step 6: Write cross-check tasks
    cross_check = build_cross_check_tasks(results)
    cross_check_path = args.output.parent / "cross_check_tasks.json"
    cross_check_path.write_text(json.dumps(cross_check, indent=2))
    print(f"Cross-check tasks saved to {cross_check_path} ({len(cross_check)} items)")


if __name__ == "__main__":
    main()
