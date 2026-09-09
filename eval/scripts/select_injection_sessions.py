"""
Select 12 sessions for the failure injection protocol — 30-query pilot only.

PILOT-ONLY. This script consumes the 30-query pilot corpus, in which
expected_trace_variant is an INTEGER (variants 4 and 5 below). The 150-query
corpus used elsewhere in the paper labels variants with STRINGS
("followup", "replace_part", ...), so the variant filters here match nothing
against it and the mandatory-slot logic is silently skipped. The 12 sessions
reported in the paper were drawn from the pilot, which is why the protocol is
documented against the pilot rather than the 150-session corpus.

Filters for sessions with sufficient audit data (>=3 chunks, >=5 nodes, PASS status),
then applies stratified sampling to ensure both intent types, safety-tagged sessions,
and trace variants 4+5 are represented.

Usage:
    python -m eval.scripts.select_injection_sessions \
        --run-log <pilot run log> \
        --queries eval/generation/eval_queries.json \
        --output eval/results/failure_injection/failure_injection_sessions.json

    The pilot run log is not redistributed in this repository; the selected
    sessions it produced are recorded in
    eval/results/metrics/failure_injection.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


TARGET_COUNT = 12
MIN_CHUNKS = 3
MIN_NODES = 5
MIN_SAFETY = 4
MIN_VARIANT_4 = 2
MIN_VARIANT_5 = 2


def load_query_metadata(queries_path: Path) -> dict[str, dict]:
    """Index eval_queries.json by query_id for cross-reference."""
    with open(queries_path) as f:
        queries = json.load(f)
    return {q["query_id"]: q for q in queries}


def filter_eligible(run_log: list[dict], query_meta: dict[str, dict]) -> list[dict]:
    """Apply eligibility filters: PASS, response_id, thread_id, >=3 chunks, >=5 nodes."""
    eligible = []
    for entry in run_log:
        if entry.get("status") != "PASS":
            continue
        if not entry.get("response_id") or not entry.get("thread_id"):
            continue
        if entry.get("chunks_retrieved", 0) < MIN_CHUNKS:
            continue
        nodes = entry.get("nodes_seen", [])
        if len(nodes) < MIN_NODES:
            continue

        qid = entry["query_id"]
        meta = query_meta.get(qid, {})

        eligible.append({
            "query_id": qid,
            "response_id": entry["response_id"],
            "thread_id": entry["thread_id"],
            "query_text": meta.get("query_text", ""),
            "expected_intent": meta.get("expected_intent", entry.get("expected_intent", "")),
            "expected_trace_variant": meta.get(
                "expected_trace_variant", entry.get("expected_trace_variant")
            ),
            "safety_expected": meta.get("safety_expected", False),
            "chunks_retrieved": entry.get("chunks_retrieved", 0),
            "nodes_seen": nodes,
        })
    return eligible


def stratified_select(eligible: list[dict], seed: int = 42) -> list[dict]:
    """Stratified sample ensuring representation constraints are met."""
    rng = random.Random(seed)

    safety_pool = [s for s in eligible if s["safety_expected"]]
    v4_pool = [s for s in eligible if s["expected_trace_variant"] == 4]
    v5_pool = [s for s in eligible if s["expected_trace_variant"] == 5]
    replace_pool = [s for s in eligible if s["expected_intent"] == "replace_part"]
    troubleshoot_pool = [s for s in eligible if s["expected_intent"] == "troubleshoot"]

    if len(safety_pool) < MIN_SAFETY:
        print(f"WARNING: Only {len(safety_pool)} safety sessions available (need {MIN_SAFETY})")
    if len(v4_pool) < MIN_VARIANT_4:
        print(f"WARNING: Only {len(v4_pool)} variant-4 sessions (need {MIN_VARIANT_4})")
    if len(v5_pool) < MIN_VARIANT_5:
        print(f"WARNING: Only {len(v5_pool)} variant-5 sessions (need {MIN_VARIANT_5})")
    if not replace_pool or not troubleshoot_pool:
        print("WARNING: Missing one of the required intent types")

    selected: list[dict] = []
    selected_ids: set[str] = set()

    def add(session: dict) -> bool:
        if session["query_id"] in selected_ids:
            return False
        selected.append(session)
        selected_ids.add(session["query_id"])
        return True

    # Mandatory slots: safety sessions
    rng.shuffle(safety_pool)
    for s in safety_pool:
        if len([x for x in selected if x["safety_expected"]]) >= MIN_SAFETY:
            break
        add(s)

    # Mandatory slots: variant 4
    rng.shuffle(v4_pool)
    for s in v4_pool:
        if len([x for x in selected if x["expected_trace_variant"] == 4]) >= MIN_VARIANT_4:
            break
        add(s)

    # Mandatory slots: variant 5
    rng.shuffle(v5_pool)
    for s in v5_pool:
        if len([x for x in selected if x["expected_trace_variant"] == 5]) >= MIN_VARIANT_5:
            break
        add(s)

    # Ensure at least 1 replace_part and 1 troubleshoot
    intents_present = {s["expected_intent"] for s in selected}
    if "replace_part" not in intents_present and replace_pool:
        rng.shuffle(replace_pool)
        add(replace_pool[0])
    if "troubleshoot" not in intents_present and troubleshoot_pool:
        rng.shuffle(troubleshoot_pool)
        add(troubleshoot_pool[0])

    # Fill remaining from general pool
    remaining = [s for s in eligible if s["query_id"] not in selected_ids]
    rng.shuffle(remaining)
    for s in remaining:
        if len(selected) >= TARGET_COUNT:
            break
        add(s)

    if len(selected) < TARGET_COUNT:
        print(
            f"WARNING: Could only select {len(selected)}/{TARGET_COUNT} sessions "
            f"({len(eligible)} eligible)"
        )

    return selected[:TARGET_COUNT]


def assign_session_ids(sessions: list[dict]) -> list[dict]:
    """Assign sequential FI-XXX session IDs."""
    for i, session in enumerate(sessions, 1):
        session["session_id"] = f"FI-{i:03d}"
        session["source_query_id"] = session.pop("query_id")
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select 12 sessions for failure injection protocol"
    )
    parser.add_argument("--run-log", type=Path, required=True, help="Path to eval_run_log.json")
    parser.add_argument("--queries", type=Path, required=True, help="Path to eval_queries.json")
    parser.add_argument("--output", type=Path, required=True, help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    with open(args.run_log) as f:
        run_log = json.load(f)
    print(f"Loaded {len(run_log)} entries from eval run log")

    query_meta = load_query_metadata(args.queries)
    print(f"Loaded {len(query_meta)} query definitions")

    eligible = filter_eligible(run_log, query_meta)
    print(f"Eligible sessions after filtering: {len(eligible)}")

    if len(eligible) < TARGET_COUNT:
        print(f"ERROR: Need {TARGET_COUNT} sessions but only {len(eligible)} eligible")
        sys.exit(1)

    selected = stratified_select(eligible, seed=args.seed)
    selected = assign_session_ids(selected)

    # Summary
    safety_count = sum(1 for s in selected if s["safety_expected"])
    v4_count = sum(1 for s in selected if s["expected_trace_variant"] == 4)
    v5_count = sum(1 for s in selected if s["expected_trace_variant"] == 5)
    intents = {s["expected_intent"] for s in selected}
    print(f"\nSelected {len(selected)} sessions:")
    print(f"  Safety-tagged:  {safety_count}")
    print(f"  Variant #4:     {v4_count}")
    print(f"  Variant #5:     {v5_count}")
    print(f"  Intent types:   {', '.join(sorted(intents))}")

    for s in selected:
        print(
            f"  {s['session_id']} <- {s['source_query_id']}  "
            f"intent={s['expected_intent']}  v={s['expected_trace_variant']}  "
            f"safety={s['safety_expected']}  chunks={s['chunks_retrieved']}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
