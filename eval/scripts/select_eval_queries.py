"""
30-query pilot selector — produces eval/generation/eval_queries.json.

This is NOT the generator for the 150-query corpus used in the paper. The
150-query corpus is shipped as data at eval/generation/eval_queries_v2.json;
no generator for it exists in this repository. This script selects the earlier
30-query pilot from which the failure-injection sessions were drawn.

Allocation: 23 from dataset (variants #3/#4/#5) + 7 manual placeholders (variants #1/#2/#6-9).

Constraints:
  - Dataset queries: ~12 table + ~11 prose, ~11 part_replacement + ~12 operational_anomaly
  - section_system diversity: max 4 per system
  - safety_expected >= 6
  - 2 obscure-system entries for variant #3 (troubleshoot short-circuit)

Input, NOT redistributed here:
    eval/test_set_data/synthetic_eval/datasets/combined_eval_dataset.json
    This file is derived from the public evaluation dataset repository
    (https://github.com/fyp-group18/aircraft-maintenance-rag-eval) and is not
    included in this repository. Without it the script cannot run; the pilot
    corpus it produced is shipped instead, at eval/generation/eval_queries.json.

Usage:
    python -m eval.scripts.select_eval_queries
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

# Repository-relative: eval/scripts/ -> eval/
_EVAL_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = _EVAL_ROOT / "test_set_data" / "synthetic_eval" / "datasets" / "combined_eval_dataset.json"
OUTPUT_DIR = _EVAL_ROOT / "generation"
QUERIES_PATH = OUTPUT_DIR / "eval_queries.json"
COVERAGE_PATH = OUTPUT_DIR / "eval_queries_coverage.md"

DEVICE_MAP = {
    "SC10000AMM": "CubCrafters CC11-100 Sport Cub S2",
    "AS-AMM-01-000": "Aerospool WT9 Dynamic LSA Club",
}

# Systems with known safety hazards (LOTO, pressurized, electrical, fuel, exhaust)
SAFETY_SYSTEMS_TABLE = {
    "Landing Gear (6.3.7)", "Fuel (Ch 28)", "Engine (6.3.10)",
    "Electrical Power Systems (6.3.17)", "Exhaust (Ch 78)", "Propellers (Ch 61)",
    "Oil (Ch 79)", "Ignition (Ch 74)", "Engine Fuel and Control (Ch 73)",
    "Power Plant General (Ch 71)", "Fuel System (6.3.15)",
}
SAFETY_SYSTEMS_PROSE = {
    "Landing Gear", "Fuel System", "Engine", "Electrical", "Electrical Power",
    "Exhaust System", "Propeller", "Powerplant", "Wheel & Brake",
}
SAFETY_SYSTEMS = SAFETY_SYSTEMS_TABLE | SAFETY_SYSTEMS_PROSE

# Systems unlikely to have ingested documentation (for variant #3 short-circuit)
OBSCURE_SYSTEMS_PROSE = {
    "Placards & Markings", "Leveling & Weighing", "Towing & Taxiing",
    "Parking & Mooring", "Lifting & Shoring", "Stabilizers",
    "Miscellaneous", "Manual Format & Guidance",
}

# 2×2 target allocation for 23 dataset queries
CELL_TARGETS = {
    ("table", "part_replacement"): 6,
    ("table", "operational_anomaly"): 6,
    ("prose", "part_replacement"): 5,
    ("prose", "operational_anomaly"): 6,
}

SEED = 42


def is_safety_system(section_system: str) -> bool:
    return section_system in SAFETY_SYSTEMS


def select_queries(dataset: list[dict]) -> list[dict]:
    rng = random.Random(SEED)

    # Filter to target observation types
    pool = [e for e in dataset if e["observation_type"] in ("part_replacement", "operational_anomaly")]

    # Build 2×2 pools
    cells: dict[tuple[str, str], list[dict]] = {}
    for entry in pool:
        key = (entry["origin"], entry["observation_type"])
        cells.setdefault(key, []).append(entry)

    # Shuffle each pool
    for entries in cells.values():
        rng.shuffle(entries)

    selected: list[dict] = []
    used_ids: set[str] = set()
    system_counts: Counter[str] = Counter()

    # Step 1: Pick 2 variant #3 candidates from obscure prose+operational_anomaly systems
    variant3_candidates = [
        e for e in cells.get(("prose", "operational_anomaly"), [])
        if e["section_system"] in OBSCURE_SYSTEMS_PROSE
    ]
    variant3_picked: list[dict] = []
    for entry in variant3_candidates:
        if len(variant3_picked) >= 2:
            break
        ss = entry["section_system"]
        if system_counts[ss] < 4:
            variant3_picked.append(entry)
            used_ids.add(entry["id"])
            system_counts[ss] += 1
    # If we couldn't get 2 from prose, try table obscure systems
    if len(variant3_picked) < 2:
        obscure_table = {
            "Lighting (6.3.19)", "Navigation / Attitude and Direction (Ch 34)",
        }
        table_oa = cells.get(("table", "operational_anomaly"), [])
        for entry in table_oa:
            if len(variant3_picked) >= 2:
                break
            if entry["id"] not in used_ids and entry["section_system"] in obscure_table:
                variant3_picked.append(entry)
                used_ids.add(entry["id"])
                system_counts[entry["section_system"]] += 1

    selected.extend(variant3_picked)

    # Step 2: Fill remaining dataset slots (23 - variant3 count)
    remaining_target = 23 - len(variant3_picked)

    # Adjust cell targets: subtract variant3 picks from their cells
    adjusted_targets = dict(CELL_TARGETS)
    for entry in variant3_picked:
        key = (entry["origin"], entry["observation_type"])
        adjusted_targets[key] = max(0, adjusted_targets[key] - 1)

    # Greedy selection per cell, prioritizing section_system diversity
    for cell_key in [
        ("table", "part_replacement"),
        ("table", "operational_anomaly"),
        ("prose", "part_replacement"),
        ("prose", "operational_anomaly"),
    ]:
        target = adjusted_targets[cell_key]
        candidates = [e for e in cells.get(cell_key, []) if e["id"] not in used_ids]

        # Sort by section_system count (prefer underrepresented systems)
        cell_selected: list[dict] = []
        while len(cell_selected) < target and candidates:
            # Re-sort candidates each round
            candidates.sort(key=lambda e: (system_counts[e["section_system"]], rng.random()))
            for entry in candidates:
                ss = entry["section_system"]
                if system_counts[ss] < 4:
                    cell_selected.append(entry)
                    used_ids.add(entry["id"])
                    system_counts[ss] += 1
                    candidates.remove(entry)
                    break
            else:
                # All remaining candidates have systems at cap; just take the first
                entry = candidates.pop(0)
                cell_selected.append(entry)
                used_ids.add(entry["id"])
                system_counts[entry["section_system"]] += 1

        selected.extend(cell_selected)

    # Build output entries
    eval_queries: list[dict] = []
    for i, entry in enumerate(selected, 1):
        obs_type = entry["observation_type"]
        is_variant3 = entry in variant3_picked
        eval_queries.append({
            "query_id": f"EQ-{i:03d}",
            "query_text": entry["observation"],
            "device_id": DEVICE_MAP[entry["manual_source"]],
            "origin": entry["origin"],
            "observation_type": obs_type,
            "section_system": entry["section_system"],
            "source_index": entry["id"],
            "expected_intent": "replace_part" if obs_type == "part_replacement" else "troubleshoot",
            "expected_trace_variant": 3 if is_variant3 else (5 if obs_type == "part_replacement" else 4),
            "safety_expected": is_safety_system(entry["section_system"]),
            "notes": build_note(entry, is_variant3),
        })

    # Add 7 manual placeholder slots
    manual_slots = [
        {
            "query_id": "EQ-024",
            "query_text": "PLACEHOLDER",
            "device_id": "",
            "origin": "manual",
            "observation_type": "n/a",
            "section_system": "n/a",
            "source_index": "MANUAL-DRC-1",
            "expected_intent": "troubleshoot",
            "expected_trace_variant": 1,
            "safety_expected": False,
            "notes": "DRC early-exit — requires active+approved DeterministicRule in DB",
        },
        {
            "query_id": "EQ-025",
            "query_text": "PLACEHOLDER",
            "device_id": "",
            "origin": "manual",
            "observation_type": "n/a",
            "section_system": "n/a",
            "source_index": "MANUAL-DRC-2",
            "expected_intent": "replace_part",
            "expected_trace_variant": 1,
            "safety_expected": False,
            "notes": "DRC early-exit — requires active+approved DeterministicRule in DB",
        },
        {
            "query_id": "EQ-026",
            "query_text": "PLACEHOLDER",
            "device_id": "",
            "origin": "manual",
            "observation_type": "n/a",
            "section_system": "n/a",
            "source_index": "MANUAL-FOLLOWUP-1",
            "expected_intent": "followup",
            "expected_trace_variant": 2,
            "safety_expected": False,
            "notes": "Follow-up — reference thread_id from EQ-001. Run after EQ-001 completes.",
            "parent_query_id": "EQ-001",
        },
        {
            "query_id": "EQ-027",
            "query_text": "PLACEHOLDER",
            "device_id": "",
            "origin": "manual",
            "observation_type": "n/a",
            "section_system": "n/a",
            "source_index": "MANUAL-FOLLOWUP-2",
            "expected_intent": "followup",
            "expected_trace_variant": 2,
            "safety_expected": False,
            "notes": "Follow-up — reference thread_id from EQ-005. Run after EQ-005 completes.",
            "parent_query_id": "EQ-005",
        },
        {
            "query_id": "EQ-028",
            "query_text": "PLACEHOLDER",
            "device_id": "",
            "origin": "manual",
            "observation_type": "n/a",
            "section_system": "n/a",
            "source_index": "MANUAL-FOLLOWUP-3",
            "expected_intent": "followup",
            "expected_trace_variant": 2,
            "safety_expected": False,
            "notes": "Follow-up — reference thread_id from EQ-010. Run after EQ-010 completes.",
            "parent_query_id": "EQ-010",
        },
        {
            "query_id": "EQ-029",
            "query_text": "PLACEHOLDER",
            "device_id": "",
            "origin": "manual",
            "observation_type": "n/a",
            "section_system": "n/a",
            "source_index": "MANUAL-UNSAFE-1",
            "expected_intent": "unsafe_method",
            "expected_trace_variant": 6,
            "safety_expected": True,
            "notes": "UnsafeMethodGate → troubleshoot path. Craft query describing dangerous bypass procedure.",
        },
        {
            "query_id": "EQ-030",
            "query_text": "PLACEHOLDER",
            "device_id": "",
            "origin": "manual",
            "observation_type": "n/a",
            "section_system": "n/a",
            "source_index": "MANUAL-UNSAFE-2",
            "expected_intent": "unsafe_method",
            "expected_trace_variant": 7,
            "safety_expected": True,
            "notes": "UnsafeMethodGate → replace_part path. Craft query about unsafe part swap.",
        },
    ]
    eval_queries.extend(manual_slots)

    return eval_queries


def build_note(entry: dict, is_variant3: bool) -> str:
    ss = entry["section_system"]
    obs = entry["observation_type"]
    parts: list[str] = []
    if is_variant3:
        parts.append(f"Variant #3 — obscure system ({ss}), expecting troubleshoot short-circuit")
    elif obs == "part_replacement":
        parts.append(f"Full replace_part path — {ss}")
    else:
        parts.append(f"Full troubleshoot path — {ss}")
    if is_safety_system(ss):
        parts.append("should retrieve safety procedures")
    return "; ".join(parts)


def write_coverage_report(queries: list[dict]) -> str:
    dataset_qs = [q for q in queries if q["query_text"] != "PLACEHOLDER"]
    manual_qs = [q for q in queries if q["query_text"] == "PLACEHOLDER"]

    lines: list[str] = []
    a = lines.append

    a("# Evaluation Queries Coverage Report\n")
    a(f"**Total queries**: {len(queries)} ({len(dataset_qs)} dataset + {len(manual_qs)} manual)\n")

    # Origin × observation_type matrix (dataset only)
    a("## ORIGIN × OBSERVATION_TYPE (dataset queries)\n")
    a("```")
    origin_obs = Counter((q["origin"], q["observation_type"]) for q in dataset_qs)
    a(f"{'':20s} part_replacement  operational_anomaly  TOTAL")
    for origin in ["table", "prose"]:
        pr = origin_obs.get((origin, "part_replacement"), 0)
        oa = origin_obs.get((origin, "operational_anomaly"), 0)
        a(f"{origin:20s} {pr:>16d}  {oa:>19d}  {pr+oa:>5d}")
    pr_total = sum(1 for q in dataset_qs if q["observation_type"] == "part_replacement")
    oa_total = sum(1 for q in dataset_qs if q["observation_type"] == "operational_anomaly")
    a(f"{'TOTAL':20s} {pr_total:>16d}  {oa_total:>19d}  {pr_total+oa_total:>5d}")
    a("```\n")

    # Trace variant coverage
    a("## TRACE VARIANT COVERAGE\n")
    a("```")
    variant_counts = Counter(q["expected_trace_variant"] for q in queries)
    variant_labels = {
        1: ("DRC early-exit", 2),
        2: ("Follow-up", 3),
        3: ("Troubleshoot short", 2),
        4: ("Full troubleshoot", 8),
        5: ("Full replace_part", 8),
        6: ("UnsafeMethodGate→troubleshoot", 1),
        7: ("UnsafeMethodGate→replace_part", 1),
    }
    total_v = 0
    for v in sorted(variant_labels):
        label, minimum = variant_labels[v]
        count = variant_counts.get(v, 0)
        status = "OK" if count >= minimum else "BELOW MIN"
        a(f"#{v}  {label:35s} {count:>3d}  (min {minimum}) {status}")
        total_v += count
    a(f"{'':39s} Total: {total_v}")
    a("```\n")

    # Section system distribution
    a("## SECTION_SYSTEM DISTRIBUTION (dataset queries)\n")
    a("```")
    ss_counts = Counter(q["section_system"] for q in dataset_qs)
    for ss, count in ss_counts.most_common():
        cap_status = " *** OVER CAP" if count > 4 else ""
        a(f"  {ss}: {count}{cap_status}")
    a(f"\nUnique systems: {len(ss_counts)} (target: as many as possible, max 4 per system)")
    a("```\n")

    # Safety coverage
    a("## SAFETY COVERAGE\n")
    a("```")
    safety_true = sum(1 for q in queries if q["safety_expected"])
    safety_false = sum(1 for q in queries if not q["safety_expected"])
    a(f"Queries with safety_expected=true:  {safety_true} (min 6)")
    a(f"Queries with safety_expected=false: {safety_false}")
    status = "OK" if safety_true >= 6 else "BELOW MIN"
    a(f"Status: {status}")
    a("```\n")

    # Manually crafted queries needed
    a("## MANUALLY CRAFTED QUERIES NEEDED\n")
    a("```")
    drc_count = sum(1 for q in manual_qs if "DRC" in q.get("source_index", ""))
    fu_count = sum(1 for q in manual_qs if "FOLLOWUP" in q.get("source_index", ""))
    um_count = sum(1 for q in manual_qs if "UNSAFE" in q.get("source_index", ""))
    a(f"DRC early-exit queries:       {drc_count}")
    a(f"Follow-up queries:            {fu_count}")
    a(f"UnsafeMethodGate queries:     {um_count}")
    a(f"Total manual queries needed:  {drc_count + fu_count + um_count}")
    a("```\n")

    # Full query list
    a("## QUERY LIST\n")
    a("| query_id | origin | obs_type | section_system | variant | safety | source_index |")
    a("|----------|--------|----------|----------------|---------|--------|--------------|")
    for q in queries:
        origin = q["origin"][:6]
        obs = q["observation_type"][:18]
        ss = q["section_system"][:25]
        a(f"| {q['query_id']} | {origin} | {obs} | {ss} | #{q['expected_trace_variant']} | {'Y' if q['safety_expected'] else 'N'} | {q['source_index']} |")

    return "\n".join(lines)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Select eval queries from combined dataset")
    parser.add_argument("--force", action="store_true", help="Overwrite existing eval_queries.json")
    args = parser.parse_args()

    if QUERIES_PATH.exists() and not args.force:
        print(f"{QUERIES_PATH} already exists. Use --force to regenerate (will overwrite manual edits).")
        print("Regenerating coverage report from existing file instead.\n")
        with open(QUERIES_PATH) as f:
            queries = json.load(f)
        coverage = write_coverage_report(queries)
        with open(COVERAGE_PATH, "w") as f:
            f.write(coverage)
        print(f"Wrote {COVERAGE_PATH}")
        print("\n" + coverage)
        return

    with open(DATASET_PATH) as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} entries from combined_eval_dataset.json")

    queries = select_queries(dataset)
    print(f"Selected {len(queries)} queries ({sum(1 for q in queries if q['query_text'] != 'PLACEHOLDER')} dataset + {sum(1 for q in queries if q['query_text'] == 'PLACEHOLDER')} manual)")

    with open(QUERIES_PATH, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Wrote {QUERIES_PATH}")

    coverage = write_coverage_report(queries)
    with open(COVERAGE_PATH, "w") as f:
        f.write(coverage)
    print(f"Wrote {COVERAGE_PATH}")

    # Print coverage to stdout
    print("\n" + coverage)


if __name__ == "__main__":
    main()
