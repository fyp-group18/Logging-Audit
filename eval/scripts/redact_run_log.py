#!/usr/bin/env python3
"""Produce the redistributable form of the evaluation run log.

The raw run log carries production identifiers, so it is not shipped. This
script rewrites it into eval/results/eval_run_v2_redacted.json, which is: the
identifiers replaced by stable pseudonyms, and every other field kept verbatim.

Identifiers are pseudonymised by first-appearance order rather than by hash, so
the relationships that the corpus depends on survive: a follow-up query keeps
the thread of its parent, and a session that produced no distinct response keeps
that structure too. The mapping is not published, so the pseudonyms cannot be
resolved back to the production rows.

The run log contains no response text and no SSE payloads — only per-session
routing, retrieval counts, scores and timings — so redaction is limited to the
three identifier fields.

Usage:
    python -m eval.scripts.redact_run_log \
        --input eval/results/eval_run_v2.json \
        --output eval/results/eval_run_v2_redacted.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Identifier fields to pseudonymise, and the prefix each gets.
ID_FIELDS = {
    "response_id": "resp",
    "thread_id": "thread",
    "trace_id": "trace",
}


def redact(entries: list[dict]) -> list[dict]:
    """Return a copy of *entries* with identifier fields pseudonymised."""
    mappings: dict[str, dict[str, str]] = {field: {} for field in ID_FIELDS}

    redacted = []
    for entry in entries:
        out = dict(entry)
        for field, prefix in ID_FIELDS.items():
            value = entry.get(field)
            if not value:
                continue
            mapping = mappings[field]
            if value not in mapping:
                mapping[value] = f"{prefix}-{len(mapping) + 1:04d}"
            out[field] = mapping[value]
        redacted.append(out)
    return redacted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input", type=Path, default=ROOT / "eval" / "results" / "eval_run_v2.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "eval" / "results" / "eval_run_v2_redacted.json",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} not found — the raw run log is not redistributed.")
        return 1

    with open(args.input) as f:
        entries = json.load(f)

    redacted = redact(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(redacted, f, indent=2)

    print(f"Wrote {len(redacted)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
