#!/usr/bin/env python3
"""Human evaluation export — export evaluation data for human annotators.

Produces:
  eval/results/human_eval/evaluation_data.json          — full context (chunks, steps, execution log)
  eval/results/human_eval/evaluator1_responses.csv      — response-level template (evaluator 1)
  eval/results/human_eval/evaluator2_responses.csv      — response-level template (evaluator 2)
  eval/results/human_eval/evaluator1_steps.csv          — step-level template (evaluator 1)
  eval/results/human_eval/evaluator2_steps.csv          — step-level template (evaluator 2)

Requires:
  DATABASE_URL env var pointing to the evaluation database
  eval/results/eval_run_v2.json (150-query corpus results)
  eval/generation/eval_queries_v2.json (query text corpus)
"""

import csv
import json
import os
import sys
from pathlib import Path

import msgpack
import psycopg

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "eval" / "results" / "human_eval"
EVAL_RUN = ROOT / "eval" / "results" / "eval_run_v2.json"
QUERY_CORPUS = ROOT / "eval" / "generation" / "eval_queries_v2.json"


def load_query_corpus() -> dict[str, dict]:
    """Load eval corpus keyed by query_id."""
    with open(QUERY_CORPUS) as f:
        queries = json.load(f)
    return {q["query_id"]: q for q in queries}


def load_eval_results() -> list[dict]:
    with open(EVAL_RUN) as f:
        return json.load(f)


def filter_evaluable(results: list[dict]) -> tuple[list[dict], dict]:
    """Return (evaluable, stats) where evaluable = went through RepairPlanner."""
    evaluable = []
    gate_rejected = 0
    deterministic_exit = 0
    other_excluded = 0

    for r in results:
        nodes = r.get("nodes_seen") or []
        if "RepairPlanner" in nodes and r.get("response_id"):
            evaluable.append(r)
        elif r.get("gate_rejected") or r.get("response_id") is None:
            gate_rejected += 1
        elif (r.get("actual_trace_variant") or "").endswith("deterministic_exit"):
            deterministic_exit += 1
        else:
            other_excluded += 1

    stats = {
        "total": len(results),
        "evaluable": len(evaluable),
        "gate_rejected": gate_rejected,
        "deterministic_exit": deterministic_exit,
        "other_excluded": other_excluded,
    }
    return evaluable, stats


def _decode_checkpoint_blob(blob: bytes):
    """Decode a LangGraph checkpoint blob (msgpack)."""
    return msgpack.unpackb(blob, raw=False)


def _extract_repair_steps(final_response: dict) -> list[str]:
    """Extract step text strings from final_response checkpoint state."""
    steps_raw = final_response.get("repair_steps") or final_response.get("steps") or []
    steps = []
    for s in steps_raw:
        if isinstance(s, str):
            steps.append(s)
        elif isinstance(s, dict):
            steps.append(s.get("text", json.dumps(s, default=str)))
    return steps


def _batch_fetch_checkpoints(cur, thread_ids: list[str]) -> dict[str, dict]:
    """Batch-fetch the earliest final_response checkpoint blob per thread."""
    # Use DISTINCT ON to get earliest version per thread
    cur.execute(
        """
        SELECT DISTINCT ON (thread_id) thread_id, blob
        FROM checkpoint_blobs
        WHERE channel = 'final_response' AND thread_id = ANY(%s)
        ORDER BY thread_id, version ASC
        """,
        (thread_ids,),
    )
    results = {}
    for row in cur.fetchall():
        tid, blob = row
        try:
            results[tid] = _decode_checkpoint_blob(blob)
        except Exception as e:
            print(f"  WARNING: failed to decode checkpoint for thread {tid}: {e}")
    return results


def fetch_response_data(conn, evaluable: list[dict], corpus: dict[str, dict]) -> list[dict]:
    """Fetch response text, execution steps, and chunk references from DB."""
    cur = conn.cursor()

    # Batch-fetch checkpoint blobs for all threads
    thread_ids = list({r["thread_id"] for r in evaluable})
    print(f"  Fetching checkpoint blobs for {len(thread_ids)} threads...")
    checkpoints = _batch_fetch_checkpoints(cur, thread_ids)
    print(f"  Got {len(checkpoints)} checkpoint blobs")

    evaluation_data = []

    for i, r in enumerate(evaluable):
        response_id = r["response_id"]
        trace_id = r["trace_id"]
        thread_id = r["thread_id"]
        query_id = r["query_id"]

        # Query text from corpus file (not DB)
        query_entry = corpus.get(query_id, {})
        query_text = query_entry.get("query_text", f"MISSING: {query_id}")
        device_id = query_entry.get("device_id", "unknown")

        # Response content from LangGraph checkpoint (final_response channel)
        final_resp = checkpoints.get(thread_id)
        if final_resp and isinstance(final_resp, dict):
            repair_steps = _extract_repair_steps(final_resp)
            root_cause = final_resp.get("root_cause", "")
            follow_ups = final_resp.get("suggested_follow_ups", [])
            raw_response_text = json.dumps(final_resp, indent=2, default=str)
        else:
            repair_steps = []
            root_cause = ""
            follow_ups = []
            raw_response_text = "NO CHECKPOINT DATA FOUND"

        # Execution log (all nodes for this trace)
        cur.execute(
            """
            SELECT node_name, operation_type, latency_ms,
                   substring(metadata::text, 1, 500) AS meta_summary,
                   created_at, chain_id
            FROM agent_execution_logs
            WHERE trace_id = %s
            ORDER BY created_at ASC
            """,
            (trace_id,),
        )
        execution_steps = [
            {
                "node_name": row[0],
                "operation_type": row[1],
                "latency_ms": row[2],
                "metadata_summary": row[3],
                "created_at": str(row[4]),
                "chain_id": str(row[5]) if row[5] else None,
            }
            for row in cur.fetchall()
        ]

        # Retrieved chunks
        cur.execute(
            """
            SELECT rcl.chunk_id, dcm.text, dcm.document_id, dcm.section_header,
                   dcm.has_safety_content, rcl.step_index
            FROM response_chunk_link rcl
            JOIN document_chunks_multimodal dcm ON rcl.chunk_id = dcm.id
            WHERE rcl.response_id = %s
            """,
            (response_id,),
        )
        chunks = [
            {
                "chunk_id": str(row[0]),
                "content_preview": (row[1] or "")[:500],
                "document_id": row[2],
                "section_header": row[3],
                "has_safety_content": row[4],
                "step_index": row[5],
            }
            for row in cur.fetchall()
        ]

        # Evaluation metrics (scores, badge)
        cur.execute(
            """
            SELECT badge, sync_faithfulness, sync_answer_relevance,
                   sync_context_relevance, sync_completeness, step_verdicts
            FROM evaluation_metrics
            WHERE response_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (response_id,),
        )
        em_row = cur.fetchone()

        evaluation_data.append(
            {
                "query_id": query_id,
                "response_id": response_id,
                "trace_id": trace_id,
                "thread_id": thread_id,
                "device_id": device_id,
                "query_text": query_text,
                "root_cause": root_cause,
                "response_text": raw_response_text,
                "repair_steps": repair_steps,
                "suggested_follow_ups": follow_ups,
                "variant": r.get("actual_trace_variant") or r.get("expected_trace_variant"),
                "badge": r.get("badge"),
                "sync_faithfulness": r.get("sync_faithfulness"),
                "sync_answer_relevance": r.get("sync_answer_relevance"),
                "sync_context_relevance": em_row[3] if em_row else None,
                "sync_completeness": em_row[4] if em_row else None,
                "step_verdicts": em_row[5] if em_row else None,
                "chunks_retrieved": len(chunks),
                "safety_chunks": sum(1 for c in chunks if c["has_safety_content"]),
                "execution_steps": execution_steps,
                "retrieved_chunks": chunks,
                "safety_expected": query_entry.get("safety_expected", False),
                "notes": query_entry.get("notes", ""),
            }
        )

        if (i + 1) % 20 == 0:
            print(f"  Fetched {i + 1}/{len(evaluable)} responses...")

    return evaluation_data


def write_response_csv(evaluation_data: list[dict], filepath: Path):
    """Write response-level evaluation template CSV."""
    fieldnames = [
        "query_id",
        "response_id",
        "variant",
        "device_id",
        "query_text",
        "response_text",
        "badge",
        "sync_faithfulness",
        "chunks_retrieved",
        "safety_expected",
        "notes",
        # Evaluator fills these:
        "correctness",         # correct / partial / incorrect
        "completeness",        # complete / partial / incomplete
        "safety_assessment",   # present / partial / missing / n_a
        "overall_verdict",     # THUMBS_UP / THUMBS_DOWN
        "correction_text",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ed in evaluation_data:
            writer.writerow(
                {
                    "query_id": ed["query_id"],
                    "response_id": ed["response_id"],
                    "variant": ed["variant"],
                    "device_id": ed["device_id"],
                    "query_text": ed["query_text"],
                    "response_text": ed["response_text"][:5000],
                    "badge": ed["badge"],
                    "sync_faithfulness": ed["sync_faithfulness"],
                    "chunks_retrieved": ed["chunks_retrieved"],
                    "safety_expected": ed["safety_expected"],
                    "notes": ed["notes"],
                    "correctness": "",
                    "completeness": "",
                    "safety_assessment": "",
                    "overall_verdict": "",
                    "correction_text": "",
                }
            )


def write_step_csv(evaluation_data: list[dict], filepath: Path):
    """Write step-level evaluation template CSV."""
    fieldnames = [
        "query_id",
        "response_id",
        "step_index",
        "step_content",
        "chunk_refs",
        # Evaluator fills:
        "verdict",          # ACCEPT / SKIP / MODIFY / INCORRECT
        "correction_text",  # required if verdict is MODIFY or INCORRECT
        "severity",         # low / medium / high (optional)
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ed in evaluation_data:
            steps = ed["repair_steps"]
            if not steps:
                # Fallback: single row with full response text
                writer.writerow(
                    {
                        "query_id": ed["query_id"],
                        "response_id": ed["response_id"],
                        "step_index": 0,
                        "step_content": ed["response_text"][:2000],
                        "chunk_refs": "",
                        "verdict": "",
                        "correction_text": "",
                        "severity": "",
                    }
                )
                continue

            # Map chunks to steps by step_index
            step_chunks: dict[int, list[str]] = {}
            for c in ed["retrieved_chunks"]:
                si = c.get("step_index")
                if si is not None:
                    step_chunks.setdefault(si, []).append(c["chunk_id"])

            for idx, step in enumerate(steps):
                refs = step_chunks.get(idx, [])
                writer.writerow(
                    {
                        "query_id": ed["query_id"],
                        "response_id": ed["response_id"],
                        "step_index": idx,
                        "step_content": step[:2000],
                        "chunk_refs": ";".join(refs),
                        "verdict": "",
                        "correction_text": "",
                        "severity": "",
                    }
                )


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and filter
    print("Loading eval results and query corpus...")
    results = load_eval_results()
    corpus = load_query_corpus()
    evaluable, stats = filter_evaluable(results)

    print(f"\n=== FILTERING ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Step 2: Fetch from DB
    print(f"\nFetching response data for {len(evaluable)} evaluable responses...")
    conn = psycopg.connect(db_url)
    try:
        evaluation_data = fetch_response_data(conn, evaluable, corpus)
    finally:
        conn.close()

    # Report steps parsed
    total_steps = sum(len(ed["repair_steps"]) for ed in evaluation_data)
    no_steps = sum(1 for ed in evaluation_data if not ed["repair_steps"])
    print(f"\n=== STEP EXTRACTION ===")
    print(f"  Total repair steps: {total_steps}")
    print(f"  Avg steps/response: {total_steps / len(evaluation_data):.1f}")
    print(f"  Responses with 0 steps (metadata missing): {no_steps}")

    # Step 3: Write response-level CSVs (one per evaluator)
    for evaluator in [1, 2]:
        path = RESULTS_DIR / f"evaluator{evaluator}_responses.csv"
        write_response_csv(evaluation_data, path)
        print(f"  Written {path.relative_to(ROOT)} ({len(evaluation_data)} rows)")

    # Step 4: Write step-level CSVs (one per evaluator)
    for evaluator in [1, 2]:
        path = RESULTS_DIR / f"evaluator{evaluator}_steps.csv"
        write_step_csv(evaluation_data, path)
        print(f"  Written {path.relative_to(ROOT)} ({total_steps} step rows)")

    # Step 5: Write full JSON context
    json_path = RESULTS_DIR / "evaluation_data.json"
    with open(json_path, "w") as f:
        json.dump(evaluation_data, f, indent=2, default=str)
    print(f"  Written {json_path.relative_to(ROOT)} ({len(evaluation_data)} entries)")

    # Summary
    print(f"\n=== HUMAN EVAL EXPORT COMPLETE ===")
    print(f"  Evaluable responses exported: {len(evaluation_data)}")
    print(f"  Total steps for evaluation: {total_steps}")
    print(f"  Output directory: {RESULTS_DIR.relative_to(ROOT)}")
    print(f"\n  Evaluator deliverables:")
    print(f"    1. evaluator{{1,2}}_responses.csv — fill correctness/completeness/safety/verdict columns")
    print(f"    2. evaluator{{1,2}}_steps.csv     — fill verdict/correction_text per step")
    print(f"    3. evaluation_data.json           — full context reference (chunks, execution log)")
    print(f"\n  STOP HERE. Run human_eval_import.py after both evaluators complete.")


if __name__ == "__main__":
    main()
