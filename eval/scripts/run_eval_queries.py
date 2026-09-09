"""
Run evaluation queries through the diagnostic pipeline sequentially.

Reads a query corpus JSON, skips PLACEHOLDER entries, handles follow-up
sequencing (runs parent queries first, then follow-ups using captured
thread_ids).

Supports phased execution (--phase 1/2/all) and resume on crash (--resume).

Usage:
    python -m eval.scripts.run_eval_queries \
        --api-url http://localhost:8000 \
        --queries-file eval/generation/eval_queries_v2.json \
        --output eval/results/eval_run_v2.json \
        --phase all --delay 2

    # Phase 2 standalone (after Phase 1):
    python -m eval.scripts.run_eval_queries \
        --phase 2 \
        --parent-threads eval/results/eval_run_v2_phase1.json \
        --output eval/results/eval_run_v2_phase2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def stream_query(
    api_url: str,
    device_id: str,
    message: str,
    thread_id: str | None = None,
) -> dict:
    """Stream a diagnostic query via SSE and capture key fields."""
    import httpx

    payload: dict = {"device_id": device_id, "message": message}
    if thread_id:
        payload["thread_id"] = thread_id

    result = {
        "thread_id": None,
        "response_id": None,
        "trace_id": None,
        "inline_scores": {},
        "badge": None,
        "nodes_seen": [],
        "error": None,
    }

    node_keys = {
        "IntentRouter", "UnsafeMethodGate", "DeterministicRuleChecker",
        "FollowUpResponder", "SymptomAnalyzer", "DirectReplacementAnalyzer",
        "KnowledgeRetriever", "RootCauseAnalyzer", "SafetyExtractor",
        "RepairPlanner", "SafetyJudgeGate", "InlineEvaluator",
    }

    try:
        with httpx.stream(
            "POST",
            f"{api_url}/api/v1/diagnose/stream",
            json=payload,
            timeout=360.0,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                if event.get("event") == "THREAD_CREATED":
                    result["thread_id"] = event.get("thread_id")

                if event.get("event") == "AGENT_DONE":
                    result["trace_id"] = event.get("trace_id")

                if "InlineEvaluator" in event:
                    ie = event["InlineEvaluator"]
                    inline_eval = ie.get("inline_eval", {})
                    result["response_id"] = inline_eval.get("response_id")
                    result["inline_scores"] = inline_eval.get("scores", {})
                    result["badge"] = inline_eval.get("badge")

                for key in node_keys:
                    if key in event and key not in result["nodes_seen"]:
                        result["nodes_seen"].append(key)

                if event.get("event") == "ERROR":
                    result["error"] = event.get("message", "Unknown error")

    except Exception as e:
        result["error"] = str(e)

    return result


def poll_shadow_eval(
    api_url: str, response_id: str, timeout_s: int = 60
) -> dict | None:
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = httpx.get(
                f"{api_url}/api/v1/evaluation/shadow/{response_id}",
                timeout=30.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("completed"):
                    return data
        except Exception:
            pass
        time.sleep(3)
    return None


def fetch_trace(api_url: str, response_id: str) -> dict | None:
    import httpx

    try:
        resp = httpx.get(
            f"{api_url}/api/v1/trace/{response_id}",
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def run_single_query(
    api_url: str,
    query: dict,
    thread_id_override: str | None = None,
) -> dict:
    """Run a single query and return structured result."""
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.time()

    device_id = query["device_id"]
    message = query["query_text"]
    tid = thread_id_override or query.get("thread_id")

    stream_result = stream_query(api_url, device_id, message, tid)
    stream_ms = int((time.time() - start) * 1000)

    # Poll shadow eval
    shadow_completed = False
    if stream_result["response_id"]:
        shadow = poll_shadow_eval(api_url, stream_result["response_id"])
        shadow_completed = shadow is not None and shadow.get("completed", False)

    # Fetch trace for variant/intent
    actual_intent = None
    actual_trace_variant = None
    chunks_retrieved = 0
    documents_hit = 0
    safety_found = False

    if stream_result["response_id"]:
        trace = fetch_trace(api_url, stream_result["response_id"])
        if trace:
            ir = trace.get("intent_routing", {})
            actual_intent = ir.get("intent")
            tc = trace.get("trace_conformance", {})
            actual_trace_variant = tc.get("variant_label")
            retrieval = trace.get("retrieval_pipeline", {})
            reranked = retrieval.get("reranked_chunks", [])
            chunks_retrieved = len(reranked)
            doc_ids = {c.get("document_id") for c in reranked if c.get("document_id")}
            documents_hit = len(doc_ids)
            reasoning = trace.get("agent_reasoning", [])
            for entry in reasoning:
                if entry.get("node_name") == "SafetyExtractor":
                    safety_found = True
                    break

    completed_at = datetime.now(timezone.utc).isoformat()
    total_ms = int((time.time() - start) * 1000)

    status = "PASS" if stream_result["error"] is None and stream_result["response_id"] else "FAIL"

    return {
        "query_id": query["query_id"],
        "response_id": stream_result["response_id"],
        "thread_id": stream_result["thread_id"],
        "trace_id": stream_result["trace_id"],
        "actual_intent": actual_intent,
        "expected_intent": query["expected_intent"],
        "actual_trace_variant": actual_trace_variant,
        "expected_trace_variant": query["expected_trace_variant"],
        "sync_faithfulness": stream_result["inline_scores"].get("faithfulness"),
        "sync_answer_relevance": stream_result["inline_scores"].get("answer_relevance"),
        "badge": stream_result["badge"],
        "shadow_completed": shadow_completed,
        "chunks_retrieved": chunks_retrieved,
        "documents_hit": documents_hit,
        "safety_found": safety_found,
        "nodes_seen": stream_result["nodes_seen"],
        "started_at": started_at,
        "completed_at": completed_at,
        "stream_ms": stream_ms,
        "duration_ms": total_ms,
        "status": status,
        "error": stream_result["error"],
    }


def _run_phase(
    queries: list[dict],
    phase_label: str,
    api_url: str,
    delay: float,
    thread_id_map: dict[str, str],
    completed_ids: set[str],
    is_followup: bool,
) -> list[dict]:
    """Execute a list of queries, returning results and updating thread_id_map."""
    results: list[dict] = []
    total = len(queries)
    skipped = 0
    print(f"\n=== {phase_label}: {total} queries ===\n")

    for i, query in enumerate(queries, 1):
        qid = query["query_id"]

        # Resume: skip already-completed queries
        if qid in completed_ids:
            skipped += 1
            continue

        # Follow-up: resolve parent thread_id
        parent_thread_id = None
        if is_followup:
            parent_id = query.get("parent_query_id")
            parent_thread_id = thread_id_map.get(parent_id) if parent_id else None
            if parent_id and not parent_thread_id:
                print(f"[{i}/{total}] {qid}: SKIP — parent {parent_id} has no thread_id")
                results.append({
                    "query_id": qid,
                    "status": "FAIL",
                    "error": f"Parent query {parent_id} has no thread_id",
                    "expected_intent": query["expected_intent"],
                    "expected_trace_variant": query["expected_trace_variant"],
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": 0,
                })
                continue

        print(f"[{i}/{total}] {qid}: {query['query_text'][:60]}...", end=" ", flush=True)
        result = run_single_query(api_url, query, thread_id_override=parent_thread_id)
        results.append(result)
        print(f"{result['status']} ({result['stream_ms']}ms, badge={result['badge']})")

        # Store thread_id for follow-up resolution
        if result["thread_id"]:
            thread_id_map[qid] = result["thread_id"]

        if i < total:
            time.sleep(delay)

    if skipped:
        print(f"  (Resumed: skipped {skipped} already-completed queries)")

    return results


def _print_summary(results: list[dict]) -> None:
    """Print evaluation run summary."""
    from collections import Counter

    print("\n" + "=" * 70)
    print("EVALUATION RUN SUMMARY")
    print("=" * 70)

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    total_stream_ms = sum(r.get("stream_ms", 0) for r in results)
    total_duration_ms = sum(r.get("duration_ms", 0) for r in results)

    print(f"\n  Total:      {len(results)} queries")
    print(f"  Passed:     {pass_count}")
    print(f"  Failed:     {fail_count}")
    print(f"  Stream time:  {total_stream_ms}ms ({total_stream_ms / 1000:.0f}s)")
    print(f"  Total time:   {total_duration_ms}ms ({total_duration_ms / 1000 / 60:.1f}min)")

    # Intent match
    intent_match = sum(1 for r in results if r.get("actual_intent") == r.get("expected_intent"))
    print(f"\n  Intent match:  {intent_match}/{len(results)}")

    # Badge distribution
    badges = Counter(r.get("badge") for r in results if r.get("badge"))
    print(f"  Badge distribution: {dict(badges)}")

    # Shadow eval completion rate
    shadow_done = sum(1 for r in results if r.get("shadow_completed"))
    shadow_eligible = sum(1 for r in results if r.get("response_id"))
    print(f"  Shadow eval completed: {shadow_done}/{shadow_eligible}")

    # Per-query table
    print(f"\n{'query_id':10s} {'status':6s} {'badge':6s} {'faith':6s} {'relev':6s} {'chunks':6s} {'intent_match':12s} {'ms':>6s}")
    print("-" * 65)
    for r in results:
        qid = r.get("query_id", "?")
        status = r.get("status", "?")
        badge = str(r.get("badge", "-"))
        faith = f"{r['sync_faithfulness']:.2f}" if r.get("sync_faithfulness") is not None else "-"
        relev = f"{r['sync_answer_relevance']:.2f}" if r.get("sync_answer_relevance") is not None else "-"
        chunks = str(r.get("chunks_retrieved", "-"))
        intent_ok = "Y" if r.get("actual_intent") == r.get("expected_intent") else "N"
        ms = str(r.get("stream_ms", "-"))
        print(f"{qid:10s} {status:6s} {badge:6s} {faith:>6s} {relev:>6s} {chunks:>6s} {intent_ok:>12s} {ms:>6s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval queries through the pipeline")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=Path(__file__).parent / "eval_queries.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "eval_run_log.json",
    )
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between queries")
    parser.add_argument(
        "--phase",
        choices=["1", "2", "all"],
        default="all",
        help="Run phase 1 (initial), phase 2 (followup), or both",
    )
    parser.add_argument(
        "--parent-threads",
        type=Path,
        default=None,
        help="Phase 1 output file for thread_id resolution in standalone phase 2",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip query_ids already present in the output file",
    )
    args = parser.parse_args()

    with open(args.queries_file) as f:
        all_queries = json.load(f)

    # Separate into initial and follow-up queries based on parent_query_id
    initial_queries = [q for q in all_queries if q["query_text"] != "PLACEHOLDER" and not q.get("parent_query_id")]
    followup_queries = [q for q in all_queries if q["query_text"] != "PLACEHOLDER" and q.get("parent_query_id")]
    placeholder_count = sum(1 for q in all_queries if q["query_text"] == "PLACEHOLDER")

    print(f"Loaded {len(all_queries)} queries: {len(initial_queries)} initial, "
          f"{len(followup_queries)} follow-up, {placeholder_count} placeholders (skipped)")

    # Determine which phases to run
    run_phase1 = args.phase in ("1", "all")
    run_phase2 = args.phase in ("2", "all")

    if not run_phase1 and not run_phase2:
        print("Nothing to run.")
        sys.exit(1)

    # Resume: load already-completed query_ids from output file
    completed_ids: set[str] = set()
    existing_results: list[dict] = []
    if args.resume and args.output.exists():
        with open(args.output) as f:
            existing_results = json.load(f)
        completed_ids = {r["query_id"] for r in existing_results if r.get("status") == "PASS"}
        print(f"Resume: {len(completed_ids)} completed queries found in {args.output}")

    # Build thread_id_map from parent-threads file (for standalone phase 2)
    # or from existing results (for resume)
    thread_id_map: dict[str, str] = {}
    if args.parent_threads and args.parent_threads.exists():
        with open(args.parent_threads) as f:
            parent_results = json.load(f)
        for r in parent_results:
            if r.get("thread_id") and r.get("status") == "PASS":
                thread_id_map[r["query_id"]] = r["thread_id"]
        print(f"Loaded {len(thread_id_map)} parent thread_ids from {args.parent_threads}")
    elif existing_results:
        for r in existing_results:
            if r.get("thread_id"):
                thread_id_map[r["query_id"]] = r["thread_id"]

    results: list[dict] = list(existing_results) if args.resume else []

    # Phase 1: initial queries
    if run_phase1:
        phase1_results = _run_phase(
            initial_queries, "Phase 1 (initial)", args.api_url, args.delay,
            thread_id_map, completed_ids, is_followup=False,
        )
        results.extend(phase1_results)

        # Incremental save after Phase 1
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nPhase 1 results saved to {args.output}")

    # Phase 2: follow-up queries
    if run_phase2:
        if not followup_queries:
            print("\nNo follow-up queries to run.")
        else:
            phase2_results = _run_phase(
                followup_queries, "Phase 2 (follow-up)", args.api_url, args.delay,
                thread_id_map, completed_ids, is_followup=True,
            )
            results.extend(phase2_results)

    # Save final results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    _print_summary(results)


if __name__ == "__main__":
    main()
