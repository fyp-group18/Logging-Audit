"""
Run all evaluation queries through the diagnostic pipeline sequentially.

Reads eval_queries.json, skips PLACEHOLDER entries, handles follow-up sequencing
(runs parent queries first, then follow-ups using captured thread_ids).

Outputs eval_run_log.json with per-query results.

Usage:
    cd backend && uv run python -m eval.generation.run_eval_queries \
        --api-url http://localhost:8000 \
        --email admin@example.com \
        --password yourpassword \
        --queries-file eval/generation/eval_queries.json \
        --output eval/generation/eval_run_log.json \
        --delay 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def login(api_url: str, email: str, password: str) -> str:
    import httpx

    response = httpx.post(
        f"{api_url}/api/v1/auth/token",
        data={"username": email, "password": password},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60.0,
    )
    response.raise_for_status()
    access_token = response.cookies.get("access_token")
    if not access_token:
        body = response.json()
        access_token = body.get("access_token", "")
    if not access_token:
        print("ERROR: Could not extract access token from login response")
        sys.exit(1)
    return access_token


def stream_query(
    api_url: str,
    access_token: str,
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
            cookies={"access_token": access_token},
            timeout=180.0,
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
    api_url: str, access_token: str, response_id: str, timeout_s: int = 60
) -> dict | None:
    import httpx

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = httpx.get(
                f"{api_url}/api/v1/evaluation/shadow/{response_id}",
                cookies={"access_token": access_token},
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


def fetch_trace(api_url: str, access_token: str, response_id: str) -> dict | None:
    import httpx

    try:
        resp = httpx.get(
            f"{api_url}/api/v1/trace/{response_id}",
            cookies={"access_token": access_token},
            timeout=60.0,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def run_single_query(
    api_url: str,
    access_token: str,
    query: dict,
    thread_id_override: str | None = None,
) -> dict:
    """Run a single query and return structured result."""
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.time()

    device_id = query["device_id"]
    message = query["query_text"]
    tid = thread_id_override or query.get("thread_id")

    stream_result = stream_query(api_url, access_token, device_id, message, tid)
    stream_ms = int((time.time() - start) * 1000)

    # Poll shadow eval
    shadow_completed = False
    if stream_result["response_id"]:
        shadow = poll_shadow_eval(api_url, access_token, stream_result["response_id"])
        shadow_completed = shadow is not None and shadow.get("completed", False)

    # Fetch trace for variant/intent
    actual_intent = None
    actual_trace_variant = None
    chunks_retrieved = 0
    documents_hit = 0
    safety_found = False

    if stream_result["response_id"]:
        trace = fetch_trace(api_url, access_token, stream_result["response_id"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all eval queries through the pipeline")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
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
    args = parser.parse_args()

    with open(args.queries_file) as f:
        all_queries = json.load(f)

    # Separate into initial and follow-up queries
    initial_queries = [q for q in all_queries if q["query_text"] != "PLACEHOLDER" and q.get("expected_trace_variant") != 2]
    followup_queries = [q for q in all_queries if q["query_text"] != "PLACEHOLDER" and q.get("expected_trace_variant") == 2]
    placeholder_count = sum(1 for q in all_queries if q["query_text"] == "PLACEHOLDER")

    runnable = len(initial_queries) + len(followup_queries)
    print(f"Loaded {len(all_queries)} queries: {len(initial_queries)} initial, {len(followup_queries)} follow-up, {placeholder_count} placeholders (skipped)")

    if runnable == 0:
        print("No runnable queries found. Fill in PLACEHOLDER entries first.")
        sys.exit(1)

    print(f"Logging in to {args.api_url}...")
    token = login(args.api_url, args.email, args.password)
    print("Authenticated.\n")

    results: list[dict] = []
    thread_id_map: dict[str, str] = {}  # query_id → thread_id (for follow-up resolution)

    # Phase 1: Run initial queries
    total = len(initial_queries)
    print(f"=== Phase 1: Running {total} initial queries ===\n")
    for i, query in enumerate(initial_queries, 1):
        print(f"[{i}/{total}] {query['query_id']}: {query['query_text'][:60]}...", end=" ", flush=True)
        result = run_single_query(args.api_url, token, query)
        results.append(result)
        print(f"{result['status']} ({result['stream_ms']}ms, badge={result['badge']})")

        # Store thread_id for follow-up resolution
        if result["thread_id"]:
            thread_id_map[query["query_id"]] = result["thread_id"]

        if i < total:
            time.sleep(args.delay)

    # Phase 2: Run follow-up queries
    if followup_queries:
        print(f"\n=== Phase 2: Running {len(followup_queries)} follow-up queries ===\n")
        for i, query in enumerate(followup_queries, 1):
            parent_id = query.get("parent_query_id")
            parent_thread_id = thread_id_map.get(parent_id) if parent_id else None

            if parent_id and not parent_thread_id:
                print(f"[{i}/{len(followup_queries)}] {query['query_id']}: SKIP — parent {parent_id} has no thread_id")
                results.append({
                    "query_id": query["query_id"],
                    "status": "FAIL",
                    "error": f"Parent query {parent_id} has no thread_id",
                    "started_at": datetime.now(timezone.utc).isoformat(),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": 0,
                })
                continue

            print(f"[{i}/{len(followup_queries)}] {query['query_id']}: {query['query_text'][:60]}...", end=" ", flush=True)
            result = run_single_query(args.api_url, token, query, thread_id_override=parent_thread_id)
            results.append(result)
            print(f"{result['status']} ({result['stream_ms']}ms, badge={result['badge']})")

            if i < len(followup_queries):
                time.sleep(args.delay)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Print summary
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
    from collections import Counter
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


if __name__ == "__main__":
    main()
