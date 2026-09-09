"""Terminal table formatter for retrieval evaluation reports."""

from __future__ import annotations

import statistics
from typing import Any


def _fmt(val: float | None, decimals: int = 3) -> str:
    """Format a metric value, or N/A if None."""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _fmt_ms(val: float | None) -> str:
    """Format a latency value in milliseconds."""
    if val is None:
        return "N/A"
    return f"{val:.1f}"


def _fmt_p(p_value: float | None) -> str:
    """Format a p-value with significance marker."""
    if p_value is None:
        return "N/A"
    marker = "*" if p_value < 0.05 else ""
    return f"{p_value:.4f}{marker}"


def _print_table(title: str, rows: list[dict], k_values: list[int]) -> None:
    """Print a section table with metric columns per k and mode."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    # Header
    headers = ["Metric"]
    for k in k_values:
        headers.extend([f"Vec@{k}", f"BM25@{k}", f"Hyb@{k}", f"Δ(h-v)@{k}"])
    headers.append("p-value")

    col_widths = [max(12, len(h) + 1) for h in headers]
    header_line = "".join(h.rjust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        cells = [row["metric"]]
        for k in k_values:
            vec = row.get(f"vector@{k}")
            bm25 = row.get(f"bm25@{k}")
            hyb = row.get(f"hybrid@{k}")
            delta = None
            if vec is not None and hyb is not None:
                delta = hyb - vec
            cells.extend([_fmt(vec), _fmt(bm25), _fmt(hyb), _fmt(delta)])
        cells.append(_fmt_p(row.get("p_value")))
        line = "".join(c.rjust(w) for c, w in zip(cells, col_widths))
        print(line)


def _print_latency_table(latency_data: dict[str, list[float]]) -> None:
    """Print latency percentiles per search mode."""
    print(f"\n{'=' * 80}")
    print("  LATENCY (ms)")
    print(f"{'=' * 80}")

    headers = ["Mode", "p50", "p95", "p99", "mean"]
    col_widths = [15, 10, 10, 10, 10]
    header_line = "".join(h.rjust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for mode, values in sorted(latency_data.items()):
        if not values:
            continue
        s = sorted(values)
        n = len(s)
        p50 = s[n // 2]
        p95 = s[int(n * 0.95)]
        p99 = s[int(n * 0.99)]
        mean = statistics.mean(values)
        cells = [
            mode,
            _fmt_ms(p50),
            _fmt_ms(p95),
            _fmt_ms(p99),
            _fmt_ms(mean),
        ]
        line = "".join(c.rjust(w) for c, w in zip(cells, col_widths))
        print(line)


def _print_ship_decision(criteria: dict[str, dict[str, Any]]) -> None:
    """Print ship/no-ship criteria evaluation."""
    print(f"\n{'=' * 80}")
    print("  SHIP DECISION")
    print(f"{'=' * 80}")

    all_pass = True
    for name, info in criteria.items():
        passed = info["passed"]
        value = info["value"]
        threshold = info["threshold"]
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}: {_fmt(value)} (threshold: {threshold})")

    print()
    if all_pass:
        print("  >>> SHIP: All criteria met. Hybrid retrieval is ready.")
    else:
        print("  >>> NO SHIP: One or more criteria failed.")


def _print_overlap_table(overlap_data: list[dict]) -> None:
    """Print Jaccard overlap between search modes."""
    print(f"\n{'=' * 80}")
    print("  MODE OVERLAP (Jaccard)")
    print(f"{'=' * 80}")

    headers = ["Query", "Vec∩BM25", "Vec∩Hyb", "BM25∩Hyb"]
    col_widths = [30, 12, 12, 12]
    header_line = "".join(h.rjust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))

    for row in overlap_data:
        cells = [
            row["query_id"][:28],
            _fmt(row.get("vec_bm25")),
            _fmt(row.get("vec_hyb")),
            _fmt(row.get("bm25_hyb")),
        ]
        line = "".join(c.rjust(w) for c, w in zip(cells, col_widths))
        print(line)


def print_report(agg: dict[str, Any], k_values: list[int]) -> None:
    """Master report function. Prints all sections.

    Args:
        agg: Aggregated results dict with keys:
            - "all": list of metric rows for all queries
            - "by_archetype": dict[archetype, list of metric rows]
            - "by_doc_type": dict[doc_type, list of metric rows]
            - "leaf_only": list of metric rows for leaf-filtered results
            - "latency": dict[mode, list[float]] in seconds
            - "overlap": list[dict] per-query overlap data
            - "ship_criteria": dict of ship/no-ship criteria
        k_values: List of k values used in evaluation.
    """
    _print_table("ALL QUERIES", agg["all"], k_values)

    for archetype, rows in sorted(agg.get("by_archetype", {}).items()):
        _print_table(f"ARCHETYPE: {archetype.upper()}", rows, k_values)

    for doc_type, rows in sorted(agg.get("by_doc_type", {}).items()):
        _print_table(f"DOC TYPE: {doc_type.upper()}", rows, k_values)

    if agg.get("leaf_only"):
        _print_table("LEAF-ONLY (level == 0)", agg["leaf_only"], k_values)

    if agg.get("latency"):
        _print_latency_table(agg["latency"])

    if agg.get("overlap"):
        _print_overlap_table(agg["overlap"])

    if agg.get("ship_criteria"):
        _print_ship_decision(agg["ship_criteria"])
