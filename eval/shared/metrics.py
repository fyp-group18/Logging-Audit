"""Pure retrieval evaluation metrics. No DB or side effects."""

from __future__ import annotations

import math


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant items found in top-k retrieved results."""
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & set(relevant_ids)) / len(relevant_ids)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of top-k retrieved results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    return sum(1 for rid in top_k if rid in relevant_set) / k


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result, 0 if none found."""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved_ids: list[str], relevance_grades: dict[str, int], k: int
) -> float:
    """Normalized Discounted Cumulative Gain at k with graded relevance (0/1/2)."""
    if not relevance_grades:
        return 0.0

    # DCG of retrieved list
    dcg = 0.0
    for i, rid in enumerate(retrieved_ids[:k]):
        grade = relevance_grades.get(rid, 0)
        dcg += (2**grade - 1) / math.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: sort all grades descending, take top-k
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for i, grade in enumerate(ideal_grades):
        idcg += (2**grade - 1) / math.log2(i + 2)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Binary: 1.0 if any relevant item is in top-k, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def jaccard_overlap(set_a: set[str], set_b: set[str]) -> float:
    """Jaccard similarity between two ID sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def summary_ratio(results: list[dict]) -> float:
    """Fraction of results with level > 0 (summary/parent chunks)."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("level", 0) > 0) / len(results)
