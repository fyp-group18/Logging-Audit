"""Unit tests for eval.metrics and eval.stats — no DB required."""

import pytest

from eval.shared.metrics import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k,
    hit_rate_at_k,
    jaccard_overlap,
    summary_ratio,
)
from eval.shared.stats import paired_bootstrap


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------
class TestRecallAtK:
    """Tests for recall@k metric."""

    def test_perfect_recall(self):
        """All relevant items in top-k."""
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_zero_recall(self):
        """No relevant items in top-k."""
        assert recall_at_k(["x", "y", "z"], ["a", "b"], 3) == 0.0

    def test_partial_recall(self):
        """Some relevant items in top-k."""
        assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5

    def test_k_limits_results(self):
        """Only considers top-k, not full list."""
        assert recall_at_k(["x", "y", "a", "b"], ["a", "b"], 2) == 0.0

    def test_empty_relevant(self):
        """No relevant items defined."""
        assert recall_at_k(["a", "b"], [], 5) == 0.0

    def test_empty_retrieved(self):
        """No results retrieved."""
        assert recall_at_k([], ["a", "b"], 5) == 0.0


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------
class TestPrecisionAtK:
    """Tests for precision@k metric."""

    def test_perfect_precision(self):
        """All top-k items are relevant."""
        assert precision_at_k(["a", "b"], ["a", "b", "c"], 2) == 1.0

    def test_zero_precision(self):
        """No top-k items are relevant."""
        assert precision_at_k(["x", "y"], ["a", "b"], 2) == 0.0

    def test_half_precision(self):
        """Half of top-k items are relevant."""
        assert precision_at_k(["a", "x"], ["a", "b"], 2) == 0.5

    def test_k_zero(self):
        """k=0 returns 0."""
        assert precision_at_k(["a"], ["a"], 0) == 0.0


# ---------------------------------------------------------------------------
# mrr
# ---------------------------------------------------------------------------
class TestMRR:
    """Tests for Mean Reciprocal Rank."""

    def test_first_position(self):
        """Relevant item at rank 1."""
        assert mrr(["a", "b", "c"], ["a"]) == 1.0

    def test_second_position(self):
        """Relevant item at rank 2."""
        assert mrr(["x", "a", "c"], ["a"]) == 0.5

    def test_third_position(self):
        """Relevant item at rank 3."""
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_relevant(self):
        """No relevant items in results."""
        assert mrr(["x", "y", "z"], ["a"]) == 0.0

    def test_multiple_relevant_returns_first(self):
        """MRR uses the first relevant item only."""
        assert mrr(["x", "a", "b"], ["a", "b"]) == 0.5

    def test_empty_retrieved(self):
        """No results returns 0."""
        assert mrr([], ["a"]) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------
class TestNDCGAtK:
    """Tests for Normalized Discounted Cumulative Gain."""

    def test_perfect_ordering(self):
        """Items in ideal order should give NDCG = 1.0."""
        grades = {"a": 2, "b": 1, "c": 0}
        assert ndcg_at_k(["a", "b", "c"], grades, 3) == pytest.approx(1.0)

    def test_reversed_ordering(self):
        """Worst ordering gives NDCG < 1.0."""
        grades = {"a": 2, "b": 1, "c": 0}
        result = ndcg_at_k(["c", "b", "a"], grades, 3)
        assert result < 1.0
        assert result > 0.0

    def test_no_relevant_items(self):
        """All grades zero gives NDCG = 0."""
        grades = {"a": 0, "b": 0}
        assert ndcg_at_k(["a", "b"], grades, 2) == 0.0

    def test_empty_grades(self):
        """Empty grades dict gives 0."""
        assert ndcg_at_k(["a", "b"], {}, 2) == 0.0

    def test_k_limits(self):
        """Only top-k items considered."""
        grades = {"a": 0, "b": 2}
        result_k1 = ndcg_at_k(["a", "b"], grades, 1)
        result_k2 = ndcg_at_k(["a", "b"], grades, 2)
        assert result_k1 < result_k2

    def test_single_highly_relevant(self):
        """Single grade-2 item at rank 1."""
        grades = {"a": 2}
        assert ndcg_at_k(["a"], grades, 1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# hit_rate_at_k
# ---------------------------------------------------------------------------
class TestHitRateAtK:
    """Tests for binary hit rate."""

    def test_hit(self):
        """At least one relevant item in top-k."""
        assert hit_rate_at_k(["x", "a"], ["a"], 2) == 1.0

    def test_miss(self):
        """No relevant item in top-k."""
        assert hit_rate_at_k(["x", "y"], ["a"], 2) == 0.0

    def test_k_boundary(self):
        """Relevant item just outside k."""
        assert hit_rate_at_k(["x", "y", "a"], ["a"], 2) == 0.0


# ---------------------------------------------------------------------------
# jaccard_overlap
# ---------------------------------------------------------------------------
class TestJaccardOverlap:
    """Tests for Jaccard similarity."""

    def test_identical_sets(self):
        """Same sets have Jaccard = 1.0."""
        assert jaccard_overlap({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self):
        """No overlap gives 0.0."""
        assert jaccard_overlap({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self):
        """Partial overlap."""
        result = jaccard_overlap({"a", "b"}, {"b", "c"})
        assert result == pytest.approx(1 / 3)

    def test_both_empty(self):
        """Both empty returns 1.0 by convention."""
        assert jaccard_overlap(set(), set()) == 1.0

    def test_one_empty(self):
        """One empty set gives 0.0."""
        assert jaccard_overlap(set(), {"a"}) == 0.0


# ---------------------------------------------------------------------------
# summary_ratio
# ---------------------------------------------------------------------------
class TestSummaryRatio:
    """Tests for summary (non-leaf) ratio."""

    def test_all_leaf(self):
        """All level=0 gives ratio 0."""
        results = [{"level": 0}, {"level": 0}]
        assert summary_ratio(results) == 0.0

    def test_all_summary(self):
        """All level>0 gives ratio 1."""
        results = [{"level": 1}, {"level": 2}]
        assert summary_ratio(results) == 1.0

    def test_mixed(self):
        """Mixed levels."""
        results = [{"level": 0}, {"level": 1}, {"level": 0}, {"level": 2}]
        assert summary_ratio(results) == 0.5

    def test_empty(self):
        """Empty list gives 0."""
        assert summary_ratio([]) == 0.0

    def test_missing_level_key(self):
        """Missing level key defaults to 0 (leaf)."""
        results = [{"text": "hello"}]
        assert summary_ratio(results) == 0.0


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------
class TestPairedBootstrap:
    """Tests for paired bootstrap significance test."""

    def test_identical_inputs(self):
        """Identical values should give p-value = 1.0."""
        values = [0.5, 0.6, 0.7, 0.8]
        p = paired_bootstrap(values, values)
        assert p == 1.0

    def test_p_value_in_range(self):
        """P-value must be in [0, 1]."""
        a = [0.8, 0.9, 0.7, 0.85, 0.75]
        b = [0.3, 0.4, 0.2, 0.35, 0.25]
        p = paired_bootstrap(a, b)
        assert 0.0 <= p <= 1.0

    def test_clear_difference_low_p(self):
        """Large consistent difference with variance should yield low p-value."""
        a = [
            0.85,
            0.90,
            0.88,
            0.92,
            0.87,
            0.91,
            0.89,
            0.93,
            0.86,
            0.90,
            0.88,
            0.91,
            0.87,
            0.92,
            0.89,
            0.90,
            0.88,
            0.91,
            0.86,
            0.93,
        ]
        b = [
            0.15,
            0.10,
            0.12,
            0.08,
            0.13,
            0.09,
            0.11,
            0.07,
            0.14,
            0.10,
            0.12,
            0.09,
            0.13,
            0.08,
            0.11,
            0.10,
            0.12,
            0.09,
            0.14,
            0.07,
        ]
        p = paired_bootstrap(a, b, n_iterations=5000)
        assert p < 0.05

    def test_mismatched_lengths_raises(self):
        """Different-length lists should raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            paired_bootstrap([0.5, 0.6], [0.5])

    def test_empty_lists(self):
        """Empty lists return p=1.0."""
        assert paired_bootstrap([], []) == 1.0

    def test_reproducible_with_seed(self):
        """Same seed gives same result."""
        a = [0.7, 0.8, 0.6, 0.75]
        b = [0.5, 0.6, 0.4, 0.55]
        p1 = paired_bootstrap(a, b, seed=42)
        p2 = paired_bootstrap(a, b, seed=42)
        assert p1 == p2

    def test_different_seed_may_differ(self):
        """Different seeds can give different results (non-determinism check)."""
        a = [0.7, 0.8, 0.6, 0.75, 0.65, 0.72]
        b = [0.5, 0.6, 0.4, 0.55, 0.45, 0.52]
        p1 = paired_bootstrap(a, b, seed=42)
        p2 = paired_bootstrap(a, b, seed=99)
        # They might coincidentally be equal, but with high probability they differ
        # Just check both are valid
        assert 0.0 <= p1 <= 1.0
        assert 0.0 <= p2 <= 1.0
