"""Tests for badge decoupling: quality badge, safety badge, composite badge.

Validates the C3 architectural isolation requirement — quality and safety
badges have zero shared computation paths.

Uses the `client` fixture from conftest to ensure proper module init.
"""


# --- Thresholds fixture matching production defaults ---

THRESHOLDS = {
    "green": {"faithfulness": 0.7, "answer_relevance": 0.7},
    "red": {"faithfulness": 0.4, "answer_relevance": 0.4},
    "max_unfaithful_step_pct": 0.3,
    "safety": {
        "green": {"confidence": 0.5},
        "red": {"confidence": 0.2},
    },
}


def _import():
    from pipeline.eval_logic import (
        compute_quality_badge,
        compute_safety_badge,
        composite_badge,
    )

    return compute_quality_badge, compute_safety_badge, composite_badge


# ===========================================================================
# compute_quality_badge
# ===========================================================================


class TestComputeQualityBadge:
    def test_green_when_all_metrics_above_green_threshold(self, client):
        compute_quality_badge, _, _ = _import()
        scores = {"faithfulness": 0.8, "answer_relevance": 0.9}
        assert compute_quality_badge(scores, THRESHOLDS) == "green"

    def test_yellow_when_metric_between_red_and_green(self, client):
        compute_quality_badge, _, _ = _import()
        scores = {"faithfulness": 0.5, "answer_relevance": 0.8}
        assert compute_quality_badge(scores, THRESHOLDS) == "yellow"

    def test_red_when_metric_below_red_threshold(self, client):
        compute_quality_badge, _, _ = _import()
        scores = {"faithfulness": 0.3, "answer_relevance": 0.8}
        assert compute_quality_badge(scores, THRESHOLDS) == "red"

    def test_gray_when_metric_is_none(self, client):
        compute_quality_badge, _, _ = _import()
        scores = {"faithfulness": None, "answer_relevance": 0.8}
        assert compute_quality_badge(scores, THRESHOLDS) == "gray"

    def test_step_verdicts_unfaithful_degrades_green_to_yellow(self, client):
        compute_quality_badge, _, _ = _import()
        scores = {"faithfulness": 0.9, "answer_relevance": 0.9}
        # 1/4 = 25% < 30% threshold → yellow (not red)
        verdicts = [
            {"step_id": "s-0", "faithful": True},
            {"step_id": "s-1", "faithful": False},
            {"step_id": "s-2", "faithful": True},
            {"step_id": "s-3", "faithful": True},
        ]
        assert (
            compute_quality_badge(scores, THRESHOLDS, step_verdicts=verdicts)
            == "yellow"
        )

    def test_step_verdicts_many_unfaithful_makes_red(self, client):
        compute_quality_badge, _, _ = _import()
        scores = {"faithfulness": 0.9, "answer_relevance": 0.9}
        verdicts = [
            {"step_id": "s-0", "faithful": False},
            {"step_id": "s-1", "faithful": False},
            {"step_id": "s-2", "faithful": True},
        ]
        assert (
            compute_quality_badge(scores, THRESHOLDS, step_verdicts=verdicts) == "red"
        )


# ===========================================================================
# compute_safety_badge
# ===========================================================================


class TestComputeSafetyBadge:
    def test_none_when_no_safety_protocols(self, client):
        _, compute_safety_badge, _ = _import()
        assert compute_safety_badge(None, THRESHOLDS) is None

    def test_none_when_legacy_list(self, client):
        """Legacy sessions store safety_protocols as list[str] — must not crash."""
        _, compute_safety_badge, _ = _import()
        assert compute_safety_badge(["some legacy doc"], THRESHOLDS) is None

    def test_none_when_no_basis(self, client):
        _, compute_safety_badge, _ = _import()
        protocols = {"rules": [], "basis": [], "confidence": 0.0}
        assert compute_safety_badge(protocols, THRESHOLDS) is None

    def test_red_when_chunks_but_no_rules(self, client):
        _, compute_safety_badge, _ = _import()
        protocols = {
            "rules": [],
            "basis": [{"chunk_id": "c1", "score": 0.6}],
            "confidence": 0.6,
        }
        assert compute_safety_badge(protocols, THRESHOLDS) == "red"

    def test_green_when_rules_and_high_confidence(self, client):
        _, compute_safety_badge, _ = _import()
        protocols = {
            "rules": [
                {"text": "Wear PPE", "severity": "WARNING", "source_chunk_id": "c1"},
            ],
            "basis": [{"chunk_id": "c1", "score": 0.7}],
            "confidence": 0.7,
        }
        assert compute_safety_badge(protocols, THRESHOLDS) == "green"

    def test_yellow_when_rules_and_medium_confidence(self, client):
        _, compute_safety_badge, _ = _import()
        protocols = {
            "rules": [
                {"text": "Wear PPE", "severity": "WARNING", "source_chunk_id": "c1"},
            ],
            "basis": [{"chunk_id": "c1", "score": 0.35}],
            "confidence": 0.35,
        }
        assert compute_safety_badge(protocols, THRESHOLDS) == "yellow"

    def test_red_when_rules_and_very_low_confidence(self, client):
        _, compute_safety_badge, _ = _import()
        protocols = {
            "rules": [
                {"text": "Wear PPE", "severity": "WARNING", "source_chunk_id": "c1"},
            ],
            "basis": [{"chunk_id": "c1", "score": 0.1}],
            "confidence": 0.1,
        }
        assert compute_safety_badge(protocols, THRESHOLDS) == "red"

    def test_yellow_when_malformed_rule(self, client):
        _, compute_safety_badge, _ = _import()
        protocols = {
            "rules": [
                {"text": "Wear PPE", "severity": None, "source_chunk_id": "c1"},
            ],
            "basis": [{"chunk_id": "c1", "score": 0.7}],
            "confidence": 0.7,
        }
        assert compute_safety_badge(protocols, THRESHOLDS) == "yellow"


# ===========================================================================
# composite_badge
# ===========================================================================


class TestCompositeBadge:
    def test_both_green(self, client):
        _, _, composite_badge = _import()
        assert composite_badge("green", "green") == "green"

    def test_quality_green_safety_red(self, client):
        _, _, composite_badge = _import()
        assert composite_badge("green", "red") == "red"

    def test_quality_red_safety_green(self, client):
        _, _, composite_badge = _import()
        assert composite_badge("red", "green") == "red"

    def test_quality_yellow_safety_red(self, client):
        _, _, composite_badge = _import()
        assert composite_badge("yellow", "red") == "red"

    def test_quality_none_safety_green(self, client):
        _, _, composite_badge = _import()
        assert composite_badge(None, "green") == "green"

    def test_both_none(self, client):
        _, _, composite_badge = _import()
        assert composite_badge(None, None) == "gray"

    def test_quality_gray_safety_green(self, client):
        _, _, composite_badge = _import()
        assert composite_badge("gray", "green") == "gray"

    def test_quality_green_safety_none(self, client):
        _, _, composite_badge = _import()
        assert composite_badge("green", None) == "green"

    def test_unrecognized_badge_clamped_to_gray(self, client):
        """Unrecognized badge values must not propagate — treated as gray/None."""
        _, _, composite_badge = _import()
        assert composite_badge("green", "unknown_typo") == "green"
        assert composite_badge("unknown_typo", "red") == "red"
        assert composite_badge("unknown_typo", None) == "gray"


# ===========================================================================
# Integration scenarios from spec
# ===========================================================================


class TestBadgeDecouplingScenarios:
    """End-to-end scenarios from the implementation spec."""

    def test_high_quality_missing_safety_items(self, client):
        """High faithfulness + answer_relevance, but safety extraction failed."""
        compute_quality_badge, compute_safety_badge, composite_badge = _import()
        scores = {"faithfulness": 0.9, "answer_relevance": 0.85}
        safety = {
            "rules": [],
            "basis": [{"chunk_id": "c1", "score": 0.6}],
            "confidence": 0.6,
        }
        quality = compute_quality_badge(scores, THRESHOLDS)
        safety_b = compute_safety_badge(safety, THRESHOLDS)
        badge = composite_badge(quality, safety_b)

        assert quality == "green"
        assert safety_b == "red"
        assert badge == "red"

    def test_low_quality_good_safety(self, client):
        """Low faithfulness but complete safety extraction."""
        compute_quality_badge, compute_safety_badge, composite_badge = _import()
        scores = {"faithfulness": 0.3, "answer_relevance": 0.8}
        safety = {
            "rules": [
                {
                    "text": "LOTO required",
                    "severity": "DANGER",
                    "source_chunk_id": "c1",
                },
            ],
            "basis": [{"chunk_id": "c1", "score": 0.8}],
            "confidence": 0.8,
        }
        quality = compute_quality_badge(scores, THRESHOLDS)
        safety_b = compute_safety_badge(safety, THRESHOLDS)
        badge = composite_badge(quality, safety_b)

        assert quality == "red"
        assert safety_b == "green"
        assert badge == "red"

    def test_backward_compat_badge_is_worst(self, client):
        """Composite badge returns the worse of the two."""
        _, _, composite_badge = _import()
        assert composite_badge("green", "yellow") == "yellow"
        assert composite_badge("yellow", "green") == "yellow"
        assert composite_badge("yellow", "red") == "red"
        assert composite_badge("gray", "red") == "gray"

    def test_non_retrieval_skips_both(self, client):
        """NON_RETRIEVAL artifact: both badges None, composite = gray."""
        _, _, composite_badge = _import()
        assert composite_badge(None, None) == "gray"


# ===========================================================================
# Review priority with dual badges
# ===========================================================================


class TestReviewPriorityDualBadge:
    def test_quality_red_returns_p0(self, client):
        from core.models import ReviewTaskPriority, ReviewTaskSource
        from modules.review_priority import compute_priority

        result = compute_priority(
            ReviewTaskSource.AUTO_EVAL,
            quality_badge="red",
        )
        assert result == ReviewTaskPriority.P0

    def test_safety_red_returns_p1(self, client):
        from core.models import ReviewTaskPriority, ReviewTaskSource
        from modules.review_priority import compute_priority

        result = compute_priority(
            ReviewTaskSource.AUTO_EVAL,
            safety_badge="red",
        )
        assert result == ReviewTaskPriority.P1

    def test_safety_red_alone_not_p0(self, client):
        """Safety badge red should NOT escalate to P0 — only P1."""
        from core.models import ReviewTaskPriority, ReviewTaskSource
        from modules.review_priority import compute_priority

        result = compute_priority(
            ReviewTaskSource.AUTO_EVAL,
            quality_badge="green",
            safety_badge="red",
        )
        assert result == ReviewTaskPriority.P1

    def test_backward_compat_badge_param_still_works(self, client):
        from core.models import ReviewTaskPriority, ReviewTaskSource
        from modules.review_priority import compute_priority

        result = compute_priority(ReviewTaskSource.AUTO_EVAL, badge="red")
        assert result == ReviewTaskPriority.P0

    def test_quality_badge_overrides_composite(self, client):
        """When quality_badge is provided, it takes precedence over badge."""
        from core.models import ReviewTaskPriority, ReviewTaskSource
        from modules.review_priority import compute_priority

        result = compute_priority(
            ReviewTaskSource.AUTO_EVAL,
            badge="red",  # composite is red
            quality_badge="green",  # but quality is green
        )
        assert result != ReviewTaskPriority.P0  # should NOT be P0
