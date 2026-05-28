"""Pydantic response schemas for the dual-loop RAGAS evaluator.

Sync schemas drive the InlineEvaluator (≤5 s, score-only signal). Async
schemas drive the ShadowEvaluator (per-claim/per-sentence evidence used by
the calibration dashboard and auto-tuner).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# --- Sync (InlineEvaluator) ---


class SyncMetricResult(BaseModel):
    score: Optional[float] = Field(
        None,
        description="0.0 to 1.0. Use null only if the input is too malformed to score.",
    )
    reason: str = Field(
        "",
        description="Plain-English justification, ≤200 characters.",
    )


class FaithfulnessSyncResult(SyncMetricResult):
    unsupported_claim_count: int = Field(
        0,
        description="Number of plan claims that could NOT be supported by the retrieved context.",
    )
    total_claim_count: int = Field(
        0, description="Total atomic claims extracted from the plan."
    )


class AnswerRelevanceSyncResult(SyncMetricResult):
    generated_questions: list[str] = Field(
        default_factory=list,
        description="Three questions this plan would answer; used to score against the user's actual question.",
    )


class ContextRelevanceSyncResult(SyncMetricResult):
    relevant_sentence_count: int = 0
    total_sentence_count: int = 0


class CompletenessSyncResult(SyncMetricResult):
    missing_facts: list[str] = Field(
        default_factory=list,
        description="Up to three key facts/steps/safety items present in context but absent from the plan.",
    )


# --- Async (ShadowEvaluator) ---
# evidence is a free-form dict so prompts can evolve without a schema migration.


class ContextRelevanceAsyncResult(SyncMetricResult):
    evidence: dict = Field(
        default_factory=dict,
        description='{"sentences": [{"text": str, "relevant": bool, "why": str}]}',
    )


class CompletenessAsyncResult(SyncMetricResult):
    evidence: dict = Field(
        default_factory=dict,
        description='{"key_facts": [{"fact": str, "present_in_plan": bool, "evidence_quote": str}]}',
    )


# --- Per-Step Faithfulness (InlineEvaluator, Group 2) ---


class StepVerdict(BaseModel):
    step_id: str = Field(description="Step identifier, e.g. 's-0', 's-1'.")
    faithful: bool = Field(
        description="True if the step is supported by at least one retrieved chunk."
    )
    reason: str = Field(description="One sentence explaining the verdict.")
    source_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Chunk IDs from the retrieved context that support this step (empty if ungrounded).",
    )
    grounding_label: Literal["verbatim", "paraphrased", "synthesized", "ungrounded"] = (
        Field(
            description="Grounding relationship between step and source context.",
        )
    )


class PerStepFaithfulnessResult(BaseModel):
    steps: list[StepVerdict] = Field(description="Per-step faithfulness verdicts.")


# --- Unified Inline Evaluation (single-call Flash-Lite) ---


class UnifiedInlineEvalResult(BaseModel):
    """Single-call schema combining per-step faithfulness and answer relevance."""

    steps: list[StepVerdict] = Field(
        default_factory=list,
        description="Per-step faithfulness verdicts for all repair steps.",
    )
    answer_relevance_score: Optional[float] = Field(
        None,
        description="0.0 to 1.0 — how well the plan addresses the user's question.",
    )
    answer_relevance_reason: str = Field(
        "",
        description="Plain-English justification for the answer relevance score, ≤200 characters.",
    )


