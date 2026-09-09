"""Post-generation alignment of safety rules to individual repair steps.

Uses a lightweight Gemini Flash call (structured output) to map each
safety rule to the step(s) it applies to.  Falls back gracefully to
``{-1: all_rules}`` when the LLM call fails, preserving the existing
flat-block rendering.
"""

from __future__ import annotations

import logging
from typing import Any

from google.genai import types
from pydantic import BaseModel, Field

from core.config import MODEL_FLASH
from pipeline.telemetry import tracked_generate_with_retry
from prompts.system_prompts import PROMPT_SAFETY_STEP_ALIGNMENT

logger = logging.getLogger(__name__)


# --- Structured output schema for the alignment LLM call ---


class RuleStepPair(BaseModel):
    rule_index: int = Field(description="Zero-based index into the safety_rules list.")
    step_indices: list[int] = Field(
        description="Zero-based step indices this rule applies to. "
        "Use -1 for general/workspace-level rules not specific to any step."
    )


class SafetyStepMapping(BaseModel):
    mappings: list[RuleStepPair] = Field(
        default_factory=list,
        description="One entry per safety rule mapping it to repair steps.",
    )


def _step_text(step: str | dict) -> str:
    if isinstance(step, dict):
        return step.get("text", str(step))
    return str(step)


def align_safety_to_steps(
    safety_rules: list[dict[str, Any]],
    repair_steps: list[str | dict],
) -> tuple[dict[int, list[dict[str, Any]]], dict | None]:
    """Map each safety rule to the repair step(s) it applies to.

    Returns ``(alignment_dict, telemetry_dict | None)``.
    ``alignment_dict`` keys are step indices; ``-1`` holds general rules.
    On failure returns ``({-1: safety_rules}, None)`` (flat fallback).
    """
    if not safety_rules or not repair_steps:
        return ({-1: safety_rules} if safety_rules else {}, None)

    rules_block = "\n".join(
        f"  [{i}] ({r.get('severity', 'CAUTION')}) {r.get('text', '')}"
        for i, r in enumerate(safety_rules)
    )
    steps_block = "\n".join(
        f"  [{i}] {_step_text(s)}" for i, s in enumerate(repair_steps)
    )

    prompt = PROMPT_SAFETY_STEP_ALIGNMENT.format(
        rules_block=rules_block, steps_block=steps_block
    )

    try:
        response, telemetry = tracked_generate_with_retry(
            model=MODEL_FLASH,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=SafetyStepMapping,
                temperature=0.0,
            ),
            node_name="SafetyStepAlignment",
        )
        if response is None:
            logger.warning(
                "Safety-step alignment LLM returned None; using flat fallback"
            )
            return {-1: safety_rules}, telemetry

        from core.utils import parse_gemini_json

        raw = parse_gemini_json(response.text)
        mapping = SafetyStepMapping.model_validate(raw)

        result: dict[int, list[dict[str, Any]]] = {}
        for pair in mapping.mappings:
            if pair.rule_index < 0 or pair.rule_index >= len(safety_rules):
                continue
            rule = safety_rules[pair.rule_index]
            for si in pair.step_indices:
                result.setdefault(si, []).append(rule)

        # Ensure every rule appears; unmatched rules go to -1 (general).
        mapped_indices = {p.rule_index for p in mapping.mappings}
        for i, rule in enumerate(safety_rules):
            if i not in mapped_indices:
                result.setdefault(-1, []).append(rule)

        return result, telemetry

    except Exception:
        logger.warning(
            "Safety-step alignment failed; using flat fallback", exc_info=True
        )
        return {-1: safety_rules}, None
