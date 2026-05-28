"""SafetyJudgeGate v2 node (L2-E09).

Verifies that safety protocols extracted by SafetyExtractor "survive" in the
generated repair plan. Computes a survival rate and gates the plan:
- pass: survival_rate >= threshold → proceed to InlineEvaluator
- fail: protocols missing → loop back to RepairPlanner with judge_feedback
- skip: no protocols to check → proceed directly

The gate prevents safety-blind plans from reaching the technician.
"""

import logging
import re
import time
from datetime import datetime, timezone

from pipeline.state import DiagnosticState
from pipeline.telemetry import make_node_telemetry

logger = logging.getLogger(__name__)

# Minimum survival rate to pass the gate
_SURVIVAL_THRESHOLD = 0.85
# Word-overlap parameters for protocol_survives()
_WORD_OVERLAP_THRESHOLD = 0.5
_MIN_WORD_LENGTH = 5


def _normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for fuzzy matching."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def protocol_survives(protocol: dict, plan_text: str) -> bool:
    """Check if a safety protocol is represented in the plan text.

    Uses keyword overlap: if 50%+ of the protocol's significant words
    appear in the plan, it's considered "survived".
    """
    protocol_text = protocol.get("text", "")
    if not protocol_text:
        return True  # Empty protocol trivially survives

    norm_plan = _normalize_text(plan_text)
    norm_protocol = _normalize_text(protocol_text)

    # Extract significant words (length >= _MIN_WORD_LENGTH) from the protocol
    word_pattern = rf"\b\w{{{_MIN_WORD_LENGTH},}}\b"
    protocol_words = set(re.findall(word_pattern, norm_protocol))
    if not protocol_words:
        # Very short protocol — check substring
        return norm_protocol in norm_plan

    # Check word overlap
    plan_words = set(re.findall(word_pattern, norm_plan))
    overlap = protocol_words & plan_words
    overlap_rate = len(overlap) / len(protocol_words)

    return overlap_rate >= _WORD_OVERLAP_THRESHOLD


def safety_judge_gate(state: DiagnosticState) -> dict:
    """Check whether safety protocols survived in the generated plan."""
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    safety_protocols = state.get("safety_protocols") or {}
    final_response = state.get("final_response") or {}

    # Extract protocols list
    if isinstance(safety_protocols, dict):
        protocols = safety_protocols.get("protocols", [])
        if not protocols:
            protocols = safety_protocols.get("rules", [])
    elif isinstance(safety_protocols, list):
        protocols = safety_protocols
    else:
        protocols = []

    # If no protocols, skip the gate
    if not protocols:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "execution_telemetry": [
                make_node_telemetry(
                    "SafetyJudgeGate",
                    started_at,
                    latency_ms,
                    metadata={
                        "gate_verdict": "skip",
                        "reason": "no_protocols",
                        "survival_rate": None,
                        "survival_threshold": _SURVIVAL_THRESHOLD,
                        "word_overlap_threshold": _WORD_OVERLAP_THRESHOLD,
                        "min_word_length": _MIN_WORD_LENGTH,
                        "per_protocol_scores": [],
                    },
                )
            ],
        }

    # Build plan text from repair steps
    repair_steps = final_response.get("repair_steps", [])
    plan_parts: list[str] = []
    for step in repair_steps:
        if isinstance(step, dict):
            plan_parts.append(step.get("text", ""))
        else:
            plan_parts.append(str(step))
    plan_text = " ".join(plan_parts)

    if not plan_text.strip():
        # No plan text to check against — skip
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "execution_telemetry": [
                make_node_telemetry(
                    "SafetyJudgeGate",
                    started_at,
                    latency_ms,
                    metadata={
                        "gate_verdict": "skip",
                        "reason": "empty_plan",
                        "survival_rate": None,
                        "survival_threshold": _SURVIVAL_THRESHOLD,
                        "word_overlap_threshold": _WORD_OVERLAP_THRESHOLD,
                        "min_word_length": _MIN_WORD_LENGTH,
                        "per_protocol_scores": [],
                    },
                )
            ],
        }

    # Check each protocol's survival
    survived: list[dict] = []
    missing: list[dict] = []

    for protocol in protocols:
        if protocol_survives(protocol, plan_text):
            survived.append(protocol)
        else:
            missing.append(protocol)

    total = len(protocols)
    survival_rate = len(survived) / total if total > 0 else 1.0
    passed = survival_rate >= _SURVIVAL_THRESHOLD

    latency_ms = int((time.perf_counter() - t0) * 1000)

    result: dict = {
        "execution_telemetry": [
            make_node_telemetry(
                "SafetyJudgeGate",
                started_at,
                latency_ms,
                metadata={
                    "gate_verdict": "pass" if passed else "fail",
                    "survival_rate": survival_rate,
                    "protocols_total": total,
                    "protocols_survived": len(survived),
                    "protocols_missing": len(missing),
                    "threshold": _SURVIVAL_THRESHOLD,
                    "survival_threshold": _SURVIVAL_THRESHOLD,
                    "word_overlap_threshold": 0.5,
                    "min_word_length": 5,
                    "protocols_checked": [p.get("text", "")[:100] for p in protocols],
                    "protocols_survived_list": [
                        p.get("text", "")[:100] for p in survived
                    ],
                    "protocols_missing_list": [
                        p.get("text", "")[:100] for p in missing
                    ],
                    "per_protocol_scores": [
                        {
                            "text": p.get("text", "")[:100],
                            "survived": p in survived,
                        }
                        for p in protocols
                    ],
                },
            )
        ],
    }

    if not passed:
        # Build feedback for RepairPlanner
        missing_descriptions = []
        for p in missing[:5]:  # Cap at 5 to avoid prompt overflow
            severity = p.get("severity", "WARNING")
            text = p.get("text", "unknown protocol")
            missing_descriptions.append(f"[{severity}] {text}")

        feedback = (
            f"SAFETY GATE FAILED: {len(missing)}/{total} safety protocols missing from plan.\n"
            f"Missing protocols:\n"
            + "\n".join(f"- {d}" for d in missing_descriptions)
            + "\n\nYou MUST incorporate these safety warnings into the repair plan."
        )
        result["judge_feedback"] = feedback
        logger.warning(
            f"[SafetyJudgeGate] FAIL — survival_rate={survival_rate:.2f} "
            f"({len(survived)}/{total} survived), threshold={_SURVIVAL_THRESHOLD}"
        )
    else:
        # Explicitly clear judge_feedback so stale values from a prior
        # iteration don't trick route_from_safety_judge into looping.
        result["judge_feedback"] = ""
        logger.info(
            f"[SafetyJudgeGate] PASS — survival_rate={survival_rate:.2f} "
            f"({len(survived)}/{total} survived)"
        )

    return result
