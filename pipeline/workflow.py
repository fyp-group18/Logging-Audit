# backend/modules/graph/workflow.py
from langgraph.graph import StateGraph, END
from pipeline.state import DiagnosticState
from pipeline.nodes import (
    deterministic_rule_checker,
    direct_replacement_analyzer,
    followup_responder,
    inline_evaluator,
    intent_router,
    knowledge_retriever,
    repair_planner,
    root_cause_analyzer,
    safety_extractor,
    symptom_analyzer,
    unsafe_method_gate,
)
from pipeline.safety_judge_gate import safety_judge_gate
from pipeline.telemetry import record_edge_decision

workflow = StateGraph(DiagnosticState)

workflow.add_node("IntentRouter", intent_router)
workflow.add_node("UnsafeMethodGate", unsafe_method_gate)
workflow.add_node("FollowUpResponder", followup_responder)
workflow.add_node("SymptomAnalyzer", symptom_analyzer)
workflow.add_node("DirectReplacementAnalyzer", direct_replacement_analyzer)
workflow.add_node("DeterministicRuleChecker", deterministic_rule_checker)
workflow.add_node("KnowledgeRetriever", knowledge_retriever)
workflow.add_node("RootCauseAnalyzer", root_cause_analyzer)
workflow.add_node("SafetyExtractor", safety_extractor)
workflow.add_node("RepairPlanner", repair_planner)
workflow.add_node("SafetyJudgeGate", safety_judge_gate)
workflow.add_node("InlineEvaluator", inline_evaluator)

workflow.set_entry_point("IntentRouter")


_INTENT_DISPATCH = {
    "DeterministicRuleChecker": "DeterministicRuleChecker",
    "FollowUpResponder": "FollowUpResponder",
}


def _dispatch_target(route: str) -> str:
    """Map a (possibly corrected) intent to its downstream node.

    Routes that need RAG (troubleshoot, replace_part, mismatch, ambiguous_model)
    go through DeterministicRuleChecker first — it short-circuits to END if a
    rule matches, otherwise continues to the appropriate analysis node.
    """
    if route in ("mismatch", "ambiguous_model", "troubleshoot", "replace_part"):
        return "DeterministicRuleChecker"
    if route == "followup":
        return "FollowUpResponder"
    return "DeterministicRuleChecker"


def route_from_intent(state: DiagnosticState):
    route = state.get("route", "troubleshoot")
    if route == "unsafe_method":
        target = "UnsafeMethodGate"
    else:
        target = _dispatch_target(route)
    record_edge_decision("route_from_intent", target, {"route": route})
    return target


workflow.add_conditional_edges(
    "IntentRouter",
    route_from_intent,
    {"UnsafeMethodGate": "UnsafeMethodGate", **_INTENT_DISPATCH},
)


def route_from_deterministic_check(state: DiagnosticState):
    """If a deterministic rule matched, short-circuit to END. Otherwise proceed
    to the original route (SymptomAnalyzer or DirectReplacementAnalyzer)."""
    override = state.get("deterministic_override")
    if override:
        target = END
    elif state.get("route", "troubleshoot") == "replace_part":
        target = "DirectReplacementAnalyzer"
    else:
        target = "SymptomAnalyzer"
    record_edge_decision(
        "route_from_deterministic_check", str(target), {"has_override": bool(override)}
    )
    return target


workflow.add_conditional_edges(
    "DeterministicRuleChecker",
    route_from_deterministic_check,
    {
        "SymptomAnalyzer": "SymptomAnalyzer",
        "DirectReplacementAnalyzer": "DirectReplacementAnalyzer",
        END: END,
    },
)


def route_from_unsafe(state: DiagnosticState):
    """UnsafeMethodGate writes the corrected intent into `route` — dispatch normally."""
    route = state.get("route", "troubleshoot")
    target = _dispatch_target(route)
    record_edge_decision("route_from_unsafe", target, {"route": route})
    return target


workflow.add_conditional_edges(
    "UnsafeMethodGate",
    route_from_unsafe,
    _INTENT_DISPATCH,
)

workflow.add_edge("SymptomAnalyzer", "KnowledgeRetriever")
workflow.add_edge("DirectReplacementAnalyzer", "KnowledgeRetriever")


def route_from_knowledge(state: DiagnosticState):
    """Route after retrieval: replace_part skips RCA and goes straight to
    SafetyExtractor; all other routes go through RootCauseAnalyzer first.
    """
    route = state.get("route", "troubleshoot")
    if route == "replace_part":
        target = "SafetyExtractor"
    else:
        target = "RootCauseAnalyzer"
    record_edge_decision("route_from_knowledge", target, {"route": route})
    return target


workflow.add_conditional_edges(
    "KnowledgeRetriever",
    route_from_knowledge,
    {"RootCauseAnalyzer": "RootCauseAnalyzer", "SafetyExtractor": "SafetyExtractor"},
)


_NO_DOCS_PATTERNS = [
    "no relevant documentation",
    "no relevant documents",
    "no documentation was found",
]


def route_from_root_cause(state: DiagnosticState):
    """After RCA: mismatch and no-docs short-circuit to END.
    All other routes proceed to SafetyExtractor (then RepairPlanner).
    """
    route = state.get("route", "")
    if route == "mismatch":
        target = END
    else:
        rcs = state.get("root_causes", [""])
        if rcs and any(pattern in rcs[0].lower() for pattern in _NO_DOCS_PATTERNS):
            target = END
        else:
            target = "SafetyExtractor"
    record_edge_decision("route_from_root_cause", str(target), {"route": route})
    return target


workflow.add_conditional_edges(
    "RootCauseAnalyzer",
    route_from_root_cause,
    {"SafetyExtractor": "SafetyExtractor", END: END},
)

# SafetyExtractor always proceeds to RepairPlanner.
workflow.add_edge("SafetyExtractor", "RepairPlanner")

# Plan-producing nodes feed the SafetyJudgeGate → InlineEvaluator pipeline.
workflow.add_edge("RepairPlanner", "SafetyJudgeGate")
workflow.add_edge("FollowUpResponder", "InlineEvaluator")


def route_from_safety_judge(state: DiagnosticState):
    """SafetyJudgeGate routing.

    - pass/skip: proceed to InlineEvaluator
    - fail (first time): loop back to RepairPlanner with judge_feedback
    - fail (already retried): proceed to InlineEvaluator (don't loop forever)
    """
    judge_feedback = state.get("judge_feedback", "")
    retry = int(state.get("safety_judge_retry_count") or 0)

    if not judge_feedback:
        target = "InlineEvaluator"
    elif retry >= 1:
        target = "InlineEvaluator"
    else:
        target = "RepairPlanner"
    record_edge_decision(
        "route_from_safety_judge",
        str(target),
        {"has_feedback": bool(judge_feedback), "safety_judge_retry_count": retry},
    )
    return target


workflow.add_conditional_edges(
    "SafetyJudgeGate",
    route_from_safety_judge,
    {"InlineEvaluator": "InlineEvaluator", "RepairPlanner": "RepairPlanner"},
)


workflow.add_edge("InlineEvaluator", END)
