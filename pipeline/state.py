# backend/modules/graph/state.py
from typing import TypedDict, List, Annotated
from operator import add
from pydantic import BaseModel, Field


def merge_dicts(a: dict, b: dict) -> dict:
    c = a.copy() if a else {}
    if b:
        c.update(b)
    return c


class SupervisorIntakeSchema(BaseModel):
    intent: str = Field(
        description="EXACTLY ONE OF: 'troubleshoot', 'replace_part', 'followup', 'unsafe_method'"
    )
    extracted_part: str = Field(
        description="If intent is 'replace_part', extract the specific part name. Else leave empty."
    )
    equipment_type: str = Field(
        description="EXACTLY ONE OF: 'heavy_industrial' OR 'consumer_electronics'"
    )
    search_queries: List[str] = Field(
        description="3-5 highly targeted search queries. DO NOT include the Device ID."
    )
    detected_image_type: str = Field(
        default="", description="What exact object or device is shown in the image?"
    )
    is_device_mismatch: bool = Field(
        default=False,
        description="True if the image shows a completely different machine category.",
    )
    is_ambiguous_model: bool = Field(
        default=False,
        description="True if image shows correct type, but exact model cannot be verified.",
    )
    is_procedural: bool = Field(
        default=False,
        description="True when the user asks a procedural/walkthrough question (e.g. 'how to start', 'how to assemble').",
    )
    is_scoped_procedural: bool = Field(
        default=False,
        description="True when the user asks about a SPECIFIC step, section, or named procedure "
        "within a walkthrough (e.g. 'step 3', 'disassembly section', 'how to remove the motor'). "
        "Must be False when is_procedural is False.",
    )
    safety_warning: str = Field(
        default="",
        description="If intent is 'unsafe_method', a one-sentence plain-English warning describing the risk.",
    )
    corrected_intent: str = Field(
        default="",
        description="If intent is 'unsafe_method', the underlying intent the user was actually attempting "
        "(troubleshoot|replace_part|followup|validate_repair).",
    )


class DiagnosticState(TypedDict):
    device_id: str
    messages: Annotated[List[dict], add]
    route: str
    extracted_part: str
    equipment_type: str
    search_queries: List[str]
    detected_image_type: str
    is_procedural: bool
    is_scoped_procedural: bool
    retrieved_manuals: List[str]
    root_causes: List[str]
    repair_steps: List[str | dict]
    safety_protocols: list  # list[SafetyProtocol dict] from SafetyExtractor; legacy: list[str] from DeterministicRuleChecker
    safety_extraction_metadata: dict  # SafetyExtractionMetadata as dict, or None
    final_response: dict
    execution_telemetry: Annotated[List[dict], add]
    retrieved_image_map: dict
    response_id: str

    # --- LEGACY (kept for history-replay compat) ---
    judge_feedback: str

    # --- EVALUATION & SAFETY JUDGE ---
    inline_eval: dict
    safety_warning: str
    corrected_intent: str
    inline_eval_passed: bool
    flagged_for_review: bool
    # Safety judge retry guard — bumped by route_from_safety_judge when
    # it loops back to RepairPlanner, prevents infinite retry.
    safety_judge_retry_count: int

    # --- RETRIEVAL SCORE PROPAGATION (Group 1 + 1b) ---
    retrieval_scores: dict  # {"manual": {cid: {"cosine","adjusted","reranker"}}}
    retrieval_trace: (
        dict  # timing, candidate counts, reranker stats, neighbor expansion
    )

    # --- RLHF & FEEDBACK STATE (Workstream R) ---
    retrieved_chunk_ids: List[str]  # R1: chunk IDs from retrieval, for feedback linkage
    device_history_digest: str  # R5: prior session summary for IntentRouter
    deterministic_override: dict  # R10: matched rule metadata, if any
    anti_examples_injected: List[dict]  # R8: anti-examples injected into RepairPlanner

    # --- CHUNK GROUPING (token reduction) ---
    chunk_groups: list  # list[ChunkGroup] — grouped chunks from KnowledgeRetriever
    grouped_manual_context: str  # structured context string for RCA/RepairPlanner
