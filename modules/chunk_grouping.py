"""
Chunk grouping by failure mode / topic.

Groups retrieved chunks to reduce redundancy before passing to downstream LLM calls.
Uses section_header metadata for grouping. Pure functions — no side effects, no DB, no LLM.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


def _chunk_score_key(c: dict) -> tuple[float, float]:
    """Sort key for ranking chunks by reranker_score then cosine similarity."""
    reranker = c.get("reranker_score")
    cosine = c.get("score")
    return (
        float(reranker) if reranker is not None else -1.0,
        float(cosine) if cosine is not None else -1.0,
    )


@dataclass
class ChunkGroup:
    """A group of related chunks sharing a failure mode / topic."""

    label: str
    chunks: list[dict]
    representatives: list[dict] = field(default_factory=list)
    chunk_ids: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.chunk_ids = {c.get("id", "") for c in self.chunks if c.get("id")}


def select_representatives(
    group: ChunkGroup,
    max_per_group: int = 2,
) -> list[dict]:
    """Select the most informative chunks from a group.

    Selection criteria (priority order):
    1. Highest reranker_score (if available)
    2. Highest cosine similarity score (if available)
    3. First by insertion order (fallback)
    """
    if len(group.chunks) <= max_per_group:
        return list(group.chunks)

    ranked = sorted(group.chunks, key=_chunk_score_key, reverse=True)
    return ranked[:max_per_group]


def group_chunks(
    chunks: list[dict],
    max_group_size: int = 8,
    max_representatives: int = 2,
) -> list[ChunkGroup]:
    """Group retrieved chunks by section_header.

    Strategy:
    1. Group by exact section_header match.
    2. Chunks without a section_header go to an "ungrouped" bucket.
    3. Split any group exceeding max_group_size by keeping the top-scored subset.
    4. Select representative chunks per group.

    Args:
        chunks: Chunk dicts from KnowledgeRetriever ordered list.
            Each must have at minimum: "id", "text".
            Optional: "section_header", "reranker_score", "score".
        max_group_size: Maximum chunks per group before trimming.
        max_representatives: Maximum representative chunks per group.

    Returns:
        List of ChunkGroup instances with representatives selected.
    """
    if not chunks:
        return []

    # Step 1: group by section_header
    buckets: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        header = (chunk.get("section_header") or "").strip()
        label = header if header else "ungrouped"
        buckets[label].append(chunk)

    groups: list[ChunkGroup] = []
    for label, members in buckets.items():
        # Step 2: trim oversized groups by score
        if len(members) > max_group_size:
            members = sorted(members, key=_chunk_score_key, reverse=True)[
                :max_group_size
            ]

        group = ChunkGroup(label=label, chunks=members)
        group.representatives = select_representatives(group, max_representatives)
        groups.append(group)

    return groups


def build_grouped_context(
    groups: list[ChunkGroup],
    include_group_headers: bool = True,
    max_total_chunks: int = 12,
) -> str:
    """Build a structured context string from grouped representative chunks.

    If include_group_headers is True, format as:
        --- {label} ({n} sources, showing {k}) ---
        [{doc_title}, p.{pages}]
        {text}
        ...

    max_total_chunks caps the total representative chunks across all groups.
    If sum of representatives exceeds this, reduce per-group allocation
    proportionally (minimum 1 per group).
    """
    if not groups:
        return ""

    # Count total representatives
    total_reps = sum(len(g.representatives) for g in groups)

    # Proportional reduction if needed
    if total_reps > max_total_chunks and len(groups) > 0:
        budget = max_total_chunks
        if len(groups) <= budget:
            # Every group gets at least 1, distribute remainder proportionally
            allocations: list[int] = [1] * len(groups)
            remainder = budget - len(groups)
            if remainder > 0:
                total_chunks = sum(len(g.chunks) for g in groups)
                for i, g in enumerate(groups):
                    if total_chunks > 0:
                        extra = int(remainder * len(g.chunks) / total_chunks)
                        allocations[i] += extra
        else:
            # More groups than budget — prioritize groups with highest-scored reps
            scored = sorted(
                enumerate(groups),
                key=lambda ig: (
                    max(
                        (r.get("reranker_score") or r.get("score") or 0)
                        for r in ig[1].representatives
                    )
                    if ig[1].representatives
                    else 0
                ),
                reverse=True,
            )
            allocations = [0] * len(groups)
            for rank, (idx, _) in enumerate(scored):
                if rank < budget:
                    allocations[idx] = 1

        effective_groups = [
            (g, g.representatives[:alloc]) for g, alloc in zip(groups, allocations)
        ]
    else:
        effective_groups = [(g, g.representatives) for g in groups]

    parts: list[str] = []
    for group, reps in effective_groups:
        if not reps:
            continue

        if include_group_headers:
            parts.append(
                f"--- {group.label} ({len(group.chunks)} sources, showing {len(reps)}) ---"
            )

        for chunk in reps:
            doc_title = chunk.get("doc_title") or chunk.get("intuitive_name") or ""
            pages = chunk.get("pages") or ""
            header_line = ""
            if doc_title or pages:
                header_parts = []
                if doc_title:
                    header_parts.append(doc_title)
                if pages:
                    header_parts.append(f"p.{pages}")
                header_line = f"[{', '.join(header_parts)}]\n"

            parts.append(f"{header_line}{chunk.get('text', '')}")

    return "\n\n".join(parts)


def filter_context_by_chunk_ids(
    all_chunks: list[dict],
    target_chunk_ids: set[str],
    fallback_to_all: bool = True,
) -> str:
    """Build a context string containing only chunks whose IDs are in target_chunk_ids.

    Used by faithfulness evaluation to scope context to only the chunks
    cited by the plan.

    If target_chunk_ids is empty and fallback_to_all is True, returns the
    full concatenated context. Logs a warning when falling back.
    """
    if not target_chunk_ids:
        if fallback_to_all:
            logger.warning(
                "filter_context_by_chunk_ids: empty target_chunk_ids, "
                "falling back to all chunks"
            )
            return "\n\n".join(c.get("text", "") for c in all_chunks if c.get("text"))
        return ""

    filtered = [c for c in all_chunks if c.get("id") in target_chunk_ids]
    return "\n\n".join(c.get("text", "") for c in filtered if c.get("text"))


def extract_cited_chunk_ids(
    repair_steps: list[str | dict],
    ordered_chunks: list[dict],
) -> set[str]:
    """Extract chunk IDs cited by the repair plan.

    Handles both formats:
    - Procedural: repair_steps is list[dict] with "chunk_refs" key per step
      (e.g., [{"text": "...", "chunk_refs": ["CHUNK-0", "CHUNK-2"]}])
    - Simple: repair_steps is list[str] — no chunk_refs, return empty set

    CHUNK-N values are resolved to actual chunk IDs using the positional
    index in ordered_chunks.
    """
    if not repair_steps or not ordered_chunks:
        return set()

    cited: set[str] = set()
    for step in repair_steps:
        if not isinstance(step, dict):
            continue
        for ref in step.get("chunk_refs", []):
            m = re.match(r"CHUNK-(\d+)", str(ref))
            if m:
                idx = int(m.group(1))
                if 0 <= idx < len(ordered_chunks):
                    chunk_id = ordered_chunks[idx].get("id", "")
                    if chunk_id:
                        cited.add(chunk_id)

    return cited
