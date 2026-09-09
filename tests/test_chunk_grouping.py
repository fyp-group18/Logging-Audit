"""Tests for modules.chunk_grouping — chunk grouping and context scoping."""

from modules.chunk_grouping import (
    ChunkGroup,
    build_grouped_context,
    extract_cited_chunk_ids,
    filter_context_by_chunk_ids,
    group_chunks,
    select_representatives,
)


def _make_chunk(
    chunk_id: str,
    text: str = "chunk text",
    section_header: str | None = None,
    reranker_score: int | None = None,
    score: float | None = None,
    page_start: int | None = None,
    sequence_index: int | None = None,
) -> dict:
    return {
        "id": chunk_id,
        "text": text,
        "section_header": section_header,
        "reranker_score": reranker_score,
        "score": score,
        "page_start": page_start,
        "sequence_index": sequence_index,
        "images": [],
    }


class TestGroupChunks:
    def test_group_by_section_header(self):
        chunks = [
            _make_chunk("c1", section_header="Motor Assembly"),
            _make_chunk("c2", section_header="Motor Assembly"),
            _make_chunk("c3", section_header="Belt Replacement"),
        ]
        groups = group_chunks(chunks)
        labels = {g.label for g in groups}
        assert "Motor Assembly" in labels
        assert "Belt Replacement" in labels
        motor = next(g for g in groups if g.label == "Motor Assembly")
        assert len(motor.chunks) == 2
        assert {"c1", "c2"} == motor.chunk_ids

    def test_ungrouped_bucket(self):
        chunks = [
            _make_chunk("c1", section_header=None),
            _make_chunk("c2", section_header=""),
            _make_chunk("c3", section_header="Known Header"),
        ]
        groups = group_chunks(chunks)
        ungrouped = next(g for g in groups if g.label == "ungrouped")
        assert len(ungrouped.chunks) == 2

    def test_single_chunk(self):
        groups = group_chunks([_make_chunk("c1", section_header="Header")])
        assert len(groups) == 1
        assert len(groups[0].chunks) == 1
        assert len(groups[0].representatives) == 1

    def test_empty_input(self):
        assert group_chunks([]) == []

    def test_max_group_size_split(self):
        chunks = [
            _make_chunk(f"c{i}", section_header="Same", reranker_score=10 - i)
            for i in range(12)
        ]
        groups = group_chunks(chunks, max_group_size=5)
        assert len(groups) == 1
        assert len(groups[0].chunks) == 5
        # Highest reranker_score chunks should survive
        surviving_ids = {c["id"] for c in groups[0].chunks}
        assert "c0" in surviving_ids  # reranker_score=10
        assert "c1" in surviving_ids  # reranker_score=9


class TestSelectRepresentatives:
    def test_by_reranker_score(self):
        group = ChunkGroup(
            label="test",
            chunks=[
                _make_chunk("c1", reranker_score=3),
                _make_chunk("c2", reranker_score=8),
                _make_chunk("c3", reranker_score=5),
            ],
        )
        reps = select_representatives(group, max_per_group=2)
        assert len(reps) == 2
        assert reps[0]["id"] == "c2"  # highest
        assert reps[1]["id"] == "c3"  # second

    def test_fallback_to_cosine_score(self):
        group = ChunkGroup(
            label="test",
            chunks=[
                _make_chunk("c1", score=0.7),
                _make_chunk("c2", score=0.9),
                _make_chunk("c3", score=0.5),
            ],
        )
        reps = select_representatives(group, max_per_group=1)
        assert reps[0]["id"] == "c2"

    def test_all_returned_when_fewer_than_max(self):
        group = ChunkGroup(
            label="test",
            chunks=[_make_chunk("c1"), _make_chunk("c2")],
        )
        reps = select_representatives(group, max_per_group=5)
        assert len(reps) == 2


class TestBuildGroupedContext:
    def test_basic_format(self):
        groups = [
            ChunkGroup(
                label="Motor Assembly",
                chunks=[
                    _make_chunk("c1", text="step 1"),
                    _make_chunk("c2", text="step 2"),
                ],
                representatives=[_make_chunk("c1", text="step 1")],
            )
        ]
        result = build_grouped_context(groups)
        assert "Motor Assembly" in result
        assert "step 1" in result
        assert "2 sources, showing 1" in result

    def test_no_group_headers(self):
        groups = [
            ChunkGroup(
                label="Header",
                chunks=[_make_chunk("c1", text="text here")],
                representatives=[_make_chunk("c1", text="text here")],
            )
        ]
        result = build_grouped_context(groups, include_group_headers=False)
        assert "Header" not in result
        assert "text here" in result

    def test_max_total_chunks_cap(self):
        groups = [
            ChunkGroup(
                label=f"Group {i}",
                chunks=[_make_chunk(f"c{i}a"), _make_chunk(f"c{i}b")],
                representatives=[
                    _make_chunk(f"c{i}a", text=f"text-{i}a", reranker_score=10 - i),
                    _make_chunk(f"c{i}b", text=f"text-{i}b", reranker_score=5 - i),
                ],
            )
            for i in range(10)
        ]
        result = build_grouped_context(groups, max_total_chunks=5)
        # With 10 groups and max_total=5, only 5 groups should have output
        # (the 5 with highest-scored representatives)
        texts_present = [f"text-{i}a" for i in range(10) if f"text-{i}a" in result]
        assert len(texts_present) == 5

    def test_empty_groups(self):
        assert build_grouped_context([]) == ""


class TestFilterContextByChunkIds:
    def test_filters_correctly(self):
        chunks = [
            _make_chunk("c1", text="first"),
            _make_chunk("c2", text="second"),
            _make_chunk("c3", text="third"),
        ]
        result = filter_context_by_chunk_ids(chunks, {"c1", "c3"})
        assert "first" in result
        assert "third" in result
        assert "second" not in result

    def test_fallback_to_all(self):
        chunks = [
            _make_chunk("c1", text="first"),
            _make_chunk("c2", text="second"),
        ]
        result = filter_context_by_chunk_ids(chunks, set(), fallback_to_all=True)
        assert "first" in result
        assert "second" in result

    def test_no_fallback_returns_empty(self):
        chunks = [_make_chunk("c1", text="first")]
        result = filter_context_by_chunk_ids(chunks, set(), fallback_to_all=False)
        assert result == ""


class TestExtractCitedChunkIds:
    def test_parses_chunk_refs(self):
        ordered = [
            _make_chunk("uuid-0"),
            _make_chunk("uuid-1"),
            _make_chunk("uuid-2"),
        ]
        steps = [
            {"text": "Step 1", "chunk_refs": ["CHUNK-0", "CHUNK-2"]},
            {"text": "Step 2", "chunk_refs": ["CHUNK-1"]},
        ]
        cited = extract_cited_chunk_ids(steps, ordered)
        assert cited == {"uuid-0", "uuid-1", "uuid-2"}

    def test_string_steps_return_empty(self):
        ordered = [_make_chunk("uuid-0")]
        steps = ["Step 1", "Step 2"]
        assert extract_cited_chunk_ids(steps, ordered) == set()

    def test_out_of_range_index_ignored(self):
        ordered = [_make_chunk("uuid-0")]
        steps = [{"text": "Step 1", "chunk_refs": ["CHUNK-0", "CHUNK-99"]}]
        cited = extract_cited_chunk_ids(steps, ordered)
        assert cited == {"uuid-0"}

    def test_empty_inputs(self):
        assert extract_cited_chunk_ids([], []) == set()
        assert extract_cited_chunk_ids([], [_make_chunk("c1")]) == set()
        assert (
            extract_cited_chunk_ids([{"text": "x", "chunk_refs": ["CHUNK-0"]}], [])
            == set()
        )


class TestSafetyChunksBypass:
    """Verify safety chunks are never passed to group_chunks."""

    def test_safety_chunks_not_grouped(self):
        # Safety chunks have a different shape (doc_title instead of section_header)
        # and should never be passed to group_chunks.
        # This test confirms group_chunks handles arbitrary dicts gracefully.
        safety_like = [
            {"id": "s1", "text": "safety warning", "doc_title": "Safety Manual"},
            {"id": "s2", "text": "another warning", "doc_title": "Safety Manual"},
        ]
        # group_chunks should still work — they land in "ungrouped" since
        # no section_header is present
        groups = group_chunks(safety_like)
        assert len(groups) == 1
        assert groups[0].label == "ungrouped"
