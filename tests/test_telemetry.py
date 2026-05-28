# backend/tests/test_telemetry.py
"""Telemetry module tests. Each test takes the `client` fixture to ensure
core.models is imported after the agent_manager patch (avoids init_db circular import)."""

from types import SimpleNamespace
from unittest.mock import MagicMock
from datetime import datetime, timezone


def test_make_record_structure(client):
    """make_record returns a dict with all required keys."""
    from pipeline.telemetry import make_record

    now = datetime.now(timezone.utc)
    record = make_record(
        node_name="TestNode",
        model="gemini-2.5-pro",
        operation_type="llm_call",
        input_tokens=100,
        output_tokens=50,
        latency_ms=500,
        started_at=now,
        completed_at=now,
    )
    assert record["node_name"] == "TestNode"
    assert record["model"] == "gemini-2.5-pro"
    assert record["input_tokens"] == 100
    assert record["output_tokens"] == 50
    assert record["latency_ms"] == 500
    assert record["error"] is None


def test_make_record_with_error(client):
    """make_record stores error string."""
    from pipeline.telemetry import make_record

    now = datetime.now(timezone.utc)
    record = make_record(
        node_name="TestNode",
        model=None,
        operation_type="node",
        input_tokens=None,
        output_tokens=None,
        latency_ms=0,
        started_at=now,
        completed_at=now,
        error="Connection timeout",
    )
    assert record["error"] == "Connection timeout"


def test_tracked_generate_success(client):
    """tracked_generate captures response and telemetry on success."""
    from pipeline.telemetry import tracked_generate

    mock_usage = SimpleNamespace(
        prompt_token_count=100,
        candidates_token_count=50,
    )
    mock_response = SimpleNamespace(usage_metadata=mock_usage)
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    response, telemetry = tracked_generate(
        client=mock_client,
        model="gemini-2.5-pro",
        contents="Hello",
        config={},
        node_name="TestNode",
    )
    assert response is mock_response
    assert telemetry["node_name"] == "TestNode"
    assert telemetry["input_tokens"] == 100
    assert telemetry["output_tokens"] == 50
    assert telemetry["error"] is None
    assert telemetry["latency_ms"] >= 0
    mock_client.models.generate_content.assert_called_once()


def test_tracked_generate_error(client):
    """tracked_generate returns None and error telemetry on failure."""
    from pipeline.telemetry import tracked_generate

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("API error")

    response, telemetry = tracked_generate(
        client=mock_client,
        model="gemini-2.5-pro",
        contents="Hello",
        config={},
        node_name="TestNode",
    )
    assert response is None
    assert telemetry["error"] == "API error"
    assert telemetry["input_tokens"] is None


def test_tracked_invoke_success(client):
    """tracked_invoke captures LangChain response metadata."""
    from pipeline.telemetry import tracked_invoke

    mock_response = SimpleNamespace(
        content="Result text",
        response_metadata={
            "usage_metadata": {
                "prompt_token_count": 200,
                "candidates_token_count": 75,
            }
        },
    )
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    response, telemetry = tracked_invoke(
        chain=mock_chain,
        input_value="test input",
        node_name="RCA",
        model_name="gemini-2.5-flash",
    )
    assert response is mock_response
    assert telemetry["input_tokens"] == 200
    assert telemetry["output_tokens"] == 75
    assert telemetry["model"] == "gemini-2.5-flash"


def test_tracked_invoke_error(client):
    """tracked_invoke returns None on error."""
    from pipeline.telemetry import tracked_invoke

    mock_chain = MagicMock()
    mock_chain.invoke.side_effect = ValueError("Bad input")

    response, telemetry = tracked_invoke(
        chain=mock_chain,
        input_value="test",
        node_name="RCA",
        model_name="gemini-2.5-flash",
    )
    assert response is None
    assert "Bad input" in telemetry["error"]


def test_tracked_stream_success(client):
    """tracked_stream accumulates tokens from LangChain stream."""
    from pipeline.telemetry import tracked_stream

    chunks = [
        SimpleNamespace(content="Hello "),
        SimpleNamespace(content="world"),
    ]
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter(chunks)

    def extract(chunk):
        return chunk.content if hasattr(chunk, "content") else str(chunk)

    text, telemetry = tracked_stream(
        llm=mock_llm,
        messages="test",
        node_name="RCA",
        model_name="gemini-2.5-flash",
        extract_fn=extract,
    )
    assert text == "Hello world"
    assert telemetry["error"] is None
    assert telemetry["latency_ms"] >= 0


def test_tracked_stream_with_writer(client):
    """tracked_stream calls writer for each token."""
    from pipeline.telemetry import tracked_stream

    chunks = [SimpleNamespace(content="A"), SimpleNamespace(content="B")]
    mock_llm = MagicMock()
    mock_llm.stream.return_value = iter(chunks)
    writer_calls = []

    def extract(chunk):
        return chunk.content

    text, telemetry = tracked_stream(
        llm=mock_llm,
        messages="test",
        node_name="RCA",
        model_name="gemini-2.5-flash",
        writer=lambda payload: writer_calls.append(payload),
        extract_fn=extract,
    )
    assert text == "AB"
    assert len(writer_calls) == 2
    assert writer_calls[0] == {"type": "rca_token", "token": "A"}


def test_tracked_stream_error(client):
    """tracked_stream returns None on mid-stream error (partial text discarded)."""
    from pipeline.telemetry import tracked_stream

    def failing_stream(messages):
        yield SimpleNamespace(content="partial")
        raise RuntimeError("Stream died")

    mock_llm = MagicMock()
    mock_llm.stream.side_effect = failing_stream

    def extract(chunk):
        return chunk.content

    text, telemetry = tracked_stream(
        llm=mock_llm,
        messages="test",
        node_name="RCA",
        model_name="gemini-2.5-flash",
        extract_fn=extract,
    )
    assert text is None
    assert "Stream died" in telemetry["error"]


def test_record_edge_decision(client):
    """record_edge_decision accumulates records in thread-keyed store."""
    from pipeline.telemetry import (
        record_edge_decision,
        collect_edge_telemetry,
        set_edge_telemetry_thread_id,
    )

    # Initialize the store for a test thread
    set_edge_telemetry_thread_id("test-thread")

    record_edge_decision("route_intent", "SYMPTOM_ANALYSIS", {"intent": "diagnostic"})
    record_edge_decision("route_intent", "FOLLOW_UP")

    records = collect_edge_telemetry("test-thread")
    assert len(records) == 2
    assert records[0]["node_name"] == "edge:route_intent"
    assert records[0]["metadata"]["decision"] == "SYMPTOM_ANALYSIS"

    # After collection, should be empty
    assert collect_edge_telemetry("test-thread") == []


def test_make_node_telemetry(client):
    """make_node_telemetry builds non-LLM node record."""
    from pipeline.telemetry import make_node_telemetry

    now = datetime.now(timezone.utc)
    record = make_node_telemetry(
        node_name="DeterministicRuleChecker",
        started_at=now,
        latency_ms=5,
        metadata={"rule_matched": True},
    )
    assert record["node_name"] == "DeterministicRuleChecker"
    assert record["operation_type"] == "node"
    assert record["model"] is None
    assert record["metadata"]["rule_matched"] is True


def test_extract_genai_tokens_success(client):
    """_extract_genai_tokens extracts token counts from response."""
    from pipeline.telemetry import _extract_genai_tokens

    mock_response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
        )
    )
    inp, out = _extract_genai_tokens(mock_response)
    assert inp == 100
    assert out == 50


def test_extract_genai_tokens_missing(client):
    """_extract_genai_tokens returns None for missing metadata."""
    from pipeline.telemetry import _extract_genai_tokens

    inp, out = _extract_genai_tokens(SimpleNamespace())
    assert inp is None
    assert out is None
