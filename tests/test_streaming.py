"""Tests for StreamingKnowledgeGraph background event processing and thread safety."""
from knowledge_graph_pkg.core import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.streaming import StreamingKnowledgeGraph


def test_streaming_minimal_add_update_delete_and_non_mutation():
    kg = KnowledgeGraph()
    skg = StreamingKnowledgeGraph(kg, auto_start=True)
    try:
        payload = {"fact_id": "s1", "fact_statement": "Streamed initial fact."}
        snapshot = dict(payload)
        skg.add_fact_from_stream(payload, source_id="stream_src", reliability=ReliabilityRating.LIKELY_TRUE)
        skg.update_fact_from_stream("s1", {"fact_statement": "Streamed updated fact.", "reliability_rating": "VERIFIED"})
        skg.event_queue.join()

        assert payload == snapshot
        fact = kg.get_fact("s1")
        assert fact["fact_statement"] == "Streamed updated fact."
        assert fact["reliability_rating"] == ReliabilityRating.VERIFIED

        skg.delete_fact_from_stream("s1")
        skg.event_queue.join()
        assert "s1" not in kg.graph
    finally:
        skg.stop()


def test_streaming_history_limit_zero_and_processor_exception_releases_queue():
    kg = KnowledgeGraph()
    skg = StreamingKnowledgeGraph(kg, auto_start=True)
    try:
        def _boom(_event):
            raise RuntimeError("boom")

        skg.register_processor("boom_event", _boom)
        skg.add_event("boom_event", {"k": "v"})
        skg.add_event("unknown_event", {"k": "v2"})
        # Queue join must not deadlock even after processor exception.
        skg.event_queue.join()

        assert skg.get_event_history(limit=0) == []
        hist = skg.get_event_history(limit=10)
        assert len(hist) == 2
    finally:
        skg.stop()
