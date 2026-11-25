"""Tests for knowledge_graph_pkg.utils import/export, filtering, merging, and URL extraction."""
import csv
import sys
import types
from datetime import datetime
from unittest.mock import patch

import pytest

from knowledge_graph_pkg.core import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.utils import (
    export_to_json,
    extract_facts_from_url,
    filter_facts_by_reliability,
    import_from_csv,
    import_from_json,
    merge_knowledge_graphs,
)


def _populate(kg: KnowledgeGraph, fact_id: str, statement: str, rating=ReliabilityRating.VERIFIED):
    now = datetime(2026, 1, 1)
    kg.add_fact(
        fact_id=fact_id,
        fact_statement=statement,
        category="Science",
        tags=["tag1", "tag2"],
        date_recorded=now,
        last_updated=now,
        reliability_rating=rating,
        source_id="s1",
        source_title="Source",
        author_creator="Author",
        publication_date=now,
        url_reference="https://example.com",
        related_facts=[],
        contextual_notes="note",
        access_level="public",
        usage_count=2,
    )


def test_json_roundtrip_handles_quality_score_and_string_tags(tmp_path):
    kg = KnowledgeGraph()
    _populate(kg, "f1", "Water boils at 100 C.", ReliabilityRating.VERIFIED)
    path = tmp_path / "kg.json"
    assert export_to_json(kg, str(path)) == 1

    kg2 = KnowledgeGraph()
    assert import_from_json(kg2, str(path)) == 1
    fact = kg2.get_fact("f1")
    assert fact["fact_statement"] == "Water boils at 100 C."
    assert fact["reliability_rating"] == ReliabilityRating.VERIFIED
    assert fact["tags"] == "tag1, tag2"


def test_import_from_csv_fills_missing_optional_columns(tmp_path):
    csv_path = tmp_path / "facts.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fact_id", "fact_statement", "reliability_rating"])
        writer.writeheader()
        writer.writerow({"fact_id": "c1", "fact_statement": "Helium is a noble gas.", "reliability_rating": "LIKELY_TRUE"})
        writer.writerow({"fact_id": "c2", "fact_statement": "Fallback rating row.", "reliability_rating": "BAD_VALUE"})

    kg = KnowledgeGraph()
    assert import_from_csv(kg, str(csv_path)) == 2
    assert kg.get_fact("c1")["reliability_rating"] == ReliabilityRating.LIKELY_TRUE
    assert kg.get_fact("c2")["reliability_rating"] == ReliabilityRating.UNVERIFIED


def test_filter_and_merge_preserve_edges_and_apply_conflict_strategies():
    kg1 = KnowledgeGraph()
    _populate(kg1, "f1", "Draft statement.", ReliabilityRating.POSSIBLY_TRUE)
    _populate(kg1, "f2", "Verified fact.", ReliabilityRating.VERIFIED)
    kg1.graph.add_edge("f1", "f2")

    filtered = filter_facts_by_reliability(kg1, ReliabilityRating.VERIFIED)
    assert list(filtered.graph.nodes) == ["f2"]

    kg2 = KnowledgeGraph()
    _populate(kg2, "f1", "Verified update.", ReliabilityRating.VERIFIED)
    _populate(kg2, "f3", "Third fact.", ReliabilityRating.LIKELY_TRUE)
    kg2.graph.add_edge("f1", "f3")

    merged_high = merge_knowledge_graphs(kg1, kg2, conflict_strategy="higher_reliability")
    assert merged_high.get_fact("f1")["fact_statement"] == "Verified update."
    assert merged_high.graph.has_edge("f1", "f3")

    merged_first = merge_knowledge_graphs(kg1, kg2, conflict_strategy="keep_first")
    assert merged_first.get_fact("f1")["fact_statement"] == "Draft statement."

    merged_second = merge_knowledge_graphs(kg1, kg2, conflict_strategy="keep_second")
    assert merged_second.get_fact("f1")["fact_statement"] == "Verified update."

    with pytest.raises(ValueError, match="Invalid conflict strategy"):
        merge_knowledge_graphs(kg1, kg2, conflict_strategy="invalid")


def test_extract_facts_from_url_validates_scheme_and_uses_timeout():
    with pytest.raises(ValueError, match="Unsupported or invalid URL"):
        extract_facts_from_url("file:///etc/passwd", lambda soup: [])

    seen_calls = []
    fake_requests = types.ModuleType("requests")

    def _fake_get(url, timeout=None):
        seen_calls.append((url, timeout))
        return types.SimpleNamespace(
            text="<html><head><title>Demo</title></head><body><p>Hello</p></body></html>",
            raise_for_status=lambda: None,
        )

    fake_requests.get = _fake_get  # type: ignore[attr-defined]

    with patch.dict(sys.modules, {"requests": fake_requests}):
        facts = extract_facts_from_url(
            "https://example.com/article",
            lambda soup: [{"fact_statement": soup.find("p").text}],
            reliability=ReliabilityRating.LIKELY_TRUE,
            timeout=7.5,
        )

    assert seen_calls == [("https://example.com/article", 7.5)]
    assert len(facts) == 1
    assert facts[0]["source_title"] == "Demo"
    assert facts[0]["reliability_rating"] == ReliabilityRating.LIKELY_TRUE
