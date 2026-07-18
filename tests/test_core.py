"""
Characterization tests for the core KnowledgeGraph — establishes a safety net
of currently-correct behavior so refactors elsewhere can't silently break it.
"""
from datetime import datetime
import pytest
from knowledge_graph_pkg import KnowledgeGraph, ReliabilityRating


def make_kg_with_fact(fact_id="f1", rating=ReliabilityRating.VERIFIED, usage=10):
    kg = KnowledgeGraph()
    kg.add_fact(
        fact_id=fact_id,
        fact_statement="The Earth is round.",
        category="Astronomy",
        tags=["earth"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=rating,
        source_id="src",
        source_title="Title",
        author_creator="Author",
        publication_date=datetime.now(),
        url_reference="http://example.com",
        related_facts=[],
        contextual_notes="notes",
        access_level="public",
        usage_count=usage,
    )
    return kg


def test_add_and_get_fact_roundtrip():
    kg = make_kg_with_fact()
    fact = kg.get_fact("f1")
    assert fact["fact_statement"] == "The Earth is round."
    assert fact["reliability_rating"] == ReliabilityRating.VERIFIED


def test_quality_score_is_computed_and_positive():
    kg = make_kg_with_fact()
    fact = kg.get_fact("f1")
    assert isinstance(fact["quality_score"], (int, float))
    assert fact["quality_score"] > 0


def test_higher_reliability_yields_higher_quality_score():
    kg_low = make_kg_with_fact(rating=ReliabilityRating.POSSIBLY_TRUE, usage=10)
    kg_high = make_kg_with_fact(rating=ReliabilityRating.VERIFIED, usage=10)
    assert kg_high.get_fact("f1")["quality_score"] > kg_low.get_fact("f1")["quality_score"]


def test_update_fact_changes_value():
    kg = make_kg_with_fact()
    kg.update_fact("f1", usage_count=99)
    assert kg.get_fact("f1")["usage_count"] == 99
