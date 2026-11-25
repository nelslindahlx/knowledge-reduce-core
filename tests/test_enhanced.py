"""Tests for EnhancedKnowledgeGraph caching, batching, and serialization."""
from datetime import datetime
import json

import pytest

from knowledge_graph_pkg.core import ReliabilityRating
from knowledge_graph_pkg.enhanced import EnhancedKnowledgeGraph


def _sample_kwargs(statement="Earth orbits the Sun.", category="Astronomy", rating=ReliabilityRating.VERIFIED, usage=3):
    now = datetime(2026, 1, 1, 12, 0, 0)
    return {
        "fact_statement": statement,
        "category": category,
        "tags": ["space", "orbit"],
        "date_recorded": now,
        "last_updated": now,
        "reliability_rating": rating,
        "source_id": "src1",
        "source_title": "Science",
        "author_creator": "Author",
        "publication_date": now,
        "url_reference": "https://example.com",
        "related_facts": [],
        "contextual_notes": "orbital mechanics",
        "access_level": "public",
        "usage_count": usage,
    }


def test_per_instance_cache_toggle_and_invalidation():
    ekg = EnhancedKnowledgeGraph(cache_enabled=False, auto_save_interval=0)
    ekg.add_fact("f1", **_sample_kwargs())

    # With cache_enabled=False, caches remain empty and updates are seen immediately.
    assert ekg.get_facts_by_category("Astronomy") == ["f1"]
    ekg.update_fact("f1", category="Physics")
    assert ekg.get_facts_by_category("Astronomy") == []
    assert ekg.get_facts_by_category("Physics") == ["f1"]

    # Enable cache and verify hits and invalidation on update/delete.
    ekg.cache_enabled = True
    assert ekg.get_fact("f1")["category"] == "Physics"
    assert "f1" in ekg._fact_cache
    assert ekg.get_facts_by_reliability(ReliabilityRating.LIKELY_TRUE) == ["f1"]

    ekg.delete_fact("f1")
    assert ekg.get_facts_by_category("Physics") == []
    assert ekg._fact_cache == {}


def test_batch_add_does_not_mutate_input_and_tracks_changes():
    ekg = EnhancedKnowledgeGraph(auto_save_interval=0)
    items = [
        {"fact_id": "f1", **_sample_kwargs("Fact one.")},
        {"fact_id": "f2", **_sample_kwargs("Fact two.", category="Physics", rating="LIKELY_TRUE", usage=1)},
    ]
    snapshot = json.loads(json.dumps(items, default=str))
    added = ekg.batch_add_facts(items)
    assert added == 2
    assert items[0]["fact_id"] == snapshot[0]["fact_id"]
    assert len(ekg.get_changes()) == 2
    hits = ekg.search_facts("Fact two")
    assert hits and hits[0][0] == "f2"


def test_save_and_load_roundtrip_and_auto_save_dir(tmp_path):
    ekg = EnhancedKnowledgeGraph(auto_save_interval=1, auto_save_dir=str(tmp_path))
    ekg._last_save = 0.0  # force auto-save trigger inside tmp_path
    ekg.add_fact("f1", **_sample_kwargs())
    autosaved = list(tmp_path.glob("autosave_knowledge_graph_*.json"))
    assert len(autosaved) == 1

    out = tmp_path / "export.json"
    ekg.save_to_file(str(out))
    assert ekg.get_changes() == []

    reloaded = EnhancedKnowledgeGraph(auto_save_interval=0)
    count = reloaded.load_from_file(str(out))
    assert count == 1
    fact = reloaded.get_fact("f1")
    assert fact["fact_statement"] == "Earth orbits the Sun."
    assert fact["tags"] == "space, orbit"
    assert fact["reliability_rating"] == ReliabilityRating.VERIFIED

    # Adding missing fact deletion raises ValueError cleanly.
    with pytest.raises(ValueError, match="not found"):
        reloaded.delete_fact("does_not_exist")
