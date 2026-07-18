"""
Tests for the SQLite catalog / index (Session 9).

The catalog builds a queryable SQLite index over all facts in a store, so
drops are findable later by source, reliability, category, or quality
without scanning every shard.
"""
import pytest

from knowledge_graph_pkg.store import KnowledgeStore, Drop, content_hash
from knowledge_graph_pkg.catalog import Catalog


def _drop(drop_id, source, facts):
    return Drop(drop_id=drop_id, source=source, source_hash=content_hash(source),
                facts=facts, engine="svo", filter_name="standard", coref=False)


@pytest.fixture
def store(tmp_path):
    s = KnowledgeStore(str(tmp_path / "store"))
    s.write_drop(_drop("d1", "a.txt", [
        {"fact_statement": "Marie Curie discovered radium.", "subject": "Marie Curie",
         "predicate": "discovered", "object": "radium", "question": "What did Marie Curie discover?",
         "answer": "radium", "reliability_rating": "VERIFIED", "quality_score": 90, "category": "Science"},
        {"fact_statement": "Change is inevitable.", "subject": "Change", "predicate": "is",
         "object": "inevitable", "question": "What is Change?", "answer": "inevitable",
         "reliability_rating": "LIKELY_TRUE", "quality_score": 30, "category": "General"},
    ]))
    s.write_drop(_drop("d2", "b.txt", [
        {"fact_statement": "Robert Putnam wrote Bowling Alone.", "subject": "Robert Putnam",
         "predicate": "wrote", "object": "Bowling Alone", "question": "What did Robert Putnam write?",
         "answer": "Bowling Alone", "reliability_rating": "LIKELY_TRUE", "quality_score": 50,
         "category": "General"},
    ]))
    return s


def test_build_index_counts_all_facts(tmp_path, store):
    cat = Catalog(str(tmp_path / "store" / "catalog.db"))
    n = cat.rebuild(store)
    assert n == 3
    assert cat.count() == 3


def test_query_by_source(tmp_path, store):
    cat = Catalog(str(tmp_path / "store" / "catalog.db"))
    cat.rebuild(store)
    rows = cat.query(source="a.txt")
    assert len(rows) == 2


def test_query_by_min_quality(tmp_path, store):
    cat = Catalog(str(tmp_path / "store" / "catalog.db"))
    cat.rebuild(store)
    rows = cat.query(min_quality=60)
    assert len(rows) == 1
    assert rows[0]["subject"] == "Marie Curie"


def test_query_by_reliability(tmp_path, store):
    cat = Catalog(str(tmp_path / "store" / "catalog.db"))
    cat.rebuild(store)
    rows = cat.query(reliability="VERIFIED")
    assert len(rows) == 1


def test_query_by_category(tmp_path, store):
    cat = Catalog(str(tmp_path / "store" / "catalog.db"))
    cat.rebuild(store)
    rows = cat.query(category="General")
    assert len(rows) == 2


def test_stats_summary(tmp_path, store):
    cat = Catalog(str(tmp_path / "store" / "catalog.db"))
    cat.rebuild(store)
    s = cat.stats()
    assert s["total_facts"] == 3
    assert s["total_drops"] == 2
    assert s["sources"] == 2
    assert "by_reliability" in s


def test_rebuild_is_idempotent(tmp_path, store):
    cat = Catalog(str(tmp_path / "store" / "catalog.db"))
    cat.rebuild(store)
    cat.rebuild(store)  # second rebuild should not double-count
    assert cat.count() == 3
