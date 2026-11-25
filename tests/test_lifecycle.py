"""
Tests for corpus lifecycle operations (Session 11).

Because each drop stores its raw source_text, the corpus can be improved
retroactively without re-fetching:
- re-extract stored sources with a (possibly better) engine -> new drop versions
- promote reliability for facts corroborated across multiple sources
- flag contradictions (same subject+predicate, different object)
"""
import pytest

from knowledge_graph_pkg.store import KnowledgeStore, Drop, content_hash
from knowledge_graph_pkg.lifecycle import (
    promote_reliability, find_contradictions, reextract_store,
)


def _f(stmt, subj, pred, obj, rel="LIKELY_TRUE", q=30):
    return {"fact_statement": stmt, "subject": subj, "predicate": pred, "object": obj,
            "question": f"What about {subj}?", "answer": obj,
            "reliability_rating": rel, "quality_score": q, "category": "General"}


@pytest.fixture
def store(tmp_path):
    return KnowledgeStore(str(tmp_path / "store"))


# ---------- reliability promotion ----------

def test_promote_reliability_across_sources(store):
    # Same fact statement from two DIFFERENT sources -> should be promoted.
    store.write_drop(Drop("d1", "a.txt", content_hash("a"),
                          [_f("Water boils at 100C.", "Water", "boils_at", "100C",
                              rel="LIKELY_TRUE")], "svo", "standard", False))
    store.write_drop(Drop("d2", "b.txt", content_hash("b"),
                          [_f("Water boils at 100C.", "Water", "boils_at", "100C",
                              rel="LIKELY_TRUE")], "svo", "standard", False))
    promotions = promote_reliability(store)
    # the corroborated statement should be flagged for promotion
    assert any(p["statement"] == "Water boils at 100C." and p["sources"] == 2
               for p in promotions)
    assert any(p["new_reliability"] == "VERIFIED" for p in promotions)


def test_no_promotion_for_single_source(store):
    store.write_drop(Drop("d1", "a.txt", content_hash("a"),
                          [_f("Only once stated.", "X", "is", "unique")],
                          "svo", "standard", False))
    assert promote_reliability(store) == []


# ---------- contradiction flagging ----------

def test_find_contradictions(store):
    store.write_drop(Drop("d1", "a.txt", content_hash("a"),
                          [_f("Paris is in France.", "Paris", "located_in", "France")],
                          "svo", "standard", False))
    store.write_drop(Drop("d2", "b.txt", content_hash("b"),
                          [_f("Paris is in Texas.", "Paris", "located_in", "Texas")],
                          "svo", "standard", False))
    conflicts = find_contradictions(store)
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c["subject"] == "Paris" and c["predicate"] == "located_in"
    assert set(c["objects"]) == {"France", "Texas"}


def test_no_contradiction_when_objects_agree(store):
    store.write_drop(Drop("d1", "a.txt", content_hash("a"),
                          [_f("Paris in France.", "Paris", "located_in", "France")],
                          "svo", "standard", False))
    store.write_drop(Drop("d2", "b.txt", content_hash("b"),
                          [_f("Paris in France.", "Paris", "located_in", "France")],
                          "svo", "standard", False))
    assert find_contradictions(store) == []


# ---------- re-extraction ----------

def test_reextract_creates_new_drop_versions(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    # Seed a drop that carries its raw source_text.
    drop = Drop("orig-abc", "doc.txt", content_hash("Marie Curie discovered radium."),
                facts=[_f("stale", "Marie Curie", "discovered", "radium")],
                engine="svo", filter_name="standard", coref=False,
                source_text="Marie Curie discovered radium. Robert Putnam wrote Bowling Alone.")
    store.write_drop(drop)

    n_before = len(store.list_drops())
    result = reextract_store(store, engine="svo")
    # a new drop version was written for the re-extracted source
    assert result["reextracted"] >= 1
    reopened = KnowledgeStore(str(tmp_path / "store"))
    assert len(reopened.list_drops()) > n_before
    # the new drop is marked as a reextraction in its drop_id or meta
    assert any("reextract" in d["drop_id"] or d.get("num_facts", 0) >= 1
               for d in reopened.list_drops())


def test_reextract_skips_drops_without_source_text(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    store.write_drop(Drop("nosrc", "x.txt", content_hash("x"),
                          [_f("a", "A", "is", "b")], "svo", "standard", False,
                          source_text=None))
    result = reextract_store(store, engine="svo")
    assert result["skipped"] >= 1


def test_reextract_does_not_multiply_on_repeat_runs_and_skips_missing_shard(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    store.write_drop(
        Drop(
            "orig",
            "doc.txt",
            content_hash("doc"),
            facts=[_f("Marie Curie discovered radium.", "Marie Curie", "discovered", "radium")],
            source_text="Marie Curie discovered radium.",
        )
    )
    first = reextract_store(store, engine="svo")
    assert first == {"reextracted": 1, "skipped": 0}
    second = reextract_store(store, engine="svo")
    assert second == {"reextracted": 1, "skipped": 0}
    # 1 original + 2 reextractions (not 1 + 1 + 2 = 4!)
    assert len(store.list_drops()) == 3
    drop_ids = [d["drop_id"] for d in store.list_drops()]
    assert "orig-reextract-svo" in drop_ids
    assert "orig-reextract-svo-v2" in drop_ids

    # Delete original shard on disk and verify missing shard is skipped gracefully.
    orig_entry = next(d for d in store.list_drops() if d["drop_id"] == "orig")
    (tmp_path / "store" / orig_entry["shard"]).unlink()
    third = reextract_store(store, engine="svo")
    assert third == {"reextracted": 0, "skipped": 1}


def test_promote_and_contradictions_normalize_case_and_none_values(store):
    store.write_drop(
        Drop("d1", "a.txt", content_hash("a"), [
            {"fact_statement": "Paris is in France.", "subject": "Paris", "predicate": "located_in",
             "object": "France", "reliability_rating": "likely_true"},
        ])
    )
    store.write_drop(
        Drop("d2", "b.txt", content_hash("b"), [
            {"fact_statement": "Paris is in France.", "subject": "paris", "predicate": "LOCATED_IN",
             "object": "Texas", "reliability_rating": "unknown_tier"},
        ])
    )
    promotions = promote_reliability(store)
    assert len(promotions) == 1
    assert promotions[0]["old_reliability"] == "LIKELY_TRUE"
    assert promotions[0]["new_reliability"] == "VERIFIED"

    conflicts = find_contradictions(store)
    assert len(conflicts) == 1
    assert conflicts[0]["subject"] == "Paris"
    assert set(conflicts[0]["objects"]) == {"France", "Texas"}
