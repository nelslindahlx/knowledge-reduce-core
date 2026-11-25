"""
Tests for the knowledge-drop store (Session 7).

A "drop" is one immutable shard produced by one ingestion effort: a set of
facts plus provenance (source, content hash, timestamp) and lineage
(extractor engine, filter settings, schema version). The store appends
drops as JSONL shards and tracks them in a manifest.
"""
import json
from pathlib import Path

import pytest

from knowledge_graph_pkg.store import (
    Drop, KnowledgeStore, SCHEMA_VERSION, content_hash,
)


def _facts():
    return [
        {"fact_statement": "Robert Putnam wrote Bowling Alone.",
         "subject": "Robert Putnam", "predicate": "wrote", "object": "Bowling Alone",
         "question": "What did Robert Putnam write?", "answer": "Bowling Alone",
         "reliability_rating": "LIKELY_TRUE", "quality_score": 32,
         "category": "General", "tags": "person,wrote"},
        {"fact_statement": "Marie Curie was born in Warsaw.",
         "subject": "Marie Curie", "predicate": "born_in", "object": "Warsaw",
         "question": "Where was Marie Curie born?", "answer": "Warsaw",
         "reliability_rating": "LIKELY_TRUE", "quality_score": 30,
         "category": "General", "tags": "person,born_in"},
    ]


# ---------- content hashing ----------

def test_content_hash_is_stable_and_distinct():
    a = content_hash("hello world")
    assert a == content_hash("hello world")
    assert a != content_hash("hello worlds")
    assert len(a) == 64  # sha256 hex


# ---------- Drop dataclass ----------

def test_drop_roundtrips_through_dict():
    drop = Drop(
        drop_id="civic-abc123",
        source="data/book.txt",
        source_hash=content_hash("raw text"),
        facts=_facts(),
        engine="svo",
        filter_name="standard",
        coref=True,
    )
    d = drop.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["num_facts"] == 2
    assert "created_at" in d
    back = Drop.from_dict(d)
    assert back.drop_id == "civic-abc123"
    assert back.facts == _facts()


# ---------- store: write + read ----------

def test_write_drop_creates_shard_and_manifest(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    drop = Drop(drop_id="d1", source="s.txt", source_hash=content_hash("x"),
                facts=_facts(), engine="svo", filter_name="standard", coref=False)
    shard_path = store.write_drop(drop)
    assert Path(shard_path).exists()
    assert (tmp_path / "store" / "manifest.json").exists()


def test_read_back_drop(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    drop = Drop(drop_id="d1", source="s.txt", source_hash=content_hash("x"),
                facts=_facts(), engine="svo", filter_name="standard", coref=False)
    store.write_drop(drop)

    reopened = KnowledgeStore(str(tmp_path / "store"))
    drops = reopened.list_drops()
    assert len(drops) == 1
    assert drops[0]["drop_id"] == "d1"
    assert drops[0]["num_facts"] == 2

    all_facts = list(reopened.iter_facts())
    assert len(all_facts) == 2
    assert all_facts[0]["fact_statement"].startswith("Robert Putnam")


# ---------- store: append-only, multiple drops ----------

def test_multiple_drops_accumulate(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    store.write_drop(Drop("d1", "a.txt", content_hash("a"), _facts(), "svo", "standard", False))
    store.write_drop(Drop("d2", "b.txt", content_hash("b"), _facts(), "svo", "standard", False))
    assert len(store.list_drops()) == 2
    assert len(list(store.iter_facts())) == 4
    assert store.stats()["total_drops"] == 2
    assert store.stats()["total_facts"] == 4


# ---------- store: idempotency helper ----------

def test_has_source_hash_detects_duplicate_effort(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    h = content_hash("same source text")
    assert store.has_source_hash(h) is False
    store.write_drop(Drop("d1", "a.txt", h, _facts(), "svo", "standard", False))
    assert store.has_source_hash(h) is True


def test_manifest_is_valid_json(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    store.write_drop(Drop("d1", "a.txt", content_hash("a"), _facts(), "svo", "standard", False))
    manifest = json.loads((tmp_path / "store" / "manifest.json").read_text())
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert len(manifest["drops"]) == 1


def test_drop_id_and_partition_traversal_sanitized_and_non_overwriting(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    d1 = Drop(
        drop_id="../../escape",
        source="a.txt",
        source_hash=content_hash("a"),
        facts=[{**_facts()[0], "record": "bogus_override"}],
        created_at="../bad-date",
    )
    shard1 = Path(store.write_drop(d1)).resolve()
    assert str(shard1).startswith(str((tmp_path / "store" / "shards").resolve()))

    # Re-using same drop_id appends -v2 without overwriting shard1 on disk.
    d2 = Drop(
        drop_id="../../escape",
        source="b.txt",
        source_hash=content_hash("b"),
        facts=[_facts()[1]],
        created_at="../bad-date",
    )
    shard2 = Path(store.write_drop(d2)).resolve()
    assert shard2 != shard1
    assert shard1.exists() and shard2.exists()

    facts = list(store.iter_facts())
    assert len(facts) == 2


def test_iter_facts_ignores_tampered_manifest_shard_and_stats_handles_none_source(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    store.write_drop(Drop("d1", None, content_hash("x"), _facts()))  # type: ignore[arg-type]
    assert store.stats()["sources"] == []

    # Inject a traversal entry in manifest and verify iter_facts ignores it safely.
    manifest_path = tmp_path / "store" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["drops"].append({"drop_id": "evil", "source": "evil", "shard": "../../etc/passwd"})
    manifest_path.write_text(json.dumps(manifest))

    reopened = KnowledgeStore(str(tmp_path / "store"))
    assert len(list(reopened.iter_facts())) == 2
