"""
Tests for ModelKnowledgeDistiller (Session 3).

The distiller reads model-derived facts, corroborates them across models
(reusing cross_model clustering), promotes reliability by distinct-model
agreement, filters/dedups/ranks, and serializes to chat/instruction/text
shards with provenance metadata. No Ollama needed.
"""
import json
import pytest

from knowledge_graph_pkg.model_distill import ModelKnowledgeDistiller
from knowledge_graph_pkg.model_drop import ModelDrop
from knowledge_graph_pkg.store import KnowledgeStore


def _fact(subject, predicate, obj, model, statement=None, category="Biochemistry",
          quality=22):
    return {
        "fact_statement": statement or f"{subject} {predicate} {obj}.",
        "subject": subject, "predicate": predicate, "object": obj,
        "category": category, "reliability_rating": "POSSIBLY_TRUE",
        "quality_score": quality,
        "question": f"State a fact about {subject}.",
        "answer": statement or f"{subject} {predicate} {obj}.",
        "model_provenance": {"model": model, "backend": "fake"},
    }


def _agreed(subject, predicate, obj, models, **kw):
    """Same fact emitted by several models -> a corroborated cluster."""
    return [_fact(subject, predicate, obj, m, **kw) for m in models]


# ---------- agreement-driven selection ----------

def test_two_model_agreement_promotes_to_likely_true():
    facts = _agreed("Mitochondria", "produce", "ATP", ["a", "b"])
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    sel = d.select_facts()
    assert len(sel) == 1
    assert sel[0]["reliability_rating"] == "LIKELY_TRUE"
    assert sel[0]["cross_model_agreement"] == 2
    assert set(sel[0]["source_models"]) == {"a", "b"}


def test_three_model_agreement_promotes_to_verified():
    facts = _agreed("Water", "boils at", "100C", ["a", "b", "c"])
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    sel = d.select_facts()
    assert sel[0]["reliability_rating"] == "VERIFIED"


def test_lone_claim_dropped_at_min_agreement_2():
    facts = [_fact("Solo", "is", "lonely", "only-model")]
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    assert d.select_facts() == []


def test_same_model_twice_does_not_corroborate():
    # one model emitting the same fact twice != 2-model agreement
    facts = _agreed("X", "is", "Y", ["solo", "solo"])
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    assert d.select_facts() == []


def test_min_reliability_verified_filters_2model():
    facts = _agreed("A", "is", "B", ["a", "b"])  # only LIKELY_TRUE
    d = ModelKnowledgeDistiller(facts, min_agreement=2, min_reliability="VERIFIED")
    assert d.select_facts() == []


# ---------- serializers carry provenance metadata ----------

def test_chat_jsonl_has_metadata():
    facts = _agreed("Mitochondria", "produce", "ATP", ["a", "b"])
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    rec = json.loads(d.to_chat_jsonl().splitlines()[0])
    assert "messages" in rec and "metadata" in rec
    assert rec["metadata"]["agreement"] == 2
    assert rec["metadata"]["reliability"] == "LIKELY_TRUE"
    assert set(rec["metadata"]["source_models"]) == {"a", "b"}


def test_instruction_jsonl_has_metadata():
    facts = _agreed("A", "is", "B", ["a", "b"])
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    rec = json.loads(d.to_instruction_jsonl().splitlines()[0])
    assert rec["instruction"] and rec["output"]
    assert rec["metadata"]["agreement"] == 2


def test_text_format_cites_models_and_reliability():
    facts = _agreed("Earth", "orbits", "Sun", ["a", "b", "c"])
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    line = d.to_text().splitlines()[0]
    assert "VERIFIED" in line
    assert "a" in line and "b" in line


# ---------- ranking, dedup, manifest ----------

def test_ranking_prefers_higher_agreement():
    facts = (_agreed("Verified", "is", "consensus", ["a", "b", "c"])
             + _agreed("Likely", "is", "corroborated", ["d", "e"]))
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    sel = d.select_facts()
    assert sel[0]["subject"] == "Verified"  # 3-model beats 2-model
    assert sel[1]["subject"] == "Likely"


def test_manifest_reports_tier_counts_and_models():
    facts = (_agreed("V", "is", "verified", ["a", "b", "c"])
             + _agreed("L", "is", "likely", ["a", "b"]))
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    m = d.manifest("biochem_v1")
    assert m["shard"] == "biochem_v1"
    assert m["verified"] == 1 and m["likely_true"] == 1
    assert set(m["models"]) == {"a", "b", "c"}
    assert m["facts"] == 2


def test_stats_reduction_ratio():
    facts = _agreed("A", "is", "B", ["a", "b"]) + [_fact("Lone", "is", "x", "z")]
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    s = d.stats()
    assert s["input_facts"] == 3
    assert s["selected_facts"] == 1
    assert 0 < s["reduction_ratio"] < 1


def test_distill_to_file_writes(tmp_path):
    facts = _agreed("A", "is", "B", ["a", "b"])
    d = ModelKnowledgeDistiller(facts, min_agreement=2)
    out = tmp_path / "shard.jsonl"
    n = d.distill_to_file(str(out), fmt="chat")
    assert n == 1
    assert out.exists() and out.read_text().strip()


def test_distill_to_file_rejects_bad_format(tmp_path):
    d = ModelKnowledgeDistiller([], min_agreement=2)
    with pytest.raises(ValueError):
        d.distill_to_file(str(tmp_path / "x"), fmt="parquet")


# ---------- from_store integration ----------

def _probe_output(model, domain, subject, predicate, obj):
    return {"model": model, "backend": "fake", "domain": domain,
            "prompt": "p", "prompt_type": "entity",
            "structured_response": {"facts": [
                {"subject": subject, "predicate": predicate, "object": obj,
                 "confidence": 0.9}]}}


def test_from_store_reads_model_facts_and_corroborates(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    # two models independently assert the same fact -> should corroborate
    store.write_drop(ModelDrop.from_probe_outputs(
        "model-a", "biochemistry",
        [_probe_output("model-a", "biochemistry", "Mitochondria", "produce", "ATP")]))
    store.write_drop(ModelDrop.from_probe_outputs(
        "model-b", "biochemistry",
        [_probe_output("model-b", "biochemistry", "Mitochondria", "produce", "ATP")]))
    d = ModelKnowledgeDistiller.from_store(store, min_agreement=2)
    sel = d.select_facts()
    assert len(sel) == 1
    assert sel[0]["cross_model_agreement"] == 2
    assert sel[0]["reliability_rating"] == "LIKELY_TRUE"


def test_from_store_ignores_non_model_facts(tmp_path):
    from knowledge_graph_pkg.store import Drop, content_hash
    store = KnowledgeStore(str(tmp_path / "store"))
    # a plain text-extracted drop (no model_provenance) must be ignored
    store.write_drop(Drop("d1", "doc.txt", content_hash("x"),
                          [{"fact_statement": "Plain fact.", "subject": "X",
                            "predicate": "is", "object": "Y",
                            "reliability_rating": "LIKELY_TRUE", "quality_score": 30}],
                          "svo", "standard", False))
    d = ModelKnowledgeDistiller.from_store(store, min_agreement=1)
    assert d.select_facts() == []
