"""
Tests for ModelDrop + probe-output -> fact conversion (Session 2).

Session 1 emits structured probe outputs; this module turns them into
KnowledgeReduce facts and immutable model-provenance drops, so the existing
store/distill/lifecycle machinery applies. No Ollama needed -- we build probe
outputs by hand.
"""
import pytest

from knowledge_graph_pkg.model_drop import (
    ModelDrop, probe_output_to_facts, probe_outputs_to_facts,
    model_fact_statement, model_content_hash,
)
from knowledge_graph_pkg.store import KnowledgeStore, Drop


def _probe_output(model="qwen2.5:14b", backend="ollama", domain="biochemistry",
                  prompt="State a fact.", prompt_type="entity", facts=None):
    return {
        "model": model, "backend": backend, "domain": domain,
        "prompt": prompt, "prompt_type": prompt_type,
        "structured_response": {"facts": facts if facts is not None else [
            {"subject": "Mitochondria", "predicate": "produce", "object": "ATP",
             "context_or_qualifier": "oxidative phosphorylation", "confidence": 0.95},
        ]},
    }


# ---------- statement rendering ----------

def test_model_fact_statement_humanizes_predicate():
    s = model_fact_statement("Marie Curie", "born_in", "Warsaw")
    assert s == "Marie Curie born in Warsaw."


def test_model_fact_statement_appends_qualifier():
    s = model_fact_statement("Mitochondria", "produce", "ATP",
                             qualifier="oxidative phosphorylation")
    assert s == "Mitochondria produce ATP (oxidative phosphorylation)."


def test_model_fact_statement_empty_is_empty():
    assert model_fact_statement("", "", "") == ""


# ---------- probe output -> facts ----------

def test_probe_output_to_facts_basic():
    facts = probe_output_to_facts(_probe_output())
    assert len(facts) == 1
    f = facts[0]
    assert f["subject"] == "Mitochondria"
    assert f["predicate"] == "produce"
    assert f["object"] == "ATP"
    assert f["reliability_rating"] == "POSSIBLY_TRUE"  # lone model claim
    assert f["category"] == "Biochemistry"
    assert f["fact_statement"].startswith("Mitochondria produce ATP")
    # provenance recorded
    prov = f["model_provenance"]
    assert prov["model"] == "qwen2.5:14b"
    assert prov["backend"] == "ollama"
    assert prov["confidence"] == 0.95
    # distillation-ready Q/A present
    assert f["question"] and f["answer"]


def test_probe_output_skips_incomplete_triples():
    po = _probe_output(facts=[
        {"subject": "X", "predicate": "", "object": "Y"},          # no predicate
        {"subject": "", "predicate": "is", "object": "Z"},          # no subject
        {"subject": "A", "predicate": "is", "object": "B"},         # ok
    ])
    facts = probe_output_to_facts(po)
    assert len(facts) == 1
    assert facts[0]["subject"] == "A"


def test_probe_output_skips_degenerate_domain_subject():
    # A model that just echoes the domain as the subject ("Biochemistry
    # common misconception ...") is noise, not a fact. Reject it.
    po = _probe_output(domain="biochemistry", facts=[
        {"subject": "Biochemistry", "predicate": "common misconception",
         "object": "Biochemistry only deals with living organisms"},
        {"subject": "Mitochondria", "predicate": "produce", "object": "ATP"},  # ok
    ])
    facts = probe_output_to_facts(po)
    subjects = [f["subject"] for f in facts]
    assert "Mitochondria" in subjects
    assert "Biochemistry" not in subjects  # domain-as-subject rejected


def test_probe_output_skips_object_repeating_subject():
    # "Biochemistry ... Biochemistry only deals ..." — subject echoed in object
    po = _probe_output(domain="chemistry", facts=[
        {"subject": "Chemistry", "predicate": "is", "object": "Chemistry is a science"},
    ])
    assert probe_output_to_facts(po) == []


def test_confidence_affects_quality_score():
    hi = probe_output_to_facts(_probe_output(facts=[
        {"subject": "A", "predicate": "is", "object": "B", "confidence": 1.0}]))[0]
    lo = probe_output_to_facts(_probe_output(facts=[
        {"subject": "A", "predicate": "is", "object": "B", "confidence": 0.0}]))[0]
    assert hi["quality_score"] > lo["quality_score"]


def test_probe_outputs_to_facts_flattens():
    facts = probe_outputs_to_facts([_probe_output(), _probe_output()])
    assert len(facts) == 2


# ---------- content hash folds in model identity ----------

def test_model_content_hash_differs_by_model():
    prompts = ["p1", "p2"]
    h1 = model_content_hash("model-a", "physics", prompts)
    h2 = model_content_hash("model-b", "physics", prompts)
    assert h1 != h2


def test_model_content_hash_stable():
    prompts = ["p1", "p2"]
    assert model_content_hash("m", "d", prompts) == model_content_hash("m", "d", prompts)


# ---------- ModelDrop ----------

def test_modeldrop_from_probe_outputs():
    outs = [_probe_output(prompt="p1"), _probe_output(prompt="p2")]
    drop = ModelDrop.from_probe_outputs("qwen2.5:14b", "biochemistry", outs,
                                        backend="ollama")
    assert drop.engine == "model-probe"
    assert drop.source == "qwen2.5:14b"
    assert len(drop.facts) == 2
    assert drop.model_provenance["model"] == "qwen2.5:14b"
    assert drop.model_provenance["n_prompts"] == 2


def test_modeldrop_roundtrips_through_dict():
    drop = ModelDrop.from_probe_outputs("m", "physics", [_probe_output()])
    d = drop.to_dict()
    assert d["model_provenance"]["model"] == "m"
    restored = ModelDrop.from_dict(d)
    assert restored.model_provenance["model"] == "m"
    assert restored.engine == "model-probe"
    assert len(restored.facts) == len(drop.facts)


def test_modeldrop_writes_to_store_and_iterates(tmp_path):
    store = KnowledgeStore(str(tmp_path / "store"))
    drop = ModelDrop.from_probe_outputs("qwen2.5:14b", "biochemistry",
                                        [_probe_output()], backend="ollama")
    store.write_drop(drop)
    # reopen and confirm the fact (with provenance) survived the round-trip
    reopened = KnowledgeStore(str(tmp_path / "store"))
    facts = list(reopened.iter_facts())
    assert len(facts) == 1
    assert facts[0]["subject"] == "Mitochondria"
    assert facts[0]["model_provenance"]["model"] == "qwen2.5:14b"


def test_modeldrop_summary_carries_model(tmp_path):
    drop = ModelDrop.from_probe_outputs("phi4:latest", "law", [_probe_output(
        model="phi4:latest", domain="law")])
    s = drop.summary()
    assert s["model"] == "phi4:latest"
    assert s["domain"] == "law"
