"""
Tests for ModelDrop + probe-output -> fact conversion (Session 2).
"""

from knowledge_graph_pkg.model_drop import (
    ModelDrop,
    _safe,
    model_content_hash,
    model_fact_statement,
    probe_output_to_facts,
    probe_outputs_to_facts,
)
from knowledge_graph_pkg.store import KnowledgeStore


def _probe_output(
    model="qwen2.5:14b",
    backend="ollama",
    domain="biochemistry",
    prompt="State a fact.",
    prompt_type="entity",
    facts=None,
    gen_config=None,
):
    po = {
        "model": model,
        "backend": backend,
        "domain": domain,
        "prompt": prompt,
        "prompt_type": prompt_type,
        "structured_response": {
            "facts": facts
            if facts is not None
            else [
                {
                    "subject": "Mitochondria",
                    "predicate": "produce",
                    "object": "ATP",
                    "context_or_qualifier": "oxidative phosphorylation",
                    "confidence": 0.95,
                },
            ]
        },
    }
    if gen_config is not None:
        po["gen_config"] = gen_config
    return po


def test_model_fact_statement_humanizes_predicate():
    s = model_fact_statement("Marie Curie", "born_in", "Warsaw")
    assert s == "Marie Curie born in Warsaw."


def test_model_fact_statement_appends_qualifier():
    s = model_fact_statement(
        "Mitochondria", "produce", "ATP", qualifier="oxidative phosphorylation"
    )
    assert s == "Mitochondria produce ATP (oxidative phosphorylation)."


def test_model_fact_statement_empty_is_empty():
    assert model_fact_statement("", "", "") == ""


def test_probe_output_to_facts_basic():
    facts = probe_output_to_facts(_probe_output())
    assert len(facts) == 1
    f = facts[0]
    assert f["subject"] == "Mitochondria"
    assert f["predicate"] == "produce"
    assert f["object"] == "ATP"
    assert f["reliability_rating"] == "POSSIBLY_TRUE"
    assert f["category"] == "Biochemistry"
    assert f["fact_statement"].startswith("Mitochondria produce ATP")
    prov = f["model_provenance"]
    assert prov["model"] == "qwen2.5:14b"
    assert prov["backend"] == "ollama"
    assert prov["confidence"] == 0.95
    assert f["question"] and f["answer"]


def test_probe_output_skips_incomplete_triples_and_malformed_shapes():
    po = _probe_output(
        facts=[
            {"subject": "X", "predicate": "", "object": "Y"},
            {"subject": "", "predicate": "is", "object": "Z"},
            "not-a-dict",
            {"subject": "A", "predicate": "is", "object": "B"},
        ]
    )
    facts = probe_output_to_facts(po)
    assert len(facts) == 1
    assert facts[0]["subject"] == "A"
    assert probe_output_to_facts({"structured_response": ["not", "a", "dict"]}) == []


def test_probe_output_skips_degenerate_domain_subject():
    po = _probe_output(
        domain="biochemistry",
        facts=[
            {
                "subject": "Biochemistry",
                "predicate": "common misconception",
                "object": "Biochemistry only deals with living organisms",
            },
            {"subject": "Mitochondria", "predicate": "produce", "object": "ATP"},
        ],
    )
    facts = probe_output_to_facts(po)
    subjects = [f["subject"] for f in facts]
    assert "Mitochondria" in subjects
    assert "Biochemistry" not in subjects


def test_probe_output_skips_object_repeating_subject_but_keeps_compounds():
    po = _probe_output(
        domain="biochemistry",
        facts=[
            {
                "subject": "Chemistry",
                "predicate": "is",
                "object": "Chemistry is a dry descriptive science",
            },
            {
                "subject": "DNA",
                "predicate": "synthesized by",
                "object": "DNA polymerase",
            },
            {
                "subject": "Water",
                "predicate": "forms",
                "object": "water vapor",
            },
        ],
    )
    facts = probe_output_to_facts(po)
    pairs = {(f["subject"], f["object"]) for f in facts}
    assert ("DNA", "DNA polymerase") in pairs
    assert ("Water", "water vapor") in pairs
    assert not any(f["subject"] == "Chemistry" for f in facts)


def test_confidence_affects_quality_score():
    hi = probe_output_to_facts(
        _probe_output(
            facts=[{"subject": "A", "predicate": "is", "object": "B", "confidence": 1.0}]
        )
    )[0]
    lo = probe_output_to_facts(
        _probe_output(
            facts=[{"subject": "A", "predicate": "is", "object": "B", "confidence": 0.0}]
        )
    )[0]
    assert hi["quality_score"] > lo["quality_score"]


def test_probe_outputs_to_facts_flattens():
    facts = probe_outputs_to_facts([_probe_output(), _probe_output()])
    assert len(facts) == 2


def test_model_content_hash_differs_by_model():
    prompts = ["p1", "p2"]
    h1 = model_content_hash("model-a", "physics", prompts)
    h2 = model_content_hash("model-b", "physics", prompts)
    assert h1 != h2


def test_model_content_hash_stable():
    prompts = ["p1", "p2"]
    assert model_content_hash("m", "d", prompts) == model_content_hash("m", "d", prompts)


def test_modeldrop_from_probe_outputs_and_domain_sanitization():
    outs = [
        _probe_output(prompt="p1", gen_config={"temperature": 0.2}),
        _probe_output(prompt="p2"),
    ]
    drop = ModelDrop.from_probe_outputs(
        "qwen2.5:14b", "../../evil/domain", outs, backend="ollama"
    )
    assert drop.engine == "model-probe"
    assert drop.source == "qwen2.5:14b"
    assert len(drop.facts) == 2
    assert drop.model_provenance["model"] == "qwen2.5:14b"
    assert drop.model_provenance["n_prompts"] == 2
    assert drop.model_provenance["gen_config"] == {"temperature": 0.2}
    assert ".." not in drop.drop_id and "/" not in drop.drop_id
    assert ".." not in _safe("../../bad")


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
    drop = ModelDrop.from_probe_outputs(
        "qwen2.5:14b", "biochemistry", [_probe_output()], backend="ollama"
    )
    store.write_drop(drop)
    reopened = KnowledgeStore(str(tmp_path / "store"))
    facts = list(reopened.iter_facts())
    assert len(facts) == 1
    assert facts[0]["subject"] == "Mitochondria"
    assert facts[0]["model_provenance"]["model"] == "qwen2.5:14b"


def test_modeldrop_summary_carries_model():
    drop = ModelDrop.from_probe_outputs(
        "phi4:latest", "law", [_probe_output(model="phi4:latest", domain="law")]
    )
    s = drop.summary()
    assert s["model"] == "phi4:latest"
    assert s["domain"] == "law"
