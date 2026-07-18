"""
Tests for ModelReduce probe templates (Session 1).
"""
from knowledge_graph_pkg.probe_templates import (
    PROMPT_TEMPLATES, render_prompt, generate_probes, PROMPT_TYPES,
)


def test_all_prompt_types_present():
    assert set(PROMPT_TYPES) == {"entity", "relation", "concept", "list", "negative"}
    for t in PROMPT_TYPES:
        assert t in PROMPT_TEMPLATES


def test_render_entity_prompt():
    p = render_prompt("entity", domain="biochemistry", entity="Mitochondria")
    assert "Mitochondria" in p and "biochemistry" in p


def test_render_relation_prompt_needs_two_entities():
    p = render_prompt("relation", domain="physics", e1="mass", e2="energy")
    assert "mass" in p and "energy" in p and "physics" in p


def test_render_unknown_type_raises():
    try:
        render_prompt("bogus", domain="x")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_generate_probes_count_and_shape():
    probes = generate_probes(
        domain="biochemistry",
        entities=["Mitochondria", "DNA polymerase", "Glucose"],
        n_prompts=10,
        seed=42,
    )
    assert len(probes) == 10
    for pr in probes:
        assert pr["domain"] == "biochemistry"
        assert pr["prompt_type"] in PROMPT_TYPES
        assert isinstance(pr["prompt"], str) and pr["prompt"]


def test_generate_probes_deterministic():
    a = generate_probes("physics", ["mass", "energy", "force"], n_prompts=8, seed=7)
    b = generate_probes("physics", ["mass", "energy", "force"], n_prompts=8, seed=7)
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b]


def test_generate_probes_without_entities_still_works():
    # list/concept/negative templates can run on the domain alone
    probes = generate_probes("law", entities=[], n_prompts=5, seed=1)
    assert len(probes) == 5
