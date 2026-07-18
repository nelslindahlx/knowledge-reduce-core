"""
Tests for ModelReduce schemas (Session 1).

schemas.py defines:
- PROBE_OUTPUT_SCHEMA: a JSON Schema dict passed to Ollama's `format`
  parameter to force structured fact output during generation.
- ProbeFact / ProbeOutput: pydantic models for validating/parsing the
  structured responses.
"""
import pytest

pytest.importorskip("pydantic")

from knowledge_graph_pkg.schemas import (
    PROBE_OUTPUT_SCHEMA, ProbeFact, ProbeOutput,
)


def test_probe_output_schema_shape():
    s = PROBE_OUTPUT_SCHEMA
    assert s["type"] == "object"
    assert "facts" in s["properties"]
    items = s["properties"]["facts"]["items"]
    # required SVO triple keys present
    for key in ("subject", "predicate", "object"):
        assert key in items["properties"]
    assert set(items["required"]) >= {"subject", "predicate", "object"}


def test_probe_fact_validates_minimal():
    f = ProbeFact(subject="Mitochondria", predicate="produce", object="ATP")
    assert f.subject == "Mitochondria"
    assert f.confidence is None or 0.0 <= f.confidence <= 1.0


def test_probe_fact_confidence_bounds():
    with pytest.raises(Exception):
        ProbeFact(subject="X", predicate="y", object="z", confidence=1.5)


def test_probe_output_parses_facts_list():
    out = ProbeOutput(facts=[
        {"subject": "DNA polymerase", "predicate": "synthesizes", "object": "DNA",
         "context_or_qualifier": "5' to 3'", "confidence": 0.9},
        {"subject": "Glucose", "predicate": "fuels", "object": "brain"},
    ])
    assert len(out.facts) == 2
    assert out.facts[0].context_or_qualifier == "5' to 3'"


def test_probe_output_from_json_string():
    raw = '{"facts": [{"subject": "Water", "predicate": "boils_at", "object": "100C"}]}'
    out = ProbeOutput.model_validate_json(raw)
    assert out.facts[0].object == "100C"


def test_probe_output_empty_facts_ok():
    out = ProbeOutput(facts=[])
    assert out.facts == []
