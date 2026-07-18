"""
Tests for Q&A generation from relations.

Q&A generation turns a structured relation triple into a natural question
and answer, so distilled facts become high-quality supervised training
pairs (e.g. born_in -> "Where was X born?" / "Warsaw").
"""
import pytest

from knowledge_graph_pkg import KnowledgeGraph, SemanticKnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.qa import QAGenerator


@pytest.fixture
def qa():
    return QAGenerator()


def _rel(subject, predicate, object_, s_type="PERSON", o_type="ENTITY"):
    return {"subject": subject, "subject_type": s_type,
            "predicate": predicate, "object": object_, "object_type": o_type}


def test_born_in_question(qa):
    q, a = qa.generate(_rel("Marie Curie", "born_in", "Warsaw"))
    assert q == "Where was Marie Curie born?"
    assert a == "Warsaw"


def test_discovered_question(qa):
    q, a = qa.generate(_rel("Marie Curie", "discovered", "radium"))
    assert q == "What did Marie Curie discover?"
    assert a == "radium"


def test_works_for_question(qa):
    q, a = qa.generate(_rel("Alice", "works_for", "Acme Corp", o_type="ORGANIZATION"))
    assert q == "Who or what does Alice work for?"
    assert a == "Acme Corp"


def test_unknown_predicate_has_generic_but_valid_question(qa):
    q, a = qa.generate(_rel("X", "some_made_up_rel", "Y"))
    assert q.endswith("?")
    assert a == "Y"
    assert "_" not in q  # predicate underscores must not leak into the question


def test_generate_returns_tuple(qa):
    result = qa.generate(_rel("A", "born_in", "B"))
    assert isinstance(result, tuple) and len(result) == 2


# ---------- integration: facts carry Q&A + triple ----------

def test_created_facts_store_question_and_answer():
    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    ids = skg.create_facts_from_text(
        "Marie Curie discovered radium.",
        source_id="demo",
        reliability=ReliabilityRating.LIKELY_TRUE,
    )
    assert ids
    node = kg.get_fact(ids[0])
    assert node.get("question") == "What did Marie Curie discover?"
    assert node.get("answer") == "radium"
    # structured triple preserved for downstream use
    assert node.get("subject") == "Marie Curie"
    assert node.get("predicate") == "discovered"
    assert node.get("object") == "radium"
