"""
Tests for semantic entity & relation extraction.

These encode the DESIRED behavior. Several are expected to fail against the
original regex implementation (RED), documenting real bugs:
  - "Apple Inc" misclassified as PERSON
  - relations matched greedily across sentence boundaries
"""
import pytest
from knowledge_graph_pkg import KnowledgeGraph, SemanticKnowledgeGraph, ReliabilityRating


@pytest.fixture
def skg():
    return SemanticKnowledgeGraph(KnowledgeGraph())


def _types_for(entities, text):
    """Return the set of entity types assigned to a given surface string."""
    return {e["type"] for e in entities if e["text"].rstrip(".") == text}


# ---------- entity classification ----------

def test_person_detected(skg):
    ents = skg.extract_entities_from_text("Barack Obama was born in Hawaii.")
    assert "Barack Obama" in {e["text"] for e in ents}
    assert _types_for(ents, "Barack Obama") == {"PERSON"}


def test_org_with_inc_suffix_is_not_person(skg):
    """'Apple Inc' must be ORGANIZATION, never PERSON."""
    ents = skg.extract_entities_from_text("Apple Inc. is based in California.")
    apple_types = _types_for(ents, "Apple Inc")
    assert "ORGANIZATION" in apple_types
    assert "PERSON" not in apple_types


def test_no_duplicate_type_for_same_span(skg):
    """An entity span should not be emitted under two conflicting types."""
    ents = skg.extract_entities_from_text("Apple Inc. is based in California.")
    apple = [e for e in ents if e["text"].rstrip(".") == "Apple Inc"]
    assert len(apple) == 1


def test_org_match_does_not_cross_sentence(skg):
    """Org detection must not swallow the previous sentence's tail."""
    ents = skg.extract_entities_from_text("He was born in Hawaii. Apple Inc is big.")
    orgs = [e["text"] for e in ents if e["type"] == "ORGANIZATION"]
    assert any(o.rstrip(".") == "Apple Inc" for o in orgs)
    assert not any("Hawaii" in o for o in orgs)


# ---------- relation extraction ----------

def test_relation_does_not_cross_sentence_boundary(skg):
    """
    'Barack Obama was born in Hawaii. Apple Inc is in California.'
    The born_in relation must connect Obama->Hawaii, NOT Obama->Apple Inc.
    """
    text = "Barack Obama was born in Hawaii. Apple Inc is in California."
    rels = skg.extract_relations_from_text(text)
    born = [r for r in rels if r["predicate"] == "born_in"]
    assert born, "expected a born_in relation"
    for r in born:
        assert r["subject"].rstrip(".") == "Barack Obama"
        # object must stay within the first sentence
        assert "Apple" not in r["object"]


def test_subject_object_within_same_sentence(skg):
    text = "Barack Obama was born in Hawaii. Apple Inc is in California."
    rels = skg.extract_relations_from_text(text)
    assert rels, "expected at least one relation"
    for r in rels:
        # crude check: the relation's source text shouldn't span the period
        assert ". " not in r["text"].rstrip("."), f"relation crosses sentences: {r}"


# ---------- end-to-end fact creation ----------

def test_create_facts_from_text_runs(skg):
    ids = skg.create_facts_from_text(
        "Barack Obama was born in Hawaii.",
        source_id="demo",
        reliability=ReliabilityRating.UNVERIFIED,
    )
    assert isinstance(ids, list)
    assert len(ids) >= 1


def test_created_fact_statement_is_natural_language(skg):
    """
    Auto-created fact statements should read as natural English, not raw
    'subject predicate object' tuples with underscores. This makes the
    distilled output usable as training data.
    """
    ids = skg.create_facts_from_text(
        "Barack Obama was born in Hawaii.",
        source_id="demo",
        reliability=ReliabilityRating.LIKELY_TRUE,
    )
    statements = [skg.kg.get_fact(i)["fact_statement"] for i in ids]
    assert statements, "expected at least one created fact"
    for s in statements:
        assert "_" not in s, f"statement contains raw predicate underscore: {s!r}"
    # the born_in relation should verbalize to 'was born in'
    assert any("was born in" in s for s in statements)


def test_verbalize_relation_maps_predicates(skg):
    rel = {"subject": "Marie Curie", "subject_type": "PERSON",
           "predicate": "born_in", "object": "Warsaw", "object_type": "ENTITY"}
    assert skg.verbalize_relation(rel) == "Marie Curie was born in Warsaw."
