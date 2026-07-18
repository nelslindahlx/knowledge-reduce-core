"""
Tests for lightweight pronoun/coreference resolution.

A common failure of sentence-by-sentence extraction is that a pronoun
subject ("She discovered radium.") loses its antecedent. This resolver
replaces leading/subject pronouns with the most recent compatible named
entity seen in prior sentences, so extracted facts read correctly and the
generated Q&A pairs become usable training data.
"""
import pytest

from knowledge_graph_pkg import KnowledgeGraph, SemanticKnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.coref import resolve_coreferences


# ---------- unit: resolve_coreferences ----------

def test_she_resolved_to_preceding_person():
    text = "Marie Curie was born in Warsaw. She discovered radium."
    out = resolve_coreferences(text)
    assert "She discovered" not in out
    assert "Marie Curie discovered radium." in out


def test_he_resolved_to_preceding_person():
    text = "Albert Einstein moved to Princeton. He developed relativity."
    out = resolve_coreferences(text)
    assert "Albert Einstein developed relativity." in out


def test_first_sentence_pronoun_left_untouched():
    # No antecedent available -> leave it alone rather than guess.
    text = "She discovered radium."
    out = resolve_coreferences(text)
    assert out == "She discovered radium."


def test_non_pronoun_text_unchanged():
    text = "Marie Curie discovered radium."
    assert resolve_coreferences(text) == text


def test_they_resolved_to_recent_entity():
    text = "The Beatles formed in Liverpool. They recorded many albums."
    out = resolve_coreferences(text)
    assert "They recorded" not in out


# ---------- integration: extraction with coref enabled ----------

def test_create_facts_resolves_pronoun_subject():
    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    ids = skg.create_facts_from_text(
        "Marie Curie was born in Warsaw. She discovered radium.",
        source_id="demo",
        reliability=ReliabilityRating.LIKELY_TRUE,
        resolve_coref=True,
    )
    statements = [kg.get_fact(i)["fact_statement"] for i in ids]
    questions = [kg.get_fact(i).get("question") for i in ids]
    # the discover fact should now be attributed to Marie Curie, not "She"
    assert any("Marie Curie discovered" in s for s in statements)
    assert all("She" not in (q or "") for q in questions)


def test_coref_is_opt_in_default_off():
    """Default behavior unchanged: without resolve_coref, pronoun stays."""
    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    ids = skg.create_facts_from_text(
        "Marie Curie was born in Warsaw. She discovered radium.",
        source_id="demo",
        reliability=ReliabilityRating.LIKELY_TRUE,
    )
    statements = [kg.get_fact(i)["fact_statement"] for i in ids]
    assert any("She discovered" in s for s in statements)
