"""
Tests for the stronger general-purpose relation extractor (SVOExtractor).

The original pattern extractor only fired on ~15 hardcoded verbs with a
named entity on both sides, so dense real-world prose produced almost
nothing. SVOExtractor adds general subject-verb-object parsing, copula/role
handling, passive voice, and multi-subject expansion -- all pure Python.
"""
import pytest

from knowledge_graph_pkg.extraction import SVOExtractor


@pytest.fixture
def ex():
    return SVOExtractor()


def _triples(rels):
    return [(r["subject"], r["predicate"], r["object"]) for r in rels]


# ---------- active voice SVO ----------

def test_active_svo_basic(ex):
    rels = ex.extract("Robert Putnam wrote Bowling Alone.")
    assert ("Robert Putnam", "wrote", "Bowling Alone") in _triples(rels)


def test_active_svo_keeps_full_object(ex):
    rels = ex.extract("Johnson County Community College developed a pilot program.")
    t = _triples(rels)
    assert any(s == "Johnson County Community College" and p == "developed"
               and "pilot program" in o for s, p, o in t)


# ---------- copula: role "X was President of Y" ----------

def test_role_of_relation(ex):
    rels = ex.extract("Charles Carlsen was President of Johnson County Community College.")
    t = _triples(rels)
    assert ("Charles Carlsen", "president_of",
            "Johnson County Community College") in t


# ---------- copula: passive "X was published in Y" ----------

def test_passive_with_prepositional_object(ex):
    rels = ex.extract("Graduation With Civic Honors was published in 2006.")
    t = _triples(rels)
    assert any(p == "published" and o == "2006" for s, p, o in t)


# ---------- copula: "X is a Y" ----------

def test_is_a_relation(ex):
    rels = ex.extract("Civic engagement is a value choice.")
    t = _triples(rels)
    assert any(p == "is_a" and "value choice" in o for s, p, o in t)


# ---------- multi-subject expansion ----------

def test_multi_subject_splits_into_multiple_facts(ex):
    rels = ex.extract("Deborah DeGrate and Chris Engle graduated with civic honors.")
    subjects = {r["subject"] for r in rels}
    assert "Deborah DeGrate" in subjects
    assert "Chris Engle" in subjects


# ---------- robustness ----------

def test_no_verb_yields_nothing(ex):
    assert ex.extract("Key Components:") == []


def test_predicate_never_contains_spaces(ex):
    # predicates must be single tokens (underscored) for clean tags/Q&A
    for rel in ex.extract("Charles Carlsen was President of Johnson County Community College."):
        assert " " not in rel["predicate"]


def test_title_abbreviation_does_not_split_sentence(ex):
    # "Dr." must not end a sentence; the fact should stay intact.
    rels = ex.extract("Dr. Raymond Davis provided advice and guidance.")
    t = _triples(rels)
    assert any(s == "Dr. Raymond Davis" or s == "Raymond Davis" for s, p, o in t)
    # the object must not be just "Dr" from a bad split
    assert all(o != "Dr" for s, p, o in t)
