"""
Tests for fact quality filtering.

Heuristic extraction casts a wide net and produces many low-quality facts:
sentence-initial words ("The", "This", "Being", "However") mistaken for
subjects, and run-on objects that are whole clauses rather than crisp
answers. FactQualityFilter scores and rejects these so the distilled
output is clean enough for training data.
"""
import pytest

from knowledge_graph_pkg.quality import FactQualityFilter


@pytest.fixture
def qf():
    return FactQualityFilter(max_object_len=80)


def _fact(subject, predicate, object_, subject_type="ENTITY", object_type="CONCEPT"):
    return {
        "subject": subject, "subject_type": subject_type,
        "predicate": predicate, "object": object_, "object_type": object_type,
    }


# ---------- accept good facts ----------

def test_accepts_clean_entity_fact(qf):
    assert qf.is_acceptable(_fact("Robert Putnam", "wrote", "Bowling Alone"))


def test_accepts_role_fact(qf):
    assert qf.is_acceptable(
        _fact("Charles Carlsen", "president_of", "Johnson County Community College")
    )


# ---------- reject stopword / non-entity subjects ----------

@pytest.mark.parametrize("subj", ["The", "This", "There", "However", "Being",
                                   "Having", "If", "When", "Without", "That"])
def test_rejects_stopword_subject(qf, subj):
    assert not qf.is_acceptable(_fact(subj, "is", "something useful"))


def test_rejects_gerund_subject(qf):
    # "Developing strategies ..." -> subject "Developing" is not an entity
    assert not qf.is_acceptable(_fact("Developing", "is", "a first step"))


def test_rejects_empty_subject_or_object(qf):
    assert not qf.is_acceptable(_fact("", "wrote", "a book"))
    assert not qf.is_acceptable(_fact("Nels Lindahl", "wrote", ""))


# ---------- reject run-on objects ----------

def test_rejects_overlong_object(qf):
    long_obj = "a very long clause " * 10  # >80 chars
    assert not qf.is_acceptable(_fact("Nels Lindahl", "wrote", long_obj))


def test_object_length_threshold_is_configurable():
    strict = FactQualityFilter(max_object_len=10)
    assert not strict.is_acceptable(_fact("A", "is", "this is more than ten chars"))
    loose = FactQualityFilter(max_object_len=200)
    assert loose.is_acceptable(_fact("Nels Lindahl", "wrote", "this is more than ten chars"))


# ---------- bulk filter + ratio ----------

def test_filter_returns_only_acceptable(qf):
    facts = [
        _fact("Robert Putnam", "wrote", "Bowling Alone"),
        _fact("The", "is", "a determiner"),
        _fact("Marie Curie", "discovered", "radium"),
    ]
    kept = qf.filter(facts)
    subjects = [f["subject"] for f in kept]
    assert subjects == ["Robert Putnam", "Marie Curie"]


def test_score_is_monotonic(qf):
    good = qf.score(_fact("Robert Putnam", "wrote", "Bowling Alone"))
    bad = qf.score(_fact("The", "is", "a determiner that runs on and on and on and on"))
    assert good > bad


# ---------- strict mode: entity-subject + short objects ----------

def test_strict_requires_entity_subject():
    strict = FactQualityFilter(require_entity_subject=True, max_object_len=60)
    # CONCEPT subject rejected even though capitalized and not a stopword
    assert not strict.is_acceptable(
        _fact("Change", "is", "inevitable", subject_type="CONCEPT")
    )
    # ENTITY subject accepted
    assert strict.is_acceptable(
        _fact("Robert Putnam", "wrote", "Bowling Alone", subject_type="ENTITY")
    )


def test_strict_rejects_single_word_concept_even_without_type():
    # When require_entity_subject is on and no type given, a lone lowercase-ish
    # abstract noun like "Civil" (from "Civil history") should be rejected
    # because it is not a multi-word proper name nor typed ENTITY.
    strict = FactQualityFilter(require_entity_subject=True)
    assert not strict.is_acceptable(_fact("Civil", "is", "history", subject_type="CONCEPT"))


def test_non_strict_still_accepts_concept_subject():
    loose = FactQualityFilter(require_entity_subject=False)
    assert loose.is_acceptable(_fact("Change", "is", "inevitable", subject_type="CONCEPT"))


def test_strict_predicate_blocklist_rejects_bare_copula_concept():
    # "X is Y" with a CONCEPT subject is usually a fragment; strict drops it.
    strict = FactQualityFilter(require_entity_subject=True)
    assert not strict.is_acceptable(_fact("Organizations", "is", "social", subject_type="CONCEPT"))
