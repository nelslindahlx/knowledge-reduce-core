"""
Tests for the extraction evaluation harness.

evaluation.evaluate() runs an extractor over a labeled gold set and
returns precision / recall / F1 by comparing predicted triples to
expected ones with lenient (case-insensitive substring) matching.
"""
import pytest

from knowledge_graph_pkg.evaluation import (
    triple_matches, evaluate, load_gold_set, GoldItem,
)
from knowledge_graph_pkg.extraction import SVOExtractor


# ---------- matching ----------

def test_triple_matches_exact():
    pred = {"subject": "Robert Putnam", "predicate": "wrote", "object": "Bowling Alone"}
    gold = {"subject": "Robert Putnam", "predicate": "wrote", "object": "Bowling Alone"}
    assert triple_matches(pred, gold)


def test_triple_matches_lenient_substring_and_case():
    pred = {"subject": "robert putnam", "predicate": "wrote", "object": "Bowling Alone in 2000"}
    gold = {"subject": "Robert Putnam", "predicate": "wrote", "object": "Bowling Alone"}
    assert triple_matches(pred, gold)


def test_triple_does_not_match_wrong_object():
    pred = {"subject": "Robert Putnam", "predicate": "wrote", "object": "something else"}
    gold = {"subject": "Robert Putnam", "predicate": "wrote", "object": "Bowling Alone"}
    assert not triple_matches(pred, gold)


# ---------- gold set loading ----------

def test_load_gold_set():
    items = load_gold_set("data/gold_set.json")
    assert len(items) >= 10
    assert all(isinstance(it, GoldItem) for it in items)
    # at least one negative example (header/fragment -> no facts)
    assert any(len(it.facts) == 0 for it in items)


# ---------- evaluation ----------

def test_evaluate_returns_metrics():
    items = load_gold_set("data/gold_set.json")
    report = evaluate(SVOExtractor(), items)
    for key in ("precision", "recall", "f1", "true_positives",
                "false_positives", "false_negatives"):
        assert key in report
    assert 0.0 <= report["precision"] <= 1.0
    assert 0.0 <= report["recall"] <= 1.0
    assert 0.0 <= report["f1"] <= 1.0


def test_evaluate_recall_is_reasonable():
    # The gold set is built from sentences the SVO extractor should handle;
    # recall should clear a sane floor (guards against regressions).
    items = load_gold_set("data/gold_set.json")
    report = evaluate(SVOExtractor(), items)
    assert report["recall"] >= 0.5, f"recall too low: {report}"


def test_evaluate_perfect_on_trivial_gold():
    # A gold set the extractor handles exactly should score F1 == 1.0.
    items = [GoldItem(text="Robert Putnam wrote Bowling Alone.",
                      facts=[{"subject": "Robert Putnam", "predicate": "wrote",
                              "object": "Bowling Alone"}])]
    report = evaluate(SVOExtractor(), items)
    assert report["f1"] == 1.0
