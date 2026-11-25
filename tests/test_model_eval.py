"""
Tests for the model shard evaluator (ModelReduce Session 5).
"""

import json

import pytest

from knowledge_graph_pkg.model_eval import (
    DEFAULT_GATES,
    ModelShardEvaluator,
    _lenient_match,
    check_gates,
    format_report,
    load_gold_facts,
)

GOLD = {
    "domain": "biochemistry",
    "facts": [
        {"subject": "Mitochondria", "predicate": "produce", "object": "ATP", "verified": True},
        {"subject": "Ribosomes", "predicate": "synthesize", "object": "proteins", "verified": True},
        {"subject": "Glucose", "predicate": "is", "object": "a monosaccharide", "verified": True},
    ],
    "negative": [
        {"subject": "Mitochondria", "predicate": "perform", "object": "photosynthesis", "verified": False},
    ],
}


def _gold_file(tmp_path):
    p = tmp_path / "gold.json"
    p.write_text(json.dumps(GOLD))
    return str(p)


def _fact(subject, predicate, obj, reliability="LIKELY_TRUE", agreement=2, models=None):
    rec = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "fact_statement": f"{subject} {predicate} {obj}.",
        "cross_model_agreement": agreement,
        "source_models": models or ["m1", "m2"],
    }
    if reliability is not None:
        rec["reliability_rating"] = reliability
    return rec


def test_load_gold_facts(tmp_path):
    pos, neg = load_gold_facts(_gold_file(tmp_path))
    assert len(pos) == 3 and len(neg) == 1


def test_perfect_shard_scores_high(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP"),
        _fact("Ribosomes", "synthesize", "proteins"),
        _fact("Glucose", "is", "a monosaccharide"),
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["overall"]["precision"] == 1.0
    assert rep["overall"]["recall"] == 1.0
    assert rep["coverage"] == 1.0
    assert rep["hallucination_rate"] == 0.0


def test_recall_with_duplicate_shard_facts_is_exact(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP"),
        _fact("Mitochondria", "generate", "ATP"),
        _fact("Mitochondria", "make", "ATP"),
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    # Only 1 of 3 gold facts matched -> recall must be 1/3, not 3/(3+2)!
    assert rep["overall"]["precision"] == 1.0
    assert rep["overall"]["recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert rep["coverage"] == pytest.approx(1 / 3, abs=1e-4)


def test_empty_shard_precision_and_recall_are_zero(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    rep = ev.evaluate([], _gold_file(tmp_path))
    assert rep["overall"]["precision"] == 0.0
    assert rep["overall"]["recall"] == 0.0
    assert rep["overall"]["f1"] == 0.0


def test_lenient_match_negation_and_short_token_boundary():
    pos = {"subject": "Mitochondria", "predicate": "produce", "object": "ATP"}
    neg = {"subject": "Mitochondria", "predicate": "do not produce", "object": "ATP"}
    assert _lenient_match(pos, neg) is False
    assert _lenient_match(
        {"subject": "ATP", "predicate": "is", "object": "energy"},
        {"subject": "catapult", "predicate": "uses", "object": "energy"},
    ) is False


def test_missing_reliability_grouped_into_unverified_tier(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [_fact("Mitochondria", "produce", "ATP", reliability=None)]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["by_tier"]["UNVERIFIED"]["count"] == 1


def test_hallucination_detected(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP"),
        _fact("Mitochondria", "perform", "photosynthesis"),
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["hallucination_rate"] > 0


def test_false_positive_lowers_precision(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP"),
        _fact("Unicorns", "grant", "wishes"),
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["overall"]["precision"] < 1.0


def test_partial_coverage(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [_fact("Mitochondria", "produce", "ATP")]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert 0 < rep["coverage"] < 1.0


def test_per_tier_metrics_present(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP", reliability="VERIFIED", agreement=3),
        _fact("Ribosomes", "synthesize", "proteins", reliability="LIKELY_TRUE", agreement=2),
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert "VERIFIED" in rep["by_tier"]
    assert "LIKELY_TRUE" in rep["by_tier"]
    for tier in ("VERIFIED", "LIKELY_TRUE"):
        assert "precision" in rep["by_tier"][tier]


def test_agreement_calibration(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP", agreement=3),
        _fact("Ribosomes", "synthesize", "proteins", agreement=2),
        _fact("Unicorns", "grant", "wishes", agreement=2),
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    cal = rep["agreement_calibration"]
    assert cal["3"]["precision"] == 1.0
    assert cal["2"]["precision"] < 1.0


def test_gates_pass_and_partial_override():
    report = {
        "overall": {"precision": 0.97, "recall": 0.7, "f1": 0.81},
        "by_tier": {"VERIFIED": {"precision": 0.98, "recall": 0.7, "count": 1}},
        "agreement_calibration": {
            "10": {"precision": 1.0, "n": 1},
            "2": {"precision": 0.8, "n": 2},
        },
        "hallucination_rate": 0.02,
        "coverage": 0.65,
        "n_shard_facts": 3,
        "n_gold_positive": 3,
        "n_gold_negative": 1,
    }
    ok, failures = check_gates(report, DEFAULT_GATES)
    assert ok is True and failures == []

    # Partial gates override must not KeyError on missing keys
    ok2, failures2 = check_gates(report, {"min_coverage": 0.9})
    assert ok2 is False
    assert any("coverage" in f for f in failures2)

    formatted = format_report(report)
    pos_2 = formatted.index("2-model")
    pos_10 = formatted.index("10-model")
    assert pos_2 < pos_10


def test_gates_fail_on_high_hallucination():
    report = {
        "overall": {"precision": 0.97, "recall": 0.7, "f1": 0.81},
        "by_tier": {"VERIFIED": {"precision": 0.98}},
        "hallucination_rate": 0.20,
        "coverage": 0.65,
    }
    ok, failures = check_gates(report, DEFAULT_GATES)
    assert ok is False
    assert any("hallucination" in f.lower() for f in failures)


def _ollama_importable():
    try:
        import ollama  # noqa: F401

        return True
    except Exception:
        return False


def _ollama_up():
    try:
        import urllib.request

        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _embed_available():
    return _ollama_importable() and _ollama_up()


@pytest.mark.skipif(not _embed_available(), reason="No Ollama embedding model")
def test_live_embedder_matches_paraphrase(tmp_path):
    from knowledge_graph_pkg.embeddings import LocalEmbedder

    ev = ModelShardEvaluator(embedder=LocalEmbedder())
    shard = [_fact("Mitochondria", "generate", "adenosine triphosphate")]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["overall"]["true_positives"] >= 1
