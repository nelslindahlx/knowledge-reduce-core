"""
Tests for the model shard evaluator (ModelReduce Session 5).

ModelShardEvaluator scores distilled shard facts against a domain gold set:
per-reliability-tier precision/recall/F1, hallucination rate (facts that
match a gold *negative*), coverage (% of gold facts recovered), and
agreement calibration (does N-model agreement predict precision?). Quality
gates turn the metrics into a pass/fail for CI.

Tests use a fake embedder (deterministic) so they run without Ollama, and a
lenient string fallback path is exercised with embedder=None.
"""
import json
import pytest

from knowledge_graph_pkg.model_eval import (
    ModelShardEvaluator, load_gold_facts, check_gates, DEFAULT_GATES,
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


def _fact(subject, predicate, obj, reliability="LIKELY_TRUE", agreement=2,
          models=None):
    return {
        "subject": subject, "predicate": predicate, "object": obj,
        "fact_statement": f"{subject} {predicate} {obj}.",
        "reliability_rating": reliability,
        "cross_model_agreement": agreement,
        "source_models": models or ["m1", "m2"],
    }


def test_load_gold_facts(tmp_path):
    pos, neg = load_gold_facts(_gold_file(tmp_path))
    assert len(pos) == 3 and len(neg) == 1


def test_perfect_shard_scores_high(tmp_path):
    ev = ModelShardEvaluator(embedder=None)  # lenient string matching
    shard = [
        _fact("Mitochondria", "produce", "ATP"),
        _fact("Ribosomes", "synthesize", "proteins"),
        _fact("Glucose", "is", "a monosaccharide"),
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["overall"]["precision"] == 1.0
    assert rep["coverage"] == 1.0
    assert rep["hallucination_rate"] == 0.0


def test_hallucination_detected(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP"),
        _fact("Mitochondria", "perform", "photosynthesis"),  # matches gold negative
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["hallucination_rate"] > 0


def test_false_positive_lowers_precision(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [
        _fact("Mitochondria", "produce", "ATP"),
        _fact("Unicorns", "grant", "wishes"),  # not in gold at all -> FP
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["overall"]["precision"] < 1.0


def test_partial_coverage(tmp_path):
    ev = ModelShardEvaluator(embedder=None)
    shard = [_fact("Mitochondria", "produce", "ATP")]  # 1 of 3 gold
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
        _fact("Unicorns", "grant", "wishes", agreement=2),  # FP at agreement 2
    ]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    cal = rep["agreement_calibration"]
    # agreement 3 should be perfectly precise; agreement 2 has a FP
    assert cal["3"]["precision"] == 1.0
    assert cal["2"]["precision"] < 1.0


# ---------- quality gates ----------

def test_gates_pass_on_good_report():
    report = {
        "overall": {"precision": 0.97, "recall": 0.7, "f1": 0.81},
        "by_tier": {"VERIFIED": {"precision": 0.98}},
        "hallucination_rate": 0.02,
        "coverage": 0.65,
    }
    ok, failures = check_gates(report, DEFAULT_GATES)
    assert ok is True and failures == []


def test_gates_fail_on_high_hallucination():
    report = {
        "overall": {"precision": 0.97, "recall": 0.7, "f1": 0.81},
        "by_tier": {"VERIFIED": {"precision": 0.98}},
        "hallucination_rate": 0.20,  # too high
        "coverage": 0.65,
    }
    ok, failures = check_gates(report, DEFAULT_GATES)
    assert ok is False
    assert any("hallucination" in f.lower() for f in failures)


# ---------- live embedder calibration (skipped without embed model) ----------

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
    # paraphrased fact should still match the gold via embeddings
    shard = [_fact("Mitochondria", "generate", "adenosine triphosphate")]
    rep = ev.evaluate(shard, _gold_file(tmp_path))
    assert rep["overall"]["true_positives"] >= 1
