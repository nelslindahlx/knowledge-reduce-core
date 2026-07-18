"""
Tests for cross-model verification (Session 2).

CrossModelVerifier probes N models with identical prompts, clusters
semantically-equivalent facts, and promotes reliability by distinct-model
agreement (1->POSSIBLY_TRUE, 2->LIKELY_TRUE, >=3->VERIFIED). Tests inject
scripted fake probes -- no Ollama required. A live test is skip-guarded.
"""
import pytest

from knowledge_graph_pkg.cross_model import (
    CrossModelVerifier, reliability_for_agreement, jaccard,
)


class ScriptedProbe:
    """A fake ModelProbe that returns a fixed fact list for every prompt."""
    def __init__(self, model, facts):
        self.model = model
        self._facts = facts

    def probe_domain(self, domain, entities=None, n_prompts=10, seed=42, **kw):
        # One probe output carrying this model's scripted facts.
        return [{
            "model": self.model, "backend": "fake", "domain": domain,
            "prompt": "p", "prompt_type": "entity",
            "structured_response": {"facts": self._facts},
        }]


def _spo(subject, predicate, obj, confidence=0.9):
    return {"subject": subject, "predicate": predicate, "object": obj,
            "confidence": confidence}


# ---------- agreement -> reliability mapping ----------

def test_reliability_for_agreement_tiers():
    assert reliability_for_agreement(1) == "POSSIBLY_TRUE"
    assert reliability_for_agreement(2) == "LIKELY_TRUE"
    assert reliability_for_agreement(3) == "VERIFIED"
    assert reliability_for_agreement(5) == "VERIFIED"
    assert reliability_for_agreement(0) == "UNVERIFIED"


def test_jaccard_basic():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b c", "x y z") == 0.0
    assert 0 < jaccard("water boils at 100", "water boils at 90") < 1


# ---------- construction guard ----------

def test_empty_probes_rejected():
    with pytest.raises(ValueError):
        CrossModelVerifier([])


# ---------- cross-model agreement ----------

def test_two_models_agree_promotes_to_likely_true():
    shared = _spo("Mitochondria", "produce", "ATP")
    v = CrossModelVerifier([
        ScriptedProbe("model-a", [shared]),
        ScriptedProbe("model-b", [shared]),
    ])
    report = v.probe_domain("biochemistry", n_prompts=1)
    assert report["likely_true"] == 1
    assert report["verified"] == 0
    cl = report["clusters"][0]
    assert cl["n_models"] == 2
    assert cl["reliability"] == "LIKELY_TRUE"
    assert set(cl["models"]) == {"model-a", "model-b"}


def test_three_models_agree_promotes_to_verified():
    shared = _spo("Water", "boils at", "100C")
    v = CrossModelVerifier([
        ScriptedProbe("a", [shared]),
        ScriptedProbe("b", [shared]),
        ScriptedProbe("c", [shared]),
    ])
    report = v.probe_domain("physics", n_prompts=1)
    assert report["verified"] == 1
    assert report["clusters"][0]["cross_model_agreement"] == 3


def test_lone_claim_stays_possibly_true():
    v = CrossModelVerifier([
        ScriptedProbe("a", [_spo("Alpha", "is", "unique")]),
        ScriptedProbe("b", [_spo("Beta", "is", "different")]),
    ])
    report = v.probe_domain("misc", n_prompts=1)
    # two distinct facts, each backed by one model
    assert report["possibly_true"] == 2
    assert report["likely_true"] == 0
    assert all(c["n_models"] == 1 for c in report["clusters"])


def test_same_model_repeating_does_not_inflate_agreement():
    # One model emitting the same fact twice must NOT count as 2-model agreement.
    fact = _spo("X", "is", "Y")
    v = CrossModelVerifier([ScriptedProbe("solo", [fact, fact])])
    report = v.probe_domain("d", n_prompts=1)
    assert report["likely_true"] == 0
    assert report["possibly_true"] == 1
    assert report["clusters"][0]["n_models"] == 1


def test_spo_match_clusters_across_phrasing():
    # Same triple, different qualifier wording -> still one cluster.
    a = _spo("Earth", "orbits", "Sun")
    b = _spo("Earth", "orbits", "Sun")
    b["context_or_qualifier"] = "in an elliptical path"
    v = CrossModelVerifier([ScriptedProbe("a", [a]), ScriptedProbe("b", [b])])
    report = v.probe_domain("astronomy", n_prompts=1)
    assert report["n_clusters"] == 1
    assert report["clusters"][0]["n_models"] == 2


def test_verified_facts_extraction_filters_by_min_models():
    shared = _spo("Mitochondria", "produce", "ATP")
    lone = _spo("Ribosome", "makes", "protein")
    v = CrossModelVerifier([
        ScriptedProbe("a", [shared, lone]),
        ScriptedProbe("b", [shared]),
    ])
    report = v.probe_domain("biochemistry", n_prompts=1)
    verified = v.verified_facts(report, min_models=2)
    # only the shared fact clears the 2-model bar
    assert len(verified) == 1
    f = verified[0]
    assert f["reliability_rating"] == "LIKELY_TRUE"
    assert f["cross_model_agreement"] == 2
    assert set(f["source_models"]) == {"a", "b"}
    assert f["question"] and f["answer"]


def test_verify_report_counts_models_list():
    v = CrossModelVerifier([
        ScriptedProbe("a", [_spo("X", "is", "Y")]),
        ScriptedProbe("b", [_spo("X", "is", "Y")]),
    ])
    report = v.probe_domain("d", n_prompts=1)
    assert report["models"] == ["a", "b"]
    assert report["total_facts"] == 2


# ---------- live test (skipped unless Ollama is up) ----------

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


@pytest.mark.skipif(not (_ollama_importable() and _ollama_up()), reason="Ollama package/server not available")
def test_live_cross_model_smoke():
    v = CrossModelVerifier.from_ollama(["qwen2.5:7b"])
    report = v.probe_domain("biochemistry", entities=["Mitochondria"],
                            n_prompts=2, seed=42)
    assert "clusters" in report and report["total_facts"] >= 0
