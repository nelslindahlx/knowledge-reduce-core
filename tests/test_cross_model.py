"""
Tests for cross-model verification (Session 2).
"""

import pytest

from knowledge_graph_pkg.cross_model import (
    CrossModelVerifier,
    cluster_facts,
    jaccard,
    reliability_for_agreement,
)


class ScriptedProbe:
    """A fake ModelProbe that returns a fixed fact list for every prompt."""

    def __init__(self, model, facts):
        self.model = model
        self._facts = facts

    def probe_domain(self, domain, entities=None, n_prompts=10, seed=42, **kw):
        return [
            {
                "model": self.model,
                "backend": "fake",
                "domain": domain,
                "prompt": "p",
                "prompt_type": "entity",
                "structured_response": {"facts": self._facts},
            }
        ]


def _spo(subject, predicate, obj, confidence=0.9):
    return {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": confidence,
    }


def test_reliability_for_agreement_tiers():
    assert reliability_for_agreement(1) == "POSSIBLY_TRUE"
    assert reliability_for_agreement(2) == "LIKELY_TRUE"
    assert reliability_for_agreement(3) == "VERIFIED"
    assert reliability_for_agreement(5) == "VERIFIED"
    assert reliability_for_agreement(0) == "UNVERIFIED"


def test_weighted_reliability_boundary_and_colon_normalization():
    assert reliability_for_agreement(1, ["gpt-4o"]) == "VERIFIED"
    assert reliability_for_agreement(1, ["qwen2.5:14b"]) == "LIKELY_TRUE"
    assert reliability_for_agreement(1, ["qwen2.5:0.5b"]) == "POSSIBLY_TRUE"
    # Bare 'pro' must not substring-match 'gemini-1.5-pro'
    assert reliability_for_agreement(1, ["pro"]) == "POSSIBLY_TRUE"


def test_jaccard_basic():
    assert jaccard("a b c", "a b c") == 1.0
    assert jaccard("a b c", "x y z") == 0.0
    assert 0 < jaccard("water boils at 100", "water boils at 90") < 1


def test_cluster_facts_does_not_merge_empty_spo_or_crash_empty_embed():
    facts = [
        {"fact_statement": "alpha beta gamma", "subject": "", "predicate": "", "object": ""},
        {"fact_statement": "delta epsilon zeta", "subject": "", "predicate": "", "object": ""},
    ]
    clusters = cluster_facts(facts, similarity_threshold=0.8)
    assert len(clusters) == 2

    class EmptyEmbedder:
        def embed_one(self, text):
            return []

    clusters2 = cluster_facts(
        [
            {"fact_statement": "Mitochondria produce ATP", "subject": "Mitochondria", "predicate": "produce", "object": "ATP"},
            {"fact_statement": "Mitochondria make ATP", "subject": "A", "predicate": "B", "object": "C"},
        ],
        similarity_threshold=0.5,
        embedder=EmptyEmbedder(),
    )
    assert len(clusters2) == 1


def test_empty_probes_rejected():
    with pytest.raises(ValueError):
        CrossModelVerifier([])


def test_two_models_agree_promotes_to_likely_true():
    shared = _spo("Mitochondria", "produce", "ATP")
    v = CrossModelVerifier(
        [
            ScriptedProbe("model-a", [shared]),
            ScriptedProbe("model-b", [shared]),
        ]
    )
    report = v.probe_domain("biochemistry", n_prompts=1)
    assert report["likely_true"] == 1
    assert report["verified"] == 0
    cl = report["clusters"][0]
    assert cl["n_models"] == 2
    assert cl["reliability"] == "LIKELY_TRUE"
    assert set(cl["models"]) == {"model-a", "model-b"}


def test_three_models_agree_promotes_to_verified():
    shared = _spo("Water", "boils at", "100C")
    v = CrossModelVerifier(
        [
            ScriptedProbe("a", [shared]),
            ScriptedProbe("b", [shared]),
            ScriptedProbe("c", [shared]),
        ]
    )
    report = v.probe_domain("physics", n_prompts=1)
    assert report["verified"] == 1
    assert report["clusters"][0]["cross_model_agreement"] == 3


def test_lone_claim_stays_possibly_true():
    v = CrossModelVerifier(
        [
            ScriptedProbe("a", [_spo("Alpha", "is", "unique")]),
            ScriptedProbe("b", [_spo("Beta", "is", "different")]),
        ]
    )
    report = v.probe_domain("misc", n_prompts=1)
    assert report["possibly_true"] == 2
    assert report["likely_true"] == 0
    assert report["unverified"] == 0
    assert all(c["n_models"] == 1 for c in report["clusters"])


def test_same_model_repeating_does_not_inflate_agreement():
    fact = _spo("X", "is", "Y")
    v = CrossModelVerifier([ScriptedProbe("solo", [fact, fact])])
    report = v.probe_domain("d", n_prompts=1)
    assert report["likely_true"] == 0
    assert report["possibly_true"] == 1
    assert report["clusters"][0]["n_models"] == 1


def test_spo_match_clusters_across_phrasing():
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
    v = CrossModelVerifier(
        [
            ScriptedProbe("a", [shared, lone]),
            ScriptedProbe("b", [shared]),
        ]
    )
    report = v.probe_domain("biochemistry", n_prompts=1)
    verified = v.verified_facts(report, min_models=2)
    assert len(verified) == 1
    f = verified[0]
    assert f["reliability_rating"] == "LIKELY_TRUE"
    assert f["cross_model_agreement"] == 2
    assert set(f["source_models"]) == {"a", "b"}
    assert f["question"] and f["answer"]


def test_verify_report_counts_models_list():
    v = CrossModelVerifier(
        [
            ScriptedProbe("a", [_spo("X", "is", "Y")]),
            ScriptedProbe("b", [_spo("X", "is", "Y")]),
        ]
    )
    report = v.probe_domain("d", n_prompts=1)
    assert report["models"] == ["a", "b"]
    assert report["total_facts"] == 2


def test_verify_handles_none_quality_and_unverified_bucket():
    v = CrossModelVerifier([ScriptedProbe("a", [])])
    res = v.verify(
        {
            "m1": [
                {
                    "model": "m1",
                    "structured_response": {
                        "facts": [{"subject": "A", "predicate": "is", "object": "B"}]
                    },
                }
            ]
        }
    )
    assert res["n_clusters"] == 1
    assert "unverified" in res


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


@pytest.mark.skipif(
    not (_ollama_importable() and _ollama_up()),
    reason="Ollama package/server not available",
)
def test_live_cross_model_smoke():
    v = CrossModelVerifier.from_ollama(["qwen2.5:7b"])
    report = v.probe_domain(
        "biochemistry", entities=["Mitochondria"], n_prompts=2, seed=42
    )
    assert "clusters" in report and report["total_facts"] >= 0
