"""
Tests for ModelProbe (Session 1).

ModelProbe drives a backend (Ollama in v1) to generate structured facts for
a domain. Tests use a FakeBackend so they run without Ollama; a separate
live test is skip-guarded behind Ollama availability.
"""
import os
import pytest

pytest.importorskip("pydantic")

from knowledge_graph_pkg.model_probe import ModelProbe
from knowledge_graph_pkg.schemas import PROBE_OUTPUT_SCHEMA


class FakeBackend:
    """Deterministic stand-in for OllamaBackend — no network/model needed."""
    def __init__(self, model="fake-model"):
        self.model = model
        self.calls = []

    def generate_structured(self, prompt, schema, **kw):
        self.calls.append(prompt)
        return {"facts": [
            {"subject": "Mitochondria", "predicate": "produce", "object": "ATP",
             "context_or_qualifier": "oxidative phosphorylation", "confidence": 0.95},
        ]}


def test_probe_domain_returns_structured_outputs():
    probe = ModelProbe(backend=FakeBackend(), model="fake-model")
    outs = probe.probe_domain("biochemistry", entities=["Mitochondria", "ATP"],
                              n_prompts=5, schema=PROBE_OUTPUT_SCHEMA, seed=1)
    assert len(outs) == 5
    for o in outs:
        assert o["model"] == "fake-model"
        assert o["backend"] == "fake"
        assert o["domain"] == "biochemistry"
        assert o["prompt_type"] in ("entity", "relation", "concept", "list", "negative")
        assert "structured_response" in o
        assert "facts" in o["structured_response"]
        assert "gen_config" in o and "timestamp" in o


def test_probe_records_prompt_and_facts():
    fb = FakeBackend()
    probe = ModelProbe(backend=fb, model="fake-model")
    outs = probe.probe_domain("physics", entities=["mass", "energy"], n_prompts=3, seed=2)
    assert len(fb.calls) == 3
    assert all(o["structured_response"]["facts"] for o in outs)


def test_probe_backend_name_inferred():
    probe = ModelProbe(backend=FakeBackend(), model="m")
    outs = probe.probe_domain("law", entities=[], n_prompts=2, seed=3)
    assert outs[0]["backend"] == "fake"


# ---------- live test (skipped unless Ollama is up) ----------

def _ollama_up():
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama server not available")
def test_live_ollama_probe_smoke():
    import pytest
    pytest.importorskip("ollama")
    from knowledge_graph_pkg.model_probe import OllamaBackend
    backend = OllamaBackend(model="qwen2.5:7b")
    probe = ModelProbe(backend=backend, model="qwen2.5:7b")
    outs = probe.probe_domain("biochemistry", entities=["Mitochondria"],
                              n_prompts=2, schema=PROBE_OUTPUT_SCHEMA, seed=42)
    assert len(outs) == 2
    assert all("structured_response" in o for o in outs)


def test_gemini_backend_interface():
    import sys
    import pytest
    pytest.importorskip("google.generativeai")
    from unittest.mock import MagicMock
    
    mock_genai = MagicMock()
    mock_model_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"facts": [{"subject": "A", "predicate": "B", "object": "C"}]}'
    mock_model_inst.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model_inst
    
    sys.modules["google.generativeai"] = mock_genai
    
    from knowledge_graph_pkg.model_probe import GeminiBackend
    try:
        backend = GeminiBackend(model="gemini-1.5-flash", api_key="dummy-key")
        assert backend.model == "gemini-1.5-flash"
        
        # 1. Test generate_structured
        res = backend.generate_structured("Explain X", schema={})
        assert "facts" in res
        assert res["facts"][0]["subject"] == "A"
        
        # 2. Test generate_text_with_logprobs
        mock_response.text = "This is text"
        text, logprob = backend.generate_text_with_logprobs("Explain Y")
        assert text == "This is text"
        assert logprob == 0.0
    finally:
        if "google.generativeai" in sys.modules:
            del sys.modules["google.generativeai"]

