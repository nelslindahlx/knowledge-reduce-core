"""
Tests for ModelProbe and probing backends (Session 1).
"""

import importlib.util
import sys
from unittest.mock import MagicMock, patch

import pytest

from knowledge_graph_pkg.model_probe import (
    GeminiBackend,
    LlamaCppBackend,
    ModelProbe,
    OllamaBackend,
    OpenAICompatibleBackend,
    _parse_structured_json,
    _sanitize_schema_for_gemini,
    get_backend,
)

PYDANTIC_AVAILABLE = importlib.util.find_spec("pydantic") is not None


class FakeBackend:
    """Deterministic stand-in for OllamaBackend -- no network/model needed."""

    def __init__(self, model="fake-model"):
        self.model = model
        self.calls = []
        self.seeds = []

    def generate_structured(self, prompt, schema, **kw):
        self.calls.append(prompt)
        self.seeds.append(kw.get("seed"))
        return {
            "facts": [
                {
                    "subject": "Mitochondria",
                    "predicate": "produce",
                    "object": "ATP",
                    "context_or_qualifier": "oxidative phosphorylation",
                    "confidence": 0.95,
                },
            ]
        }


def test_probe_domain_returns_structured_outputs_and_varies_seed():
    pytest.importorskip("pydantic")
    from knowledge_graph_pkg.schemas import PROBE_OUTPUT_SCHEMA

    fb = FakeBackend()
    probe = ModelProbe(backend=fb, model="fake-model")
    outs = probe.probe_domain(
        "biochemistry",
        entities=["Mitochondria", "ATP"],
        n_prompts=5,
        schema=PROBE_OUTPUT_SCHEMA,
        seed=1,
    )
    assert len(outs) == 5
    assert fb.seeds == [1, 2, 3, 4, 5]
    for o in outs:
        assert o["model"] == "fake-model"
        assert o["backend"] == "fake"
        assert o["domain"] == "biochemistry"
        assert o["prompt_type"] in ("entity", "relation", "concept", "list", "negative")
        assert "structured_response" in o
        assert "facts" in o["structured_response"]
        assert "gen_config" in o and "timestamp" in o


def test_probe_records_prompt_and_facts_with_explicit_schema():
    fb = FakeBackend()
    probe = ModelProbe(backend=fb, model="fake-model")
    outs = probe.probe_domain(
        "physics",
        entities=["mass", "energy"],
        n_prompts=3,
        schema={"type": "object", "properties": {"facts": {"type": "array"}}},
        seed=2,
    )
    assert len(fb.calls) == 3
    assert all(o["structured_response"]["facts"] for o in outs)


def test_probe_backend_name_inferred():
    probe = ModelProbe(
        backend=FakeBackend(),
        model="m",
    )
    outs = probe.probe_domain(
        "law",
        entities=[],
        n_prompts=2,
        schema={"type": "object", "properties": {"facts": {"type": "array"}}},
        seed=3,
    )
    assert outs[0]["backend"] == "fake"


def test_parse_structured_json_normalizes_non_dict():
    assert _parse_structured_json('["not", "a", "dict"]') == {"facts": []}
    assert _parse_structured_json('{"facts": "bad"}')["facts"] == []
    assert _parse_structured_json('```json\n{"facts": [{"subject": "A"}]}\n```') == {
        "facts": [{"subject": "A"}]
    }


def test_ollama_backend_passes_timeout_and_handles_none_text():
    mock_ollama = MagicMock()
    mock_client = MagicMock()
    mock_client.generate.return_value = {"response": None}
    mock_ollama.Client.return_value = mock_client

    with patch.dict(sys.modules, {"ollama": mock_ollama}):
        backend = OllamaBackend(model="qwen2.5:7b", timeout=15.0)
        mock_ollama.Client.assert_called_with(host="http://localhost:11434", timeout=15.0)
        assert backend.generate_structured("hi", {"properties": {"facts": {}}}) == {"facts": []}
        text, lp = backend.generate_text_with_logprobs("hi")
        assert text == "" and lp == 0.0


def test_llama_cpp_backend_sets_logits_all_and_handles_none_logprobs():
    mock_llama_mod = MagicMock()
    mock_llama_inst = MagicMock()
    mock_llama_inst.create_chat_completion.return_value = {
        "choices": [
            {
                "message": {"content": "Hello world"},
                "logprobs": {"token_logprobs": [None, -0.5, -1.5]},
            }
        ]
    }
    mock_llama_mod.Llama.return_value = mock_llama_inst

    with patch.dict(sys.modules, {"llama_cpp": mock_llama_mod}):
        backend = get_backend("llama-cpp", model=None, model_path="/tmp/m.gguf", n_ctx=4096)
        assert isinstance(backend, LlamaCppBackend)
        assert mock_llama_mod.Llama.call_args.kwargs["logits_all"] is True
        assert mock_llama_mod.Llama.call_args.kwargs["n_ctx"] == 4096
        text, avg_lp = backend.generate_text_with_logprobs("hi")
        assert text == "Hello world"
        assert avg_lp == pytest.approx(-1.0)

        with pytest.raises(ValueError):
            get_backend("llama-cpp", model=None)


def test_openai_backend_uses_json_schema_and_timeout():
    mock_openai = MagicMock()
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = '{"facts": [{"subject": "Water", "predicate": "is", "object": "liquid"}]}'
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
    mock_openai.OpenAI.return_value = mock_client

    with patch.dict(sys.modules, {"openai": mock_openai}):
        backend = OpenAICompatibleBackend(model="gpt-4o", api_key="sk-test", timeout=25.0)
        mock_openai.OpenAI.assert_called_with(api_key="sk-test", base_url=None, timeout=25.0)
        res = backend.generate_structured("Water?", {"type": "object", "properties": {"facts": {}}})
        assert res["facts"][0]["subject"] == "Water"
        rf = mock_client.chat.completions.create.call_args.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert "json_schema" in rf


def test_gemini_backend_sanitizes_schema_without_module_leak():
    mock_google = MagicMock()
    mock_genai = MagicMock()
    mock_model_inst = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"facts": [{"subject": "A", "predicate": "B", "object": "C"}]}'
    mock_model_inst.generate_content.return_value = mock_response
    mock_genai.GenerativeModel.return_value = mock_model_inst
    mock_google.generativeai = mock_genai

    dirty_schema = {
        "type": "object",
        "properties": {
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "additionalProperties": False,
    }
    clean = _sanitize_schema_for_gemini(dirty_schema)
    assert "minimum" not in clean["properties"]["confidence"]
    assert "maximum" not in clean["properties"]["confidence"]
    assert "additionalProperties" not in clean

    with patch.dict(sys.modules, {"google": mock_google, "google.generativeai": mock_genai}):
        backend = GeminiBackend(model="gemini-2.0-flash", api_key="dummy-key", timeout=20.0)
        assert backend.model == "gemini-2.0-flash"

        res = backend.generate_structured("Explain X", schema=dirty_schema)
        assert res["facts"][0]["subject"] == "A"
        gen_cfg = mock_model_inst.generate_content.call_args.kwargs["generation_config"]
        assert "minimum" not in str(gen_cfg["response_schema"])

        mock_response.text = "This is text"
        text, logprob = backend.generate_text_with_logprobs("Explain Y")
        assert text == "This is text"
        assert logprob == 0.0


def _ollama_up():
    try:
        import urllib.request

        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _ollama_up(), reason="Ollama server not available")
def test_live_ollama_probe_smoke():
    pytest.importorskip("ollama")
    pytest.importorskip("pydantic")
    from knowledge_graph_pkg.schemas import PROBE_OUTPUT_SCHEMA

    backend = OllamaBackend(model="qwen2.5:7b")
    probe = ModelProbe(backend=backend, model="qwen2.5:7b")
    outs = probe.probe_domain(
        "biochemistry",
        entities=["Mitochondria"],
        n_prompts=2,
        schema=PROBE_OUTPUT_SCHEMA,
        seed=42,
    )
    assert len(outs) == 2
    assert all("structured_response" in o for o in outs)
