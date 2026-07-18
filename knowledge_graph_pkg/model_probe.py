"""
Model probing engine (ModelReduce Session 1).

:class:`ModelProbe` drives a backend to generate *structured* facts for a
domain: it builds domain-parameterized prompts (see :mod:`probe_templates`),
asks the backend to emit JSON conforming to :data:`schemas.PROBE_OUTPUT_SCHEMA`,
and returns one structured record per prompt with full provenance (model,
backend, domain, prompt, generation config, timestamp).

v1 ships :class:`OllamaBackend` (local, free, private). The backend is a
simple protocol -- anything with ``generate_structured(prompt, schema,
**kw) -> dict`` and a ``model`` attribute works -- so tests inject a fake
and future sessions can add HF/vLLM/API backends behind the same contract.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .probe_templates import generate_probes


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _backend_name(backend: Any) -> str:
    """Infer a short backend label from the backend class name."""
    cls = type(backend).__name__.lower()
    for tag in ("ollama", "fake", "hf", "vllm", "openai", "api", "llama"):
        if tag in cls:
            return tag
    return cls.replace("backend", "") or "unknown"


class OllamaBackend:
    """Primary backend: local Ollama server with JSON-schema structured output."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise ImportError(
                "OllamaBackend requires the model-reduce extra: "
                "pip install knowledgereduce[model-reduce]"
            ) from exc
        self._ollama = ollama
        self.client = ollama.Client(host=host)
        self.model = model

    def generate_structured(self, prompt: str, schema: dict, **gen_kwargs) -> dict:
        """Generate one structured response, enforcing ``schema`` via Ollama
        ``format``. Returns the parsed dict (``{"facts": [...]}``)."""
        import json
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            format=schema,  # native JSON-schema enforcement
            options={
                "temperature": gen_kwargs.get("temperature", 0.3),
                "top_p": gen_kwargs.get("top_p", 0.9),
                "num_predict": gen_kwargs.get("max_tokens", 512),
                "seed": gen_kwargs.get("seed", 42),
            },
        )
        text = response["response"] if isinstance(response, dict) else response.response
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return {"facts": []}


class ModelProbe:
    """Probe a model for structured facts across a domain."""

    def __init__(self, backend: Any, model: Optional[str] = None):
        self.backend = backend
        self.model = model or getattr(backend, "model", "unknown")
        self.backend_name = _backend_name(backend)

    def probe_domain(self, domain: str, entities: Optional[List[str]] = None,
                     n_prompts: int = 10, schema: Optional[dict] = None,
                     seed: int = 42, **gen_kwargs) -> List[Dict[str, Any]]:
        """Generate ``n_prompts`` structured probes for ``domain``.

        Returns a list of provenance-stamped records, each containing the
        prompt, the backend's structured response, and generation config.
        """
        if schema is None:
            from .schemas import PROBE_OUTPUT_SCHEMA
            schema = PROBE_OUTPUT_SCHEMA

        gen_config = {
            "temperature": gen_kwargs.get("temperature", 0.3),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "max_tokens": gen_kwargs.get("max_tokens", 512),
            "seed": seed,
        }

        probes = generate_probes(domain, entities or [], n_prompts=n_prompts, seed=seed)
        outputs: List[Dict[str, Any]] = []
        for spec in probes:
            structured = self.backend.generate_structured(
                spec["prompt"], schema, seed=seed, **gen_kwargs
            )
            outputs.append({
                "model": self.model,
                "backend": self.backend_name,
                "domain": domain,
                "prompt_type": spec["prompt_type"],
                "prompt": spec["prompt"],
                "structured_response": structured,
                "gen_config": gen_config,
                "timestamp": _utc_now(),
            })
        return outputs


class LlamaCppBackend:
    """Local GGUF model execution via llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 2048, **kwargs):
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "LlamaCppBackend requires the llama-cpp extra: "
                "pip install knowledgereduce[llama-cpp]"
            ) from exc
        import os
        self.model = os.path.basename(model_path)
        self.model_path = model_path
        self.client = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kwargs)

    def generate_structured(self, prompt: str, schema: dict, **gen_kwargs) -> dict:
        """Generate one structured response, enforcing JSON schema via llama-cpp."""
        import json
        try:
            response = self.client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_object",
                    "schema": schema
                },
                temperature=gen_kwargs.get("temperature", 0.3),
                top_p=gen_kwargs.get("top_p", 0.9),
                max_tokens=gen_kwargs.get("max_tokens", 512),
            )
            text = response["choices"][0]["message"]["content"]
            return json.loads(text)
        except Exception:
            return {"facts": []}


class OpenAICompatibleBackend:
    """Remote API probing (OpenAI, Anthropic, Cohere, local vLLM)."""

    def __init__(self, model: str, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleBackend requires the openai extra: "
                "pip install knowledgereduce[openai]"
            ) from exc
        import os
        self.client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "mock-key"),
            base_url=base_url or os.environ.get("OPENAI_BASE_URL")
        )
        self.model = model

    def generate_structured(self, prompt: str, schema: dict, **gen_kwargs) -> dict:
        """Generate one structured response, enforcing JSON schema via OpenAI SDK."""
        import json
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={
                    "type": "json_object",
                    "schema": schema
                },
                temperature=gen_kwargs.get("temperature", 0.3),
                top_p=gen_kwargs.get("top_p", 0.9),
                max_tokens=gen_kwargs.get("max_tokens", 512),
            )
            text = response.choices[0].message.content
            return json.loads(text)
        except Exception:
            return {"facts": []}


def get_backend(backend_type: str, model: str, **kwargs) -> Any:
    """Factory function to instantiate the correct model probing backend."""
    backend_type = backend_type.lower()
    if backend_type == "ollama":
        return OllamaBackend(model=model, host=kwargs.get("host", "http://localhost:11434"))
    elif backend_type == "llama-cpp":
        return LlamaCppBackend(model_path=kwargs.get("model_path") or model)
    elif backend_type == "openai":
        return OpenAICompatibleBackend(
            model=model,
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url")
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
