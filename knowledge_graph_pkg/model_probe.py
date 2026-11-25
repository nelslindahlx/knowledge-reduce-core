"""
Model probing engine (ModelReduce Session 1).

:class:`ModelProbe` drives a backend to generate *structured* facts for a
domain: it builds domain-parameterized prompts (see :mod:`probe_templates`),
asks the backend to emit JSON conforming to :data:`schemas.PROBE_OUTPUT_SCHEMA`,
and returns one structured record per prompt with full provenance (model,
backend, domain, prompt, generation config, timestamp).
"""

from datetime import datetime, timezone
import json
import os
import sys
from typing import Any, Dict, List, Optional

from .probe_templates import generate_probes

_GEMINI_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"minimum", "maximum", "additionalProperties", "$schema", "title", "default"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _backend_name(backend: Any) -> str:
    """Infer a short backend label from the backend class name."""
    cls = type(backend).__name__.lower()
    for tag in ("ollama", "fake", "hf", "vllm", "openai", "gemini", "api", "llama"):
        if tag in cls:
            return tag
    return cls.replace("backend", "") or "unknown"


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _parse_structured_json(raw: Any, schema: Optional[dict] = None) -> dict:
    """Parse structured JSON output into a dict with a safe empty fallback."""
    empty: Dict[str, Any] = (
        {"facts": []}
        if (schema is None or "facts" in (schema.get("properties") or {}))
        else {}
    )
    if isinstance(raw, dict):
        if "facts" in empty and "facts" in raw and not isinstance(raw["facts"], list):
            out = dict(raw)
            out["facts"] = []
            return out
        return raw
    if not isinstance(raw, str):
        return empty
    cleaned = _strip_json_fences(raw)
    if not cleaned:
        return empty
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return empty
    if not isinstance(parsed, dict):
        return empty
    if "facts" in empty and "facts" in parsed and not isinstance(parsed["facts"], list):
        parsed["facts"] = []
    return parsed


def _sanitize_schema_for_gemini(schema: Any) -> Any:
    """Recursively remove JSON Schema keys unsupported by Gemini response_schema."""
    if isinstance(schema, dict):
        return {
            k: _sanitize_schema_for_gemini(v)
            for k, v in schema.items()
            if k not in _GEMINI_UNSUPPORTED_SCHEMA_KEYS
        }
    if isinstance(schema, list):
        return [_sanitize_schema_for_gemini(item) for item in schema]
    return schema


def _clean_logprobs(values: Any) -> List[float]:
    """Filter logprob sequences to valid numeric floats."""
    cleaned: List[float] = []
    for v in values or []:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            cleaned.append(float(v))
    return cleaned


class OllamaBackend:
    """Primary backend: local Ollama server with JSON-schema structured output."""

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ):
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - needs the extra
            raise ImportError(
                "OllamaBackend requires the model-reduce extra: "
                "pip install knowledgereduce[model-reduce]"
            ) from exc
        self._ollama = ollama
        self.timeout = timeout
        try:
            self.client = ollama.Client(host=host, timeout=timeout)
        except TypeError:
            self.client = ollama.Client(host=host)
        self.model = model

    def generate_structured(self, prompt: str, schema: dict, **gen_kwargs) -> dict:
        """Generate one structured response, enforcing ``schema`` via Ollama ``format``."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            format=schema,
            options={
                "temperature": gen_kwargs.get("temperature", 0.3),
                "top_p": gen_kwargs.get("top_p", 0.9),
                "num_predict": gen_kwargs.get("max_tokens", 512),
                "seed": gen_kwargs.get("seed", 42),
            },
        )
        text = (
            response.get("response")
            if isinstance(response, dict)
            else getattr(response, "response", "")
        )
        return _parse_structured_json(text, schema)

    def generate_text_with_logprobs(self, prompt: str, **gen_kwargs) -> tuple:
        """Generate text and return the generated text along with its average log probability."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": gen_kwargs.get("temperature", 0.3),
                "top_p": gen_kwargs.get("top_p", 0.9),
                "num_predict": gen_kwargs.get("max_tokens", 512),
            },
        )
        text = (
            response.get("response")
            if isinstance(response, dict)
            else getattr(response, "response", "")
        )
        return ("" if text is None else str(text)), 0.0


class ModelProbe:
    """Probe a model for structured facts across a domain."""

    def __init__(self, backend: Any, model: Optional[str] = None):
        self.backend = backend
        self.model = model or getattr(backend, "model", "unknown")
        self.backend_name = _backend_name(backend)

    def probe_domain(
        self,
        domain: str,
        entities: Optional[List[str]] = None,
        n_prompts: int = 10,
        schema: Optional[dict] = None,
        seed: int = 42,
        **gen_kwargs,
    ) -> List[Dict[str, Any]]:
        """Generate ``n_prompts`` structured probes for ``domain``."""
        if schema is None:
            from .schemas import PROBE_OUTPUT_SCHEMA

            schema = PROBE_OUTPUT_SCHEMA

        gen_config = {
            "temperature": gen_kwargs.get("temperature", 0.3),
            "top_p": gen_kwargs.get("top_p", 0.9),
            "max_tokens": gen_kwargs.get("max_tokens", 512),
            "seed": seed,
        }

        probes = generate_probes(
            domain, entities or [], n_prompts=n_prompts, seed=seed
        )
        outputs: List[Dict[str, Any]] = []
        for idx, spec in enumerate(probes):
            probe_seed = seed + idx
            raw_structured = self.backend.generate_structured(
                spec["prompt"], schema, seed=probe_seed, **gen_kwargs
            )
            structured = _parse_structured_json(raw_structured, schema)
            outputs.append(
                {
                    "model": self.model,
                    "backend": self.backend_name,
                    "domain": domain,
                    "prompt_type": spec["prompt_type"],
                    "prompt": spec["prompt"],
                    "structured_response": structured,
                    "gen_config": gen_config,
                    "timestamp": _utc_now(),
                }
            )
        return outputs


class LlamaCppBackend:
    """Local GGUF model execution via llama-cpp-python."""

    def __init__(self, model_path: str, n_ctx: int = 2048, **kwargs):
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("LlamaCppBackend requires a non-empty model_path.")
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "LlamaCppBackend requires the llama-cpp extra: "
                "pip install knowledgereduce[llama-cpp]"
            ) from exc
        kwargs.setdefault("logits_all", True)
        self.model = os.path.basename(model_path)
        self.model_path = model_path
        self.client = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kwargs)

    def generate_structured(self, prompt: str, schema: dict, **gen_kwargs) -> dict:
        """Generate one structured response, enforcing JSON schema via llama-cpp."""
        try:
            response = self.client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object", "schema": schema},
                temperature=gen_kwargs.get("temperature", 0.3),
                top_p=gen_kwargs.get("top_p", 0.9),
                max_tokens=gen_kwargs.get("max_tokens", 512),
            )
            text = response["choices"][0]["message"]["content"]
            return _parse_structured_json(text, schema)
        except Exception as exc:
            print(f"[LlamaCppBackend] generate_structured failed: {exc}", file=sys.stderr)
            return _parse_structured_json(None, schema)

    def generate_text_with_logprobs(self, prompt: str, **gen_kwargs) -> tuple:
        """Generate text and return the generated text along with its average log probability."""
        try:
            response = self.client.create_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=gen_kwargs.get("temperature", 0.3),
                top_p=gen_kwargs.get("top_p", 0.9),
                max_tokens=gen_kwargs.get("max_tokens", 512),
                logprobs=True,
            )
            choice = response["choices"][0]
            text = choice["message"].get("content")
            text_str = "" if text is None else str(text)
            logprobs = choice.get("logprobs") or {}
            token_logprobs: List[float] = []
            if isinstance(logprobs, dict):
                token_logprobs = _clean_logprobs(logprobs.get("token_logprobs"))
                if not token_logprobs and logprobs.get("content"):
                    raw = [
                        item.get("logprob")
                        for item in logprobs["content"]
                        if isinstance(item, dict)
                    ]
                    token_logprobs = _clean_logprobs(raw)
            avg_logprob = (
                sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0
            )
            return text_str, avg_logprob
        except Exception as exc:
            print(f"[LlamaCppBackend] generate_text failed: {exc}", file=sys.stderr)
            return "", 0.0


class OpenAICompatibleBackend:
    """Remote API probing (OpenAI, Anthropic, Cohere, local vLLM)."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "OpenAICompatibleBackend requires the openai extra: "
                "pip install knowledgereduce[openai]"
            ) from exc
        key = api_key or os.environ.get("OPENAI_API_KEY")
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not key and not url:
            raise ValueError(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable "
                "or pass api_key to the backend constructor."
            )
        self.timeout = timeout
        try:
            self.client = openai.OpenAI(
                api_key=key or "mock-key",
                base_url=url,
                timeout=timeout,
            )
        except TypeError:
            self.client = openai.OpenAI(
                api_key=key or "mock-key",
                base_url=url,
            )
        self.model = model

    def generate_structured(self, prompt: str, schema: dict, **gen_kwargs) -> dict:
        """Generate one structured response, enforcing JSON schema via OpenAI SDK."""
        response_format: Dict[str, Any]
        if schema:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "schema": schema,
                    "strict": False,
                },
            }
        else:
            response_format = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format,
                temperature=gen_kwargs.get("temperature", 0.3),
                top_p=gen_kwargs.get("top_p", 0.9),
                max_tokens=gen_kwargs.get("max_tokens", 512),
            )
            text = response.choices[0].message.content
            return _parse_structured_json(text, schema)
        except Exception as exc:
            print(f"[OpenAIBackend] generate_structured failed: {exc}", file=sys.stderr)
            return _parse_structured_json(None, schema)

    def generate_text_with_logprobs(self, prompt: str, **gen_kwargs) -> tuple:
        """Generate text and return the generated text along with its average log probability."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=gen_kwargs.get("temperature", 0.3),
                top_p=gen_kwargs.get("top_p", 0.9),
                max_tokens=gen_kwargs.get("max_tokens", 512),
                logprobs=True,
            )
            text = response.choices[0].message.content
            text_str = "" if text is None else str(text)
            logprobs_content = getattr(
                getattr(response.choices[0], "logprobs", None), "content", None
            )
            token_logprobs: List[float] = []
            if logprobs_content:
                token_logprobs = _clean_logprobs(
                    [getattr(item, "logprob", None) for item in logprobs_content]
                )
            avg_logprob = (
                sum(token_logprobs) / len(token_logprobs) if token_logprobs else 0.0
            )
            return text_str, avg_logprob
        except Exception as exc:
            print(f"[OpenAIBackend] generate_text failed: {exc}", file=sys.stderr)
            return "", 0.0


def get_backend(backend_type: str, model: Optional[str] = None, **kwargs) -> Any:
    """Factory function to instantiate the correct model probing backend."""
    kind = str(backend_type or "").lower()
    host = kwargs.pop("host", None)
    model_path = kwargs.pop("model_path", None)
    api_key = kwargs.pop("api_key", None)
    base_url = kwargs.pop("base_url", None)
    timeout = kwargs.pop("timeout", 60.0)

    if kind == "ollama":
        if not model:
            raise ValueError("OllamaBackend requires a model name.")
        return OllamaBackend(
            model=model,
            host=host or "http://localhost:11434",
            timeout=timeout,
        )
    elif kind == "llama-cpp":
        resolved_path = model_path or model
        if not resolved_path:
            raise ValueError("LlamaCppBackend requires model_path or model.")
        return LlamaCppBackend(model_path=resolved_path, **kwargs)
    elif kind == "openai":
        if not model:
            raise ValueError("OpenAICompatibleBackend requires a model name.")
        return OpenAICompatibleBackend(
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
    elif kind == "gemini":
        return GeminiBackend(
            model=model or "gemini-2.0-flash",
            api_key=api_key,
            timeout=timeout,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


class GeminiBackend:
    """Official Google Gemini API backend."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        try:
            import google.generativeai as genai
        except ImportError as exc:
            raise ImportError(
                "GeminiBackend requires the google-generativeai package: "
                "pip install google-generativeai"
            ) from exc
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable "
                "or pass api_key to the backend constructor."
            )
        genai.configure(api_key=key)
        self.client = genai.GenerativeModel(model)
        self.model = model
        self.timeout = timeout

    def generate_structured(self, prompt: str, schema: dict, **gen_kwargs) -> dict:
        """Generate one structured response, enforcing schema via Gemini response_schema."""
        gen_config: Dict[str, Any] = {
            "response_mime_type": "application/json",
            "temperature": gen_kwargs.get("temperature", 0.3),
        }
        if schema:
            gen_config["response_schema"] = _sanitize_schema_for_gemini(schema)
        try:
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config=gen_config,
                    request_options={"timeout": self.timeout},
                )
            except TypeError:
                response = self.client.generate_content(
                    prompt,
                    generation_config=gen_config,
                )
            text = getattr(response, "text", None)
            return _parse_structured_json(text, schema)
        except Exception as exc:
            print(f"[GeminiBackend] generate_structured failed: {exc}", file=sys.stderr)
            return _parse_structured_json(None, schema)

    def generate_text_with_logprobs(self, prompt: str, **gen_kwargs) -> tuple:
        """Generate text. Gemini standard SDK does not expose logprobs on generate_content."""
        gen_config = {"temperature": gen_kwargs.get("temperature", 0.3)}
        try:
            try:
                response = self.client.generate_content(
                    prompt,
                    generation_config=gen_config,
                    request_options={"timeout": self.timeout},
                )
            except TypeError:
                response = self.client.generate_content(
                    prompt,
                    generation_config=gen_config,
                )
            text = getattr(response, "text", None)
            return ("" if text is None else str(text)), 0.0
        except Exception as exc:
            print(f"[GeminiBackend] generate_text failed: {exc}", file=sys.stderr)
            return "", 0.0
