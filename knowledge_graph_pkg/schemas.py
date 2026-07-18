"""
ModelReduce schemas: structured-output contract for model probing.

Instead of logging raw model strings and running post-hoc extraction, we
force models to emit structured facts *during* generation via Ollama's
native ``format`` parameter (JSON Schema enforcement). :data:`PROBE_OUTPUT_SCHEMA`
is that schema; :class:`ProbeFact` / :class:`ProbeOutput` validate and parse
the responses.

Pydantic is an optional dependency (the ``model-reduce`` extra); importing
this module without it raises a helpful error.
"""

try:
    from pydantic import BaseModel, Field, ConfigDict
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "ModelReduce schemas require pydantic: pip install knowledgereduce[model-reduce]"
    ) from exc

from typing import List, Optional


# JSON Schema handed to Ollama's `format` parameter to enforce structured
# output during the forward pass. Keep in sync with ProbeOutput below.
PROBE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "context_or_qualifier": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["subject", "predicate", "object"],
            },
        }
    },
    "required": ["facts"],
}


class ProbeFact(BaseModel):
    """A single structured fact emitted by a probed model."""

    model_config = ConfigDict(extra="ignore")

    subject: str
    predicate: str
    object: str
    context_or_qualifier: Optional[str] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class ProbeOutput(BaseModel):
    """The structured response from one probe (a list of facts)."""

    model_config = ConfigDict(extra="ignore")

    facts: List[ProbeFact] = Field(default_factory=list)
