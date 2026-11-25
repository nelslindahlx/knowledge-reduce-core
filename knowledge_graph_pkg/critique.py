"""Fact auditing and hallucination critique via LLM backends or offline heuristics."""

import json
import sys
from typing import Any, Dict, List, Optional

from .graph_store_base import _s
from .model_probe import _strip_json_fences, get_backend

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "is_factual": {"type": "boolean"},
        "confidence_score": {"type": "number"},
    },
    "required": ["reasoning", "is_factual", "confidence_score"],
}


def _parse_bool(val: Any) -> bool:
    """Strictly parse booleans including string literals like 'false'/'true'."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        norm = val.strip().lower()
        if norm in ("true", "1", "yes"):
            return True
        if norm in ("false", "0", "no"):
            return False
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(val)
    raise ValueError(f"Invalid boolean value for is_factual: {val!r}")


class FactCritic:
    """Audits facts using a high-capability model backend to flag hallucinations."""

    def __init__(
        self,
        backend_name: Optional[str] = None,
        model_name: Optional[str] = None,
        fallback_to_heuristics: bool = True,
        **kwargs,
    ):
        self.backend = None
        if backend_name and backend_name.lower() != "none":
            try:
                self.backend = get_backend(backend_name, model=model_name, **kwargs)
            except ImportError as exc:
                if not fallback_to_heuristics:
                    raise
                print(
                    f"[Critic] Warning: Could not initialize backend '{backend_name}' ({exc}). "
                    "Falling back to offline rule-based critique."
                )

    def critique_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the critic model or offline heuristics to verify a fact's accuracy."""
        statement = fact.get("fact_statement") or fact.get("statement")
        if not statement:
            parts = [
                _s(fact.get("subject")).strip(),
                _s(fact.get("predicate")).strip(),
                _s(fact.get("object")).strip(),
            ]
            joined = " ".join(p for p in parts if p)
            statement = f"{joined}." if joined else ""

        if self.backend is None:
            return self.heuristic_critique(fact, _s(statement))

        prompt = (
            "You are a factual audit engine. Evaluate the accuracy of the following factual assertion:\n"
            f'Fact: "{statement}"\n\n'
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "reasoning": "A short explanation of whether the fact is true or false.",\n'
            '  "is_factual": true or false,\n'
            '  "confidence_score": a float between 0.0 and 1.0\n'
            "}\n"
        )

        block_id = fact.get("block_id") or fact.get("fact_id")
        try:
            raw_res: Any
            if hasattr(self.backend, "generate_structured"):
                raw_res = self.backend.generate_structured(prompt, CRITIQUE_SCHEMA)
            elif hasattr(self.backend, "generate_structured_json"):
                raw_res = self.backend.generate_structured_json(prompt, CRITIQUE_SCHEMA)
            elif hasattr(self.backend, "generate_text_with_logprobs"):
                raw_text, _ = self.backend.generate_text_with_logprobs(prompt)
                raw_res = raw_text
            else:
                raw_res = self.backend.generate_text(prompt)

            if isinstance(raw_res, dict):
                data = raw_res
            elif isinstance(raw_res, str):
                data = json.loads(_strip_json_fences(raw_res))
            else:
                raise ValueError(f"Unsupported backend response type: {type(raw_res).__name__}")

            if not isinstance(data, dict) or "is_factual" not in data:
                raise ValueError("Critique payload missing 'is_factual' field")

            is_factual = _parse_bool(data["is_factual"])
            conf = float(data.get("confidence_score", 1.0))
            conf = max(0.0, min(1.0, conf))
            return {
                "block_id": block_id,
                "statement": statement,
                "is_factual": is_factual,
                "reasoning": _s(data.get("reasoning")),
                "confidence_score": conf,
            }
        except Exception as exc:
            print(f"[Critic] Error critiquing fact '{statement}': {exc}", file=sys.stderr)
            return {
                "block_id": block_id,
                "statement": statement,
                "is_factual": False,
                "reasoning": f"Critique failed: {exc}",
                "confidence_score": 0.0,
                "error": True,
            }

    def heuristic_critique(self, fact: Dict[str, Any], statement: str) -> Dict[str, Any]:
        """Offline rule-based heuristics to critique a fact without LLM execution."""
        subj = _s(fact.get("subject")).strip().lower()
        pred = _s(fact.get("predicate")).strip().lower()
        obj = _s(fact.get("object")).strip().lower()
        stmt = _s(statement).strip()
        block_id = fact.get("block_id") or fact.get("fact_id")

        pronouns = {"he", "she", "it", "they", "this", "that", "them", "these", "those"}
        if subj in pronouns or obj in pronouns:
            rejected_term = fact.get("subject") if subj in pronouns else fact.get("object")
            role = "subject" if subj in pronouns else "object"
            return {
                "block_id": block_id,
                "statement": stmt,
                "is_factual": False,
                "reasoning": f"Rejected by heuristic: Pronoun {role} '{rejected_term}' violates fact independence.",
                "confidence_score": 0.95,
            }

        if len(stmt) < 5 or not subj or not pred or not obj:
            return {
                "block_id": block_id,
                "statement": stmt,
                "is_factual": False,
                "reasoning": "Rejected by heuristic: Statement is too short or has empty SVO components.",
                "confidence_score": 0.9,
            }

        if subj == pred or pred == obj or subj == obj:
            return {
                "block_id": block_id,
                "statement": stmt,
                "is_factual": False,
                "reasoning": "Rejected by heuristic: Redundant SVO components (subject, predicate, or object are identical).",
                "confidence_score": 0.85,
            }

        return {
            "block_id": block_id,
            "statement": stmt,
            "is_factual": True,
            "reasoning": "Passed offline heuristic validations (non-pronoun subject, valid SVO length, distinct terms).",
            "confidence_score": 0.7,
        }

    def critique_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Audit a list of facts and return verification reports."""
        return [self.critique_fact(fact) for fact in facts]
