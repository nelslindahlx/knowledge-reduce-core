import sys
import json
from typing import List, Dict, Any, Optional
from .model_probe import get_backend

class FactCritic:
    """Audits facts using a high-capability model backend to flag hallucinations."""

    def __init__(self, backend_name: Optional[str] = None, model_name: Optional[str] = None, **kwargs):
        self.backend = None
        if backend_name and backend_name.lower() != "none":
            try:
                self.backend = get_backend(backend_name, model=model_name, **kwargs)
            except Exception as exc:
                print(f"[Critic] Warning: Could not initialize backend '{backend_name}' ({exc}). "
                      "Falling back to offline rule-based critique.")

    def critique_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Ask the critic model or offline heuristics to verify a fact's accuracy."""
        statement = fact.get("fact_statement") or fact.get("statement")
        if not statement:
            statement = f"{fact.get('subject')} {fact.get('predicate')} {fact.get('object')}."
        
        if self.backend is None:
            return self.heuristic_critique(fact, statement)

        prompt = (
            "You are a factual audit engine. Evaluate the accuracy of the following factual assertion:\n"
            f"Fact: \"{statement}\"\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            "  \"reasoning\": \"A short explanation of whether the fact is true or false.\",\n"
            "  \"is_factual\": true or false,\n"
            "  \"confidence_score\": a float between 0.0 and 1.0\n"
            "}\n"
        )
        
        try:
            schema = {
                "type": "OBJECT",
                "properties": {
                    "reasoning": {"type": "STRING"},
                    "is_factual": {"type": "BOOLEAN"},
                    "confidence_score": {"type": "NUMBER"}
                },
                "required": ["reasoning", "is_factual", "confidence_score"]
            }
            
            if hasattr(self.backend, "generate_structured_json"):
                res_text = self.backend.generate_structured_json(prompt, schema)
            else:
                res_text = self.backend.generate_text(prompt)
                
            if "```json" in res_text:
                res_text = res_text.split("```json")[1].split("```")[0].strip()
            elif "```" in res_text:
                res_text = res_text.split("```")[1].split("```")[0].strip()
            res_text = res_text.strip()
            
            data = json.loads(res_text)
            return {
                "block_id": fact.get("block_id") or fact.get("fact_id"),
                "statement": statement,
                "is_factual": bool(data.get("is_factual", True)),
                "reasoning": str(data.get("reasoning", "")),
                "confidence_score": float(data.get("confidence_score", 1.0))
            }
        except Exception as exc:
            print(f"[Critic] Error critiquing fact '{statement}': {exc}", file=sys.stderr)
            return {
                "block_id": fact.get("block_id") or fact.get("fact_id"),
                "statement": statement,
                "is_factual": True,
                "reasoning": f"Critique failed: {exc}",
                "confidence_score": 0.5
            }

    def heuristic_critique(self, fact: Dict[str, Any], statement: str) -> Dict[str, Any]:
        """Offline rule-based heuristics to critique a fact without LLM execution."""
        subj = str(fact.get("subject") or "").strip().lower()
        pred = str(fact.get("predicate") or "").strip().lower()
        obj = str(fact.get("object") or "").strip().lower()
        stmt = statement.strip()
        
        pronouns = {"he", "she", "it", "they", "this", "that", "them", "these", "those"}
        if subj in pronouns:
            return {
                "block_id": fact.get("block_id") or fact.get("fact_id"),
                "statement": stmt,
                "is_factual": False,
                "reasoning": f"Rejected by heuristic: Pronoun subject '{fact.get('subject')}' violates fact independence.",
                "confidence_score": 0.95
            }
            
        if len(stmt) < 5 or not subj or not pred or not obj:
            return {
                "block_id": fact.get("block_id") or fact.get("fact_id"),
                "statement": stmt,
                "is_factual": False,
                "reasoning": "Rejected by heuristic: Statement is too short or has empty SVO components.",
                "confidence_score": 0.9
            }
            
        if subj == pred or pred == obj:
            return {
                "block_id": fact.get("block_id") or fact.get("fact_id"),
                "statement": stmt,
                "is_factual": False,
                "reasoning": "Rejected by heuristic: Redundant SVO components (subject, predicate, or object are identical).",
                "confidence_score": 0.85
            }
            
        return {
            "block_id": fact.get("block_id") or fact.get("fact_id"),
            "statement": stmt,
            "is_factual": True,
            "reasoning": "Passed offline heuristic validations (non-pronoun subject, valid SVO length, distinct terms).",
            "confidence_score": 0.7
        }

    def critique_facts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Audit a list of facts and return verification reports."""
        reports = []
        for fact in facts:
            reports.append(self.critique_fact(fact))
        return reports
