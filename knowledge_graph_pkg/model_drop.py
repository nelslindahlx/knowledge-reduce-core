"""
Model-output ingestion (ModelReduce Session 2).

Converts structured probe outputs into store-ready fact dicts and immutable
:class:`ModelDrop` shards with model provenance.
"""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, List, Optional

from .store import SCHEMA_VERSION, Drop, _utc_now

_ECHO_FOLLOW_TOKENS = frozenset(
    {"is", "are", "was", "were", "only", "does", "do", "can", "has", "have"}
)


def model_fact_statement(
    subject: str, predicate: str, obj: str, qualifier: Optional[str] = None
) -> str:
    """Render a readable sentence from an SPO triple."""
    pred = " ".join(str(predicate or "").replace("_", " ").split())
    subj = str(subject or "").strip()
    o = str(obj or "").strip()
    core = " ".join(p for p in (subj, pred, o) if p)
    statement = core[:1].upper() + core[1:] if core else core
    if not statement:
        return statement
    if not statement.endswith((".", "!", "?")):
        statement += "."
    if qualifier:
        q = str(qualifier).strip().rstrip(".")
        if q:
            statement = f"{statement[:-1]} ({q})."
    return statement


def _confidence_to_quality(confidence: Optional[float]) -> float:
    """Map a model's [0,1] confidence onto a quality score."""
    base = 20.0
    if confidence is None:
        return base
    try:
        c = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        return base
    return base + c * 10.0


def _is_degenerate(
    subject: str, predicate: str, obj: str, domain_norm: str
) -> bool:
    """Reject domain-echo or tautological sentence fillers without flagging valid compounds."""
    s = " ".join(subject.lower().split())
    p = " ".join(predicate.lower().replace("_", " ").split())
    o = " ".join(obj.lower().split())
    if domain_norm and s == domain_norm:
        return True
    if not s or o == s:
        return True
    if p and o.startswith(f"{s} {p}"):
        return True
    if o.startswith(s + " "):
        remainder = o[len(s) + 1 :].strip().split()
        if len(o.split()) >= 4 and remainder and remainder[0] in _ECHO_FOLLOW_TOKENS:
            return True
    return False


def probe_output_to_facts(
    probe_output: Dict[str, Any], reliability: str = "POSSIBLY_TRUE"
) -> List[Dict[str, Any]]:
    """Convert one probe output into a list of KnowledgeReduce fact dicts."""
    if not isinstance(probe_output, dict):
        return []
    model = probe_output.get("model", "unknown")
    backend = probe_output.get("backend", "unknown")
    domain = probe_output.get("domain", "general")
    prompt = probe_output.get("prompt", "")
    prompt_type = probe_output.get("prompt_type", "")
    structured = probe_output.get("structured_response")
    if not isinstance(structured, dict):
        return []
    raw_facts = structured.get("facts")
    if not isinstance(raw_facts, list):
        return []

    domain_str = str(domain or "")
    category = domain_str[:1].upper() + domain_str[1:] if domain_str else "General"
    domain_norm = " ".join(domain_str.lower().split())
    facts: List[Dict[str, Any]] = []
    for rf in raw_facts:
        if not isinstance(rf, dict):
            continue
        subject = str(rf.get("subject") or "").strip()
        predicate = str(rf.get("predicate") or "").strip()
        obj = str(rf.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue
        if _is_degenerate(subject, predicate, obj, domain_norm):
            continue
        qualifier = rf.get("context_or_qualifier")
        confidence = rf.get("confidence")
        statement = model_fact_statement(subject, predicate, obj, qualifier)
        facts.append(
            {
                "fact_statement": statement,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "category": category,
                "reliability_rating": reliability,
                "quality_score": _confidence_to_quality(confidence),
                "question": f"State a verified fact about {subject} in {domain_str}.",
                "answer": statement,
                "model_provenance": {
                    "model": model,
                    "backend": backend,
                    "domain": domain_str,
                    "prompt": prompt,
                    "prompt_type": prompt_type,
                    "confidence": confidence,
                },
            }
        )
    return facts


def probe_outputs_to_facts(
    probe_outputs: List[Dict[str, Any]], reliability: str = "POSSIBLY_TRUE"
) -> List[Dict[str, Any]]:
    """Flatten many probe outputs into one fact list."""
    facts: List[Dict[str, Any]] = []
    for po in probe_outputs:
        facts.extend(probe_output_to_facts(po, reliability=reliability))
    return facts


def model_content_hash(model: str, domain: str, prompts: List[str]) -> str:
    """Content hash that folds in *model identity*."""
    h = hashlib.sha256()
    h.update(f"{model}\x00{domain}\x00".encode("utf-8"))
    for p in prompts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class ModelDrop(Drop):
    """A :class:`Drop` whose source is a *model*, not a text document."""

    model_provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_probe_outputs(
        cls,
        model: str,
        domain: str,
        probe_outputs: List[Dict[str, Any]],
        backend: str = "unknown",
        reliability: str = "POSSIBLY_TRUE",
        gen_config: Optional[Dict[str, Any]] = None,
        drop_id: Optional[str] = None,
    ) -> "ModelDrop":
        """Build a ModelDrop from one model's probe outputs over a domain."""
        facts = probe_outputs_to_facts(probe_outputs, reliability=reliability)
        prompts = [po.get("prompt", "") for po in probe_outputs if isinstance(po, dict)]
        src_hash = model_content_hash(model, domain, prompts)
        if drop_id is None:
            drop_id = f"model-{_safe(model)}-{_safe(domain)}-{src_hash[:12]}"
        resolved_gen_config = gen_config
        if resolved_gen_config is None and probe_outputs and isinstance(probe_outputs[0], dict):
            resolved_gen_config = probe_outputs[0].get("gen_config") or {}
        return cls(
            drop_id=drop_id,
            source=model,
            source_hash=src_hash,
            facts=facts,
            engine="model-probe",
            filter_name="standard",
            coref=False,
            source_text=None,
            model_provenance={
                "model": model,
                "backend": backend,
                "domain": domain,
                "n_prompts": len(prompts),
                "gen_config": resolved_gen_config or {},
            },
            meta={"domain": domain, "n_prompts": len(prompts)},
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["model_provenance"] = self.model_provenance
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelDrop":
        return cls(
            drop_id=d["drop_id"],
            source=d["source"],
            source_hash=d["source_hash"],
            facts=d.get("facts", []),
            engine=d.get("engine", "model-probe"),
            filter_name=d.get("filter_name", "standard"),
            coref=d.get("coref", False),
            created_at=d.get("created_at", _utc_now()),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            source_text=d.get("source_text"),
            meta=d.get("meta", {}),
            model_provenance=d.get("model_provenance", {}),
        )

    def summary(self) -> Dict[str, Any]:
        s = super().summary()
        s["model"] = self.model_provenance.get("model", self.source)
        s["domain"] = self.model_provenance.get("domain")
        return s


def _safe(name: str) -> str:
    """Filesystem-safe, traversal-proof rendering of an identifier."""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
    cleaned = re.sub(r"\.{2,}", "_", cleaned).strip("._")
    return cleaned or "item"
