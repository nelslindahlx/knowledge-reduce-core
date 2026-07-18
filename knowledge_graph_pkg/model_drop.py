"""
Model-output ingestion (ModelReduce Session 2).

Session 1 produced *structured probe outputs* -- one record per prompt, each
carrying a model's emitted facts plus provenance (model/backend/domain/prompt/
gen_config/timestamp). This module turns those outputs into the corpus's
native currency: **drops** of **facts**, so the existing distillation,
lifecycle, and store machinery applies unchanged.

Two pieces:

* :func:`probe_output_to_facts` -- convert one probe output's structured
  facts into KnowledgeReduce fact dicts (``fact_statement``/``subject``/
  ``predicate``/``object``/``reliability_rating``/``quality_score``/...),
  stamping each with its model provenance so cross-model agreement can be
  computed later.
* :class:`ModelDrop` -- a :class:`~knowledge_graph_pkg.store.Drop` subclass
  that records *model* provenance (model identity, backend, domain, the
  probe prompts) rather than a text source. Its content hash folds in the
  model identity, so the same prompt answered by two different models yields
  two distinct drops (the whole point of cross-model corroboration).

A model claim is *unverified* by construction: a single model asserting
something is not evidence it is true. Facts therefore enter at
``POSSIBLY_TRUE`` and only climb the reliability ladder once multiple
independent models agree (see :mod:`cross_model`).
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .store import Drop, SCHEMA_VERSION, _json_safe, _utc_now


def model_fact_statement(subject: str, predicate: str, obj: str,
                         qualifier: Optional[str] = None) -> str:
    """Render a readable sentence from an SPO triple.

    Predicates from probes/extractors are often underscore-joined tokens
    (``born_in``, ``is_a``); we humanize them back to spaces. An optional
    qualifier is appended parenthetically.
    """
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
    """Map a model's [0,1] confidence onto a quality score.

    Mirrors the graph's ``reliability.value * 10`` scale (POSSIBLY_TRUE = 2
    => base 20) so model facts rank sensibly alongside text-extracted ones.
    Confidence nudges within the tier rather than overriding reliability.
    """
    base = 20.0  # POSSIBLY_TRUE tier baseline
    if confidence is None:
        return base
    try:
        c = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        return base
    return base + c * 10.0


def _is_degenerate(subject: str, predicate: str, obj: str, domain_norm: str) -> bool:
    """Reject low-signal "facts" that probes (esp. list/negative) emit.

    Two common failure modes from forcing prose answers into the SVO schema:

    * **domain echo** -- the model just restates the domain as the subject
      ("Biochemistry / common misconception / ..."), which carries no atomic
      fact;
    * **subject echoed in object** -- the object simply repeats the subject
      ("Chemistry is a science"), i.e. a tautological/run-on filler.
    """
    s = " ".join(subject.lower().split())
    o = " ".join(obj.lower().split())
    if domain_norm and s == domain_norm:
        return True
    # object leads with the subject token(s) -> echo / run-on, not an SVO fact
    if s and (o == s or o.startswith(s + " ")):
        return True
    return False


def probe_output_to_facts(probe_output: Dict[str, Any],
                          reliability: str = "POSSIBLY_TRUE") -> List[Dict[str, Any]]:
    """Convert one probe output into a list of KnowledgeReduce fact dicts.

    ``probe_output`` is a record produced by
    :meth:`ModelProbe.probe_domain`: it has ``model``/``backend``/``domain``/
    ``prompt``/``prompt_type`` and a ``structured_response`` of the form
    ``{"facts": [{subject, predicate, object, context_or_qualifier?,
    confidence?}, ...]}``.

    Each emitted fact is store-ready and carries a ``model_provenance`` block
    plus a generated ``question``/``answer`` pair (so the distiller's chat/
    instruction serializers produce natural training records).
    """
    model = probe_output.get("model", "unknown")
    backend = probe_output.get("backend", "unknown")
    domain = probe_output.get("domain", "general")
    prompt = probe_output.get("prompt", "")
    prompt_type = probe_output.get("prompt_type", "")
    structured = probe_output.get("structured_response") or {}
    raw_facts = structured.get("facts") or []

    category = domain[:1].upper() + domain[1:] if domain else "General"
    domain_norm = " ".join(str(domain or "").lower().split())
    facts: List[Dict[str, Any]] = []
    for rf in raw_facts:
        subject = (rf.get("subject") or "").strip()
        predicate = (rf.get("predicate") or "").strip()
        obj = (rf.get("object") or "").strip()
        if not subject or not predicate or not obj:
            continue  # SPO is mandatory; skip malformed emissions
        if _is_degenerate(subject, predicate, obj, domain_norm):
            continue  # reject domain-echo / subject-in-object noise
        qualifier = rf.get("context_or_qualifier")
        confidence = rf.get("confidence")
        statement = model_fact_statement(subject, predicate, obj, qualifier)
        facts.append({
            "fact_statement": statement,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "category": category,
            "reliability_rating": reliability,
            "quality_score": _confidence_to_quality(confidence),
            "question": f"State a verified fact about {subject} in {domain}.",
            "answer": statement,
            "model_provenance": {
                "model": model,
                "backend": backend,
                "domain": domain,
                "prompt": prompt,
                "prompt_type": prompt_type,
                "confidence": confidence,
            },
        })
    return facts


def probe_outputs_to_facts(probe_outputs: List[Dict[str, Any]],
                           reliability: str = "POSSIBLY_TRUE") -> List[Dict[str, Any]]:
    """Flatten many probe outputs into one fact list (convenience wrapper)."""
    facts: List[Dict[str, Any]] = []
    for po in probe_outputs:
        facts.extend(probe_output_to_facts(po, reliability=reliability))
    return facts


def model_content_hash(model: str, domain: str, prompts: List[str]) -> str:
    """Content hash that folds in *model identity*.

    The same probe set answered by two models must produce two distinct
    drops, so the hash combines model + domain + the ordered prompts.
    """
    h = hashlib.sha256()
    h.update(f"{model}\x00{domain}\x00".encode("utf-8"))
    for p in prompts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class ModelDrop(Drop):
    """A :class:`Drop` whose source is a *model*, not a text document.

    Adds a ``model_provenance`` block (model identity, backend, domain, the
    probe prompts, generation config). The ``engine`` is fixed to
    ``"model-probe"`` so model-derived drops are distinguishable in the
    manifest, and ``source`` is set to the model name by convention.
    """

    model_provenance: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_probe_outputs(cls, model: str, domain: str,
                           probe_outputs: List[Dict[str, Any]],
                           backend: str = "unknown",
                           reliability: str = "POSSIBLY_TRUE",
                           gen_config: Optional[Dict[str, Any]] = None,
                           drop_id: Optional[str] = None) -> "ModelDrop":
        """Build a ModelDrop from one model's probe outputs over a domain."""
        facts = probe_outputs_to_facts(probe_outputs, reliability=reliability)
        prompts = [po.get("prompt", "") for po in probe_outputs]
        src_hash = model_content_hash(model, domain, prompts)
        if drop_id is None:
            drop_id = f"model-{_safe(model)}-{domain}-{src_hash[:12]}"
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
                "gen_config": gen_config or {},
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
    """Filesystem-safe rendering of a model name (``qwen2.5:14b`` -> ...)."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(name))
