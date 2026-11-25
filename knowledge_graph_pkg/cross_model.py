"""
Cross-model verification (ModelReduce Session 2).

Corroborates facts independently emitted across multiple models, clusters
equivalent statements (via exact SPO, semantic embeddings, or Jaccard word
overlap), and promotes reliability using model agreement and capability weights.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional

from .embeddings import cosine_similarity as _cosine
from .model_drop import probe_output_to_facts

_AGREEMENT_RELIABILITY = {
    1: "POSSIBLY_TRUE",
    2: "LIKELY_TRUE",
}
_VERIFIED = "VERIFIED"

MODEL_CAPABILITY_WEIGHTS = {
    "gemini-2.5-pro": 3.0,
    "gemini-1.5-pro": 3.0,
    "gpt-4o": 3.0,
    "gpt-4": 3.0,
    "claude-3-5-sonnet": 3.0,
    "gemini-2.0-flash": 2.0,
    "gemini-1.5-flash": 2.0,
    "gpt-4o-mini": 2.0,
    "qwen2.5-14b": 2.0,
    "gpt-3.5-turbo": 1.5,
    "llama3-8b": 1.5,
    "qwen2.5-3b": 1.0,
    "qwen2.5-0.5b": 0.5,
}

_SORTED_CAPABILITY_WEIGHTS = sorted(
    MODEL_CAPABILITY_WEIGHTS.items(), key=lambda kv: (-len(kv[0]), kv[0])
)


def _norm(s: Any) -> str:
    return " ".join(str(s or "").lower().split())


def jaccard(a: str, b: str) -> float:
    """Dependency-free Jaccard word-overlap similarity in ``[0, 1]``."""
    if a == b:
        return 1.0
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def reliability_for_agreement(
    n_models: int, models: Optional[List[str]] = None
) -> str:
    """Map a distinct-model agreement count or weighted model list to a reliability rating."""
    if not models:
        if n_models >= 3:
            return _VERIFIED
        return _AGREEMENT_RELIABILITY.get(max(n_models, 0), "UNVERIFIED")

    total_weight = 0.0
    for m in models:
        weight = 1.0
        ml = str(m or "").strip().lower().replace(":", "-")
        for name, w in _SORTED_CAPABILITY_WEIGHTS:
            if (
                ml == name
                or ml.startswith(name + "-")
                or ml.startswith(name + "_")
                or ml.startswith(name + ".")
            ):
                weight = w
                break
        total_weight += weight

    if total_weight >= 3.0:
        return _VERIFIED
    elif total_weight >= 1.5:
        return "LIKELY_TRUE"
    elif total_weight >= 0.5:
        return "POSSIBLY_TRUE"
    return "UNVERIFIED"


def cluster_facts(
    facts: List[Dict[str, Any]],
    similarity_threshold: float = 0.8,
    embedder: Any = None,
    embed_threshold: float = 0.82,
) -> List[List[Dict[str, Any]]]:
    """Greedy single-pass clustering of facts by statement similarity."""
    use_embed = embedder is not None
    clusters: List[List[Dict[str, Any]]] = []
    reps: List[Dict[str, Any]] = []
    for fact in facts:
        stmt = _norm(fact.get("fact_statement") or fact.get("statement"))
        s = _norm(fact.get("subject"))
        p = _norm(fact.get("predicate"))
        o = _norm(fact.get("object"))
        spo: Optional[str] = "\x00".join((s, p, o)) if (s and p and o) else None

        emb = None
        if use_embed and stmt:
            try:
                vec = embedder.embed_one(stmt)
                if isinstance(vec, list) and len(vec) > 0:
                    emb = vec
            except Exception:
                emb = None

        placed = False
        for idx, rep in enumerate(reps):
            if spo is not None and rep["spo"] == spo:
                clusters[idx].append(fact)
                placed = True
                break
            if emb is not None and rep.get("emb") is not None:
                if _cosine(emb, rep["emb"]) >= embed_threshold:
                    clusters[idx].append(fact)
                    placed = True
                    break
            elif stmt and rep["stmt"] and jaccard(stmt, rep["stmt"]) >= similarity_threshold:
                clusters[idx].append(fact)
                placed = True
                break

        if not placed:
            clusters.append([fact])
            reps.append({"stmt": stmt, "spo": spo, "emb": emb})
    return clusters


class CrossModelVerifier:
    """Probe N models with identical prompts and corroborate their facts."""

    def __init__(
        self,
        probes: List[Any],
        similarity_threshold: float = 0.8,
        embedder: Any = None,
        embed_threshold: float = 0.82,
    ):
        if not probes:
            raise ValueError("CrossModelVerifier needs at least one probe.")
        self.probes = list(probes)
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self.embed_threshold = embed_threshold

    @classmethod
    def from_ollama(
        cls,
        models: List[str],
        host: str = "http://localhost:11434",
        similarity_threshold: float = 0.8,
        embedder: Any = None,
        embed_model: Optional[str] = "mxbai-embed-large",
        embed_threshold: float = 0.82,
    ) -> "CrossModelVerifier":
        """Build a verifier backed by local Ollama models."""
        from .model_probe import ModelProbe, OllamaBackend

        probes = [
            ModelProbe(backend=OllamaBackend(model=m, host=host), model=m)
            for m in models
        ]
        if embedder is None and embed_model is not None:
            try:
                from .embeddings import LocalEmbedder

                embedder = LocalEmbedder(model=embed_model, host=host)
            except Exception:
                embedder = None
        return cls(
            probes,
            similarity_threshold=similarity_threshold,
            embedder=embedder,
            embed_threshold=embed_threshold,
        )

    def probe_domain(
        self,
        domain: str,
        entities: Optional[List[str]] = None,
        n_prompts: int = 10,
        seed: int = 42,
        **gen_kwargs,
    ) -> Dict[str, Any]:
        """Probe every model over ``domain`` with the same prompts, then verify."""
        outputs_by_model: Dict[str, List[Dict[str, Any]]] = OrderedDict()
        for probe in self.probes:
            outs = probe.probe_domain(
                domain, entities=entities, n_prompts=n_prompts, seed=seed, **gen_kwargs
            )
            outputs_by_model.setdefault(probe.model, []).extend(outs)
        return self.verify(outputs_by_model)

    def verify(self, outputs_by_model: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Cluster facts across models and assign agreement-based reliability."""
        all_facts: List[Dict[str, Any]] = []
        for model, outputs in outputs_by_model.items():
            for fact in _facts_from_outputs(outputs):
                fact_copy = dict(fact)
                fact_copy["_model"] = model
                all_facts.append(fact_copy)

        clusters = self._cluster(all_facts)

        report_clusters: List[Dict[str, Any]] = []
        verified = likely = possibly = unverified = 0
        for cl in clusters:
            models = sorted({f["_model"] for f in cl})
            n = len(models)
            reliability = reliability_for_agreement(n, models=models)
            if reliability == _VERIFIED:
                verified += 1
            elif reliability == "LIKELY_TRUE":
                likely += 1
            elif reliability == "POSSIBLY_TRUE":
                possibly += 1
            else:
                unverified += 1

            rep = max(cl, key=lambda f: f.get("quality_score") or 0)
            report_clusters.append(
                {
                    "statement": rep.get("fact_statement") or "",
                    "subject": rep.get("subject"),
                    "predicate": rep.get("predicate"),
                    "object": rep.get("object"),
                    "category": rep.get("category"),
                    "models": models,
                    "n_models": n,
                    "reliability": reliability,
                    "cross_model_agreement": n,
                    "member_count": len(cl),
                }
            )

        report_clusters.sort(
            key=lambda c: (-c["n_models"], -c["member_count"], c["statement"])
        )
        return {
            "clusters": report_clusters,
            "n_clusters": len(report_clusters),
            "verified": verified,
            "likely_true": likely,
            "possibly_true": possibly,
            "unverified": unverified,
            "models": sorted(outputs_by_model.keys()),
            "total_facts": len(all_facts),
        }

    def _cluster(self, facts: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Cluster facts using this verifier's similarity settings."""
        return cluster_facts(
            facts,
            self.similarity_threshold,
            embedder=self.embedder,
            embed_threshold=self.embed_threshold,
        )

    def verified_facts(
        self, report: Dict[str, Any], min_models: int = 2
    ) -> List[Dict[str, Any]]:
        """Extract corroborated facts from a report as store-ready fact dicts."""
        out: List[Dict[str, Any]] = []
        for cl in report["clusters"]:
            if cl["n_models"] < min_models:
                continue
            statement = cl["statement"]
            subject = cl.get("subject") or ""
            category = cl.get("category") or "General"
            reliability = cl["reliability"]
            quality = {"VERIFIED": 40, "LIKELY_TRUE": 30}.get(reliability, 20)
            quality += cl["n_models"]
            out.append(
                {
                    "fact_statement": statement,
                    "subject": subject,
                    "predicate": cl.get("predicate"),
                    "object": cl.get("object"),
                    "category": category,
                    "reliability_rating": reliability,
                    "quality_score": quality,
                    "question": (
                        f"State a verified fact about {subject}."
                        if subject
                        else f"State a verified fact about {category}."
                    ),
                    "answer": statement,
                    "cross_model_agreement": cl["n_models"],
                    "source_models": cl["models"],
                }
            )
        return out


def _facts_from_outputs(outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert a model's probe outputs to facts (POSSIBLY_TRUE per claim)."""
    facts: List[Dict[str, Any]] = []
    for po in outputs:
        facts.extend(probe_output_to_facts(po, reliability="POSSIBLY_TRUE"))
    return facts
