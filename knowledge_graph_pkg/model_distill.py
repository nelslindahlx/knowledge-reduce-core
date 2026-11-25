"""
Distillation pipeline for model-derived knowledge (ModelReduce Session 3).

Reads model facts carrying ``model_provenance``, corroborates them across
models via :mod:`cross_model`, promotes reliability, filters, deduplicates,
ranks, and serializes training shards.
"""

import json
import os
from typing import Any, Dict, List, Optional

from .cross_model import cluster_facts, reliability_for_agreement
from .quality import FactQualityFilter

_RELIABILITY_ORDER = ["UNVERIFIED", "POSSIBLY_TRUE", "LIKELY_TRUE", "VERIFIED"]
_RELIABILITY_QUALITY = {
    "UNVERIFIED": 10,
    "POSSIBLY_TRUE": 20,
    "LIKELY_TRUE": 30,
    "VERIFIED": 40,
}


def _models_in_fact(fact: Dict[str, Any]) -> List[str]:
    """Best-effort extraction of contributing model name(s) from a fact."""
    prov = fact.get("model_provenance") or {}
    model = prov.get("model") or fact.get("_source") or fact.get("model")
    return [str(model)] if model else []


class ModelKnowledgeDistiller:
    """Distill cross-model-corroborated facts into training shards."""

    def __init__(
        self,
        facts: List[Dict[str, Any]],
        min_agreement: int = 2,
        min_reliability: str = "LIKELY_TRUE",
        similarity_threshold: float = 0.8,
        dedup_threshold: float = 0.9,
        quality_filter: Optional[FactQualityFilter] = None,
        top_k: Optional[int] = None,
        embedder: Any = None,
        embed_threshold: float = 0.82,
    ):
        norm_rel = str(min_reliability or "LIKELY_TRUE").strip().upper()
        if norm_rel not in _RELIABILITY_ORDER:
            raise ValueError(
                f"Unknown min_reliability {min_reliability!r}. Expected one of {_RELIABILITY_ORDER}."
            )
        self.facts = list(facts)
        self.min_agreement = min_agreement
        self.min_reliability = norm_rel
        self.similarity_threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.quality_filter = quality_filter
        self.top_k = None if top_k is None else max(0, int(top_k))
        self.embedder = embedder
        self.embed_threshold = embed_threshold
        self._selected: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_store(cls, store, **kwargs) -> "ModelKnowledgeDistiller":
        """Build a distiller from every model-derived fact in ``store``."""
        facts = [f for f in store.iter_facts() if f.get("model_provenance")]
        return cls(facts, **kwargs)

    def select_facts(self) -> List[Dict[str, Any]]:
        """Cluster, promote, filter, dedup, rank. Memoized."""
        if self._selected is not None:
            return self._selected

        clusters = cluster_facts(
            self.facts,
            self.similarity_threshold,
            embedder=self.embedder,
            embed_threshold=self.embed_threshold,
        )
        min_rel_idx = _RELIABILITY_ORDER.index(self.min_reliability)

        promoted: List[Dict[str, Any]] = []
        for cl in clusters:
            models = sorted({m for f in cl for m in _models_in_fact(f)})
            n = len(models)
            if n < self.min_agreement:
                continue
            reliability = reliability_for_agreement(n, models=models)
            if _RELIABILITY_ORDER.index(reliability) < min_rel_idx:
                continue
            rep = max(cl, key=lambda f: f.get("quality_score") or 0)
            statement = str(rep.get("fact_statement") or rep.get("statement") or "")
            quality = _RELIABILITY_QUALITY.get(reliability, 10) + n
            record: Dict[str, Any] = {
                "fact_statement": statement,
                "subject": rep.get("subject"),
                "predicate": rep.get("predicate"),
                "object": rep.get("object"),
                "category": rep.get("category", "General"),
                "reliability_rating": reliability,
                "quality_score": quality,
                "question": rep.get("question")
                or (
                    f"State a verified fact about {rep.get('subject')}."
                    if rep.get("subject")
                    else f"State a verified fact about {rep.get('category', 'General')}."
                ),
                "answer": rep.get("answer") or statement,
                "source_models": models,
                "cross_model_agreement": n,
            }
            subj_type = rep.get("subject_type") or next(
                (f.get("subject_type") for f in cl if f.get("subject_type") is not None),
                None,
            )
            if subj_type is not None:
                record["subject_type"] = subj_type
            promoted.append(record)

        if self.quality_filter is not None:
            promoted = [f for f in promoted if self.quality_filter.is_acceptable(f)]

        promoted.sort(
            key=lambda f: (
                -f["cross_model_agreement"],
                -(f.get("quality_score") or 0),
                f["fact_statement"],
            )
        )

        if self.dedup_threshold and self.dedup_threshold > 0:
            promoted = self._dedup(promoted)

        if self.top_k is not None:
            promoted = promoted[: self.top_k]

        self._selected = promoted
        return promoted

    def _dedup(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from .cross_model import jaccard

        kept: List[Dict[str, Any]] = []
        for f in facts:
            stmt = f["fact_statement"]
            if not any(jaccard(stmt, k["fact_statement"]) >= self.dedup_threshold for k in kept):
                kept.append(f)
        return kept

    def to_chat_jsonl(self) -> str:
        records = []
        for f in self.select_facts():
            records.append(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": f["question"]},
                            {"role": "assistant", "content": f["answer"]},
                        ],
                        "metadata": {
                            "source_models": f["source_models"],
                            "agreement": f["cross_model_agreement"],
                            "reliability": f["reliability_rating"],
                        },
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(records)

    def to_instruction_jsonl(self) -> str:
        records = []
        for f in self.select_facts():
            records.append(
                json.dumps(
                    {
                        "instruction": f["question"],
                        "input": "",
                        "output": f["answer"],
                        "metadata": {
                            "source_models": f["source_models"],
                            "agreement": f["cross_model_agreement"],
                            "reliability": f["reliability_rating"],
                        },
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(records)

    def to_text(self) -> str:
        lines = []
        for i, f in enumerate(self.select_facts(), start=1):
            models = ", ".join(f["source_models"])
            lines.append(
                f"{i}. {f['fact_statement']} [{models}; {f['reliability_rating']}]"
            )
        return "\n".join(lines)

    def manifest(self, shard_name: str = "model_shard") -> Dict[str, Any]:
        """Provenance manifest: which models contributed, tier totals."""
        selected = self.select_facts()
        all_models = sorted({m for f in selected for m in f["source_models"]})
        by_tier: Dict[str, int] = {}
        for f in selected:
            by_tier[f["reliability_rating"]] = by_tier.get(f["reliability_rating"], 0) + 1
        return {
            "shard": shard_name,
            "models": all_models,
            "facts": len(selected),
            "verified": by_tier.get("VERIFIED", 0),
            "likely_true": by_tier.get("LIKELY_TRUE", 0),
            "possibly_true": by_tier.get("POSSIBLY_TRUE", 0),
            "unverified": by_tier.get("UNVERIFIED", 0),
            "min_agreement": self.min_agreement,
            "input_facts": len(self.facts),
        }

    def stats(self) -> Dict[str, Any]:
        selected = self.select_facts()
        total = len(self.facts)
        return {
            "input_facts": total,
            "selected_facts": len(selected),
            "reduction_ratio": 0.0 if total == 0 else 1.0 - len(selected) / total,
        }

    def distill_to_file(
        self, path: str, fmt: str = "chat", encoding: str = "utf-8"
    ) -> int:
        serializers = {
            "chat": self.to_chat_jsonl,
            "instruction": self.to_instruction_jsonl,
            "text": self.to_text,
        }
        if fmt not in serializers:
            raise ValueError(
                f"Unknown format '{fmt}'. Expected one of: {', '.join(sorted(serializers))}."
            )
        content = serializers[fmt]()
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding=encoding) as fh:
            fh.write(content)
            if content and not content.endswith("\n"):
                fh.write("\n")
        os.replace(tmp_path, path)
        return len([ln for ln in content.splitlines() if ln.strip()])
