"""
Model shard evaluation + quality gates (ModelReduce Session 5).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from .graph_store_base import _NEGATION_TOKENS

DEFAULT_GATES = {
    "min_precision_overall": 0.80,
    "min_precision_verified": 0.90,
    "max_hallucination_rate": 0.10,
    "min_coverage": 0.30,
}

_TIER_ORDER = {"VERIFIED": 0, "LIKELY_TRUE": 1, "POSSIBLY_TRUE": 2, "UNVERIFIED": 3}


def _norm(s: Any) -> str:
    return " ".join(str(s or "").lower().split())


def _tier_of(fact: Dict[str, Any]) -> str:
    raw = str(fact.get("reliability_rating") or "UNVERIFIED").strip().upper()
    return raw or "UNVERIFIED"


def _has_negation(predicate: Any) -> bool:
    tokens = set(re.findall(r"[a-z0-9']+", _norm(predicate)))
    return bool(tokens & _NEGATION_TOKENS)


def _term_matches(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) <= 3:
        ta = set(re.findall(r"[a-z0-9]+", a))
        tb = set(re.findall(r"[a-z0-9]+", b))
        return a in tb or b in ta
    return a in b or b in a


def load_gold_facts(path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load a gold set JSON; return (positive_facts, negative_facts)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return list(data.get("facts", [])), list(data.get("negative", []))


def _triple_text(f: Dict[str, Any]) -> str:
    """Concatenated SVO(+context) text used for matching/embedding."""
    parts = [f.get("subject"), f.get("predicate"), f.get("object"), f.get("context")]
    return " ".join(_norm(p) for p in parts if p)


def _lenient_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Boundary-aware case-insensitive match on subject & object with negation check."""
    sa, sb = _norm(a.get("subject")), _norm(b.get("subject"))
    oa, ob = _norm(a.get("object")), _norm(b.get("object"))
    if not (sa and sb and oa and ob):
        return False
    pa, pb = _norm(a.get("predicate")), _norm(b.get("predicate"))
    if pa and pb and _has_negation(pa) != _has_negation(pb):
        return False
    return _term_matches(sa, sb) and _term_matches(oa, ob)


class ModelShardEvaluator:
    """Score shard facts against a domain gold set."""

    def __init__(self, embedder: Any = None, similarity_threshold: float = 0.82):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    def _matches(self, fact: Dict[str, Any], gold: Dict[str, Any]) -> bool:
        if self.embedder is not None:
            try:
                from .embeddings import cosine_similarity

                sim = cosine_similarity(
                    self.embedder.embed_one(_triple_text(fact)),
                    self.embedder.embed_one(_triple_text(gold)),
                )
                return sim >= self.similarity_threshold
            except Exception:
                pass
        return _lenient_match(fact, gold)

    def _matches_any(self, fact: Dict[str, Any], gold_list: List[Dict[str, Any]]) -> bool:
        return any(self._matches(fact, g) for g in gold_list)

    def evaluate(
        self, shard_facts: List[Dict[str, Any]], gold_path: str
    ) -> Dict[str, Any]:
        """Evaluate ``shard_facts`` against the gold set at ``gold_path``."""
        positives, negatives = load_gold_facts(gold_path)

        def pr_block(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
            tp = sum(1 for f in facts if self._matches_any(f, positives))
            fp = len(facts) - tp
            matched_gold = sum(
                1 for g in positives if any(self._matches(f, g) for f in facts)
            )
            fn = len(positives) - matched_gold
            precision = tp / len(facts) if facts else 0.0
            recall = matched_gold / len(positives) if positives else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall)
                else 0.0
            )
            return {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "count": len(facts),
            }

        overall = pr_block(shard_facts)

        by_tier: Dict[str, Any] = {}
        tiers = {_tier_of(f) for f in shard_facts}
        for tier in tiers:
            tier_facts = [f for f in shard_facts if _tier_of(f) == tier]
            by_tier[tier] = pr_block(tier_facts)

        calibration: Dict[str, Any] = {}
        agreements = {int(f.get("cross_model_agreement", 1)) for f in shard_facts}
        for n in sorted(agreements):
            bucket = [
                f for f in shard_facts if int(f.get("cross_model_agreement", 1)) == n
            ]
            tp = sum(1 for f in bucket if self._matches_any(f, positives))
            calibration[str(n)] = {
                "precision": round(tp / len(bucket), 4) if bucket else 0.0,
                "n": len(bucket),
            }

        halluc = sum(1 for f in shard_facts if self._matches_any(f, negatives))
        hallucination_rate = (
            round(halluc / len(shard_facts), 4) if shard_facts else 0.0
        )

        recovered = sum(
            1 for g in positives if any(self._matches(f, g) for f in shard_facts)
        )
        coverage = round(recovered / len(positives), 4) if positives else 0.0

        return {
            "overall": overall,
            "by_tier": by_tier,
            "agreement_calibration": calibration,
            "hallucination_rate": hallucination_rate,
            "coverage": coverage,
            "n_shard_facts": len(shard_facts),
            "n_gold_positive": len(positives),
            "n_gold_negative": len(negatives),
        }


def check_gates(
    report: Dict[str, Any], gates: Optional[Dict[str, float]] = None
) -> Tuple[bool, List[str]]:
    """Compare a report to threshold gates. Returns (passed, failure_msgs)."""
    merged_gates = {**DEFAULT_GATES, **(gates or {})}
    failures: List[str] = []

    p = report["overall"]["precision"]
    if p < merged_gates["min_precision_overall"]:
        failures.append(
            f"overall precision {p:.3f} < {merged_gates['min_precision_overall']}"
        )

    verified = report.get("by_tier", {}).get("VERIFIED")
    if verified is not None:
        vp = verified["precision"]
        if vp < merged_gates["min_precision_verified"]:
            failures.append(
                f"VERIFIED precision {vp:.3f} < {merged_gates['min_precision_verified']}"
            )

    hr = report["hallucination_rate"]
    if hr > merged_gates["max_hallucination_rate"]:
        failures.append(
            f"hallucination rate {hr:.3f} > {merged_gates['max_hallucination_rate']}"
        )

    cov = report["coverage"]
    if cov < merged_gates["min_coverage"]:
        failures.append(f"coverage {cov:.3f} < {merged_gates['min_coverage']}")

    return (len(failures) == 0), failures


def format_report(report: Dict[str, Any]) -> str:
    """Render a model-eval report as a human-readable block."""
    o = report["overall"]
    lines = [
        "Model shard evaluation",
        f"  shard facts:       {report['n_shard_facts']}",
        f"  gold (pos/neg):    {report['n_gold_positive']}/{report['n_gold_negative']}",
        f"  precision:         {o['precision']:.3f}",
        f"  recall:            {o['recall']:.3f}",
        f"  f1:                {o['f1']:.3f}",
        f"  coverage:          {report['coverage']:.3f}",
        f"  hallucination:     {report['hallucination_rate']:.3f}",
        "  by tier:",
    ]
    for tier, m in sorted(
        report["by_tier"].items(),
        key=lambda kv: (_TIER_ORDER.get(kv[0], 99), kv[0]),
    ):
        lines.append(
            f"    {tier:<14} precision={m['precision']:.3f} "
            f"recall={m['recall']:.3f} (n={m['count']})"
        )
    lines.append("  agreement calibration:")
    for n, m in sorted(
        report["agreement_calibration"].items(),
        key=lambda kv: (0, int(kv[0])) if str(kv[0]).isdigit() else (1, str(kv[0])),
    ):
        lines.append(f"    {n}-model        precision={m['precision']:.3f} (n={m['n']})")
    return "\n".join(lines)
