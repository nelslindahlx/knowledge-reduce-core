"""
Model shard evaluation + quality gates (ModelReduce Session 5).

Turns "is this shard good enough to train on?" into measured numbers. Given
a distilled shard (SVO facts carrying reliability + cross-model agreement)
and a domain gold set (human-verified facts plus known misconceptions),
:class:`ModelShardEvaluator` reports:

* **precision / recall / F1** overall and per reliability tier,
* **hallucination rate** -- fraction of shard facts that match a gold
  *negative* (a known-false claim),
* **coverage** -- fraction of gold facts the shard recovered,
* **agreement calibration** -- precision bucketed by cross-model agreement
  count, which answers the open question from Sessions 2-4: *does 2-model
  agreement actually mean high precision?*

Matching uses semantic embeddings when an embedder is supplied (catches
paraphrases like "generate adenosine triphosphate" ~ "produce ATP"), and a
lenient case-insensitive substring fallback otherwise -- so the evaluator
runs in CI with no Ollama.

:func:`check_gates` compares a report against threshold gates; the CLI uses
it to exit non-zero when a shard fails, making this a CI quality gate.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


# CI-ready default thresholds. Conservative but not punishing for v1.
DEFAULT_GATES = {
    "min_precision_overall": 0.80,
    "min_precision_verified": 0.90,
    "max_hallucination_rate": 0.10,
    "min_coverage": 0.30,
}


def _norm(s: Any) -> str:
    return " ".join(str(s or "").lower().split())


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
    """Case-insensitive substring match on subject AND object."""
    sa, sb = _norm(a.get("subject")), _norm(b.get("subject"))
    oa, ob = _norm(a.get("object")), _norm(b.get("object"))
    if not (sa and sb and oa and ob):
        return False
    subj_ok = sa in sb or sb in sa
    obj_ok = oa in ob or ob in oa
    return subj_ok and obj_ok


class ModelShardEvaluator:
    """Score shard facts against a domain gold set."""

    def __init__(self, embedder: Any = None, similarity_threshold: float = 0.82):
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

    # ------------------------------------------------------------------ #
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
                pass  # fall back to lenient string matching
        return _lenient_match(fact, gold)

    def _matches_any(self, fact: Dict[str, Any], gold_list: List[Dict[str, Any]]) -> bool:
        return any(self._matches(fact, g) for g in gold_list)

    # ------------------------------------------------------------------ #
    def evaluate(self, shard_facts: List[Dict[str, Any]], gold_path: str) -> Dict[str, Any]:
        """Evaluate ``shard_facts`` against the gold set at ``gold_path``."""
        positives, negatives = load_gold_facts(gold_path)

        def pr_block(facts: List[Dict[str, Any]]) -> Dict[str, Any]:
            """precision/recall/F1 of a fact subset vs. the positive gold."""
            tp = sum(1 for f in facts if self._matches_any(f, positives))
            fp = len(facts) - tp
            matched_gold = sum(1 for g in positives
                               if any(self._matches(f, g) for f in facts))
            fn = len(positives) - matched_gold
            precision = tp / (tp + fp) if (tp + fp) else (1.0 if not facts else 0.0)
            recall = tp / (tp + fn) if (tp + fn) else 1.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            return {"precision": round(precision, 4), "recall": round(recall, 4),
                    "f1": round(f1, 4), "true_positives": tp, "false_positives": fp,
                    "false_negatives": fn, "count": len(facts)}

        overall = pr_block(shard_facts)

        # Per reliability tier.
        by_tier: Dict[str, Any] = {}
        tiers = {f.get("reliability_rating", "UNVERIFIED") for f in shard_facts}
        for tier in tiers:
            tier_facts = [f for f in shard_facts if f.get("reliability_rating") == tier]
            by_tier[tier] = pr_block(tier_facts)

        # Agreement calibration: precision bucketed by cross-model agreement.
        calibration: Dict[str, Any] = {}
        agreements = {int(f.get("cross_model_agreement", 1)) for f in shard_facts}
        for n in sorted(agreements):
            bucket = [f for f in shard_facts if int(f.get("cross_model_agreement", 1)) == n]
            tp = sum(1 for f in bucket if self._matches_any(f, positives))
            calibration[str(n)] = {
                "precision": round(tp / len(bucket), 4) if bucket else 0.0,
                "n": len(bucket),
            }

        # Hallucination: shard facts matching a known-false gold negative.
        halluc = sum(1 for f in shard_facts if self._matches_any(f, negatives))
        hallucination_rate = round(halluc / len(shard_facts), 4) if shard_facts else 0.0

        # Coverage: fraction of positive gold facts recovered.
        recovered = sum(1 for g in positives
                        if any(self._matches(f, g) for f in shard_facts))
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


def check_gates(report: Dict[str, Any],
                gates: Optional[Dict[str, float]] = None) -> Tuple[bool, List[str]]:
    """Compare a report to threshold gates. Returns (passed, failure_msgs)."""
    gates = gates or DEFAULT_GATES
    failures: List[str] = []

    p = report["overall"]["precision"]
    if p < gates["min_precision_overall"]:
        failures.append(f"overall precision {p:.3f} < {gates['min_precision_overall']}")

    verified = report.get("by_tier", {}).get("VERIFIED")
    if verified is not None:
        vp = verified["precision"]
        if vp < gates["min_precision_verified"]:
            failures.append(f"VERIFIED precision {vp:.3f} < {gates['min_precision_verified']}")

    hr = report["hallucination_rate"]
    if hr > gates["max_hallucination_rate"]:
        failures.append(f"hallucination rate {hr:.3f} > {gates['max_hallucination_rate']}")

    cov = report["coverage"]
    if cov < gates["min_coverage"]:
        failures.append(f"coverage {cov:.3f} < {gates['min_coverage']}")

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
    for tier, m in sorted(report["by_tier"].items()):
        lines.append(f"    {tier:<14} precision={m['precision']:.3f} "
                     f"recall={m['recall']:.3f} (n={m['count']})")
    lines.append("  agreement calibration:")
    for n, m in sorted(report["agreement_calibration"].items()):
        lines.append(f"    {n}-model        precision={m['precision']:.3f} (n={m['n']})")
    return "\n".join(lines)
