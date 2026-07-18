"""
Evaluation harness for the extraction layer.

Turns "did extraction get better?" from a vibe into a number. Given a
labeled gold set (sentences paired with the triples a correct extractor
should produce), :func:`evaluate` runs an extractor and reports
precision, recall, and F1.

Matching is deliberately **lenient** -- heuristic extraction rarely
reproduces an object span character-for-character, so a predicted triple
counts as a match when:

* subjects match case-insensitively by substring (either contains the
  other),
* objects match case-insensitively by substring,
* predicates belong to the same family (e.g. ``born`` ~ ``born_in``).

This rewards getting the fact right without penalizing harmless span or
predicate-spelling differences.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class GoldItem:
    """A labeled example: input sentence + expected triples."""
    text: str
    facts: List[Dict[str, str]] = field(default_factory=list)


def load_gold_set(path: str) -> List[GoldItem]:
    """Load a gold set JSON file into a list of :class:`GoldItem`."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    items = data["items"] if isinstance(data, dict) else data
    return [GoldItem(text=it["text"], facts=it.get("facts", [])) for it in items]


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def _predicate_family(p: str) -> str:
    """Collapse predicate spelling variants to a comparable root.

    e.g. 'born_in' -> 'born', 'president_of' -> 'president', 'wrote' -> 'wrote'.
    """
    p = str(p).lower()
    p = p.split("_")[0]
    return p


def _subobj_match(a: str, b: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return na == nb
    return na in nb or nb in na


def triple_matches(pred: Dict[str, str], gold: Dict[str, str]) -> bool:
    """Return True if a predicted triple satisfies a gold triple (lenient)."""
    if not _subobj_match(pred.get("subject", ""), gold.get("subject", "")):
        return False
    if not _subobj_match(pred.get("object", ""), gold.get("object", "")):
        return False
    # Predicate: same family, or one contains the other.
    pf, gf = _predicate_family(pred.get("predicate", "")), _predicate_family(gold.get("predicate", ""))
    return pf == gf or pf in gf or gf in pf


def evaluate(extractor: Any, gold_items: List[GoldItem]) -> Dict[str, Any]:
    """Run ``extractor`` over the gold set and compute precision/recall/F1.

    Args:
        extractor: An object with ``.extract(text) -> List[dict]`` (e.g.
            :class:`knowledge_graph_pkg.extraction.SVOExtractor`).
        gold_items: The labeled examples.

    Returns:
        A dict with precision, recall, f1, and the raw TP/FP/FN counts.
    """
    tp = fp = fn = 0

    for item in gold_items:
        predicted = extractor.extract(item.text)
        gold = list(item.facts)

        matched_gold = set()
        matched_pred = set()
        for pi, pred in enumerate(predicted):
            for gi, g in enumerate(gold):
                if gi in matched_gold:
                    continue
                if triple_matches(pred, g):
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

        tp += len(matched_pred)
        fp += len(predicted) - len(matched_pred)
        fn += len(gold) - len(matched_gold)

    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "num_items": len(gold_items),
    }


def format_report(report: Dict[str, Any]) -> str:
    """Render an evaluation report as a human-readable block."""
    return (
        "Extraction evaluation\n"
        f"  items:            {report['num_items']}\n"
        f"  precision:        {report['precision']:.3f}\n"
        f"  recall:           {report['recall']:.3f}\n"
        f"  f1:               {report['f1']:.3f}\n"
        f"  true positives:   {report['true_positives']}\n"
        f"  false positives:  {report['false_positives']}\n"
        f"  false negatives:  {report['false_negatives']}"
    )
