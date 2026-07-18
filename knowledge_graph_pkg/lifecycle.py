"""
Corpus lifecycle operations (the "improve the past" layer).

Because every drop stores its raw ``source_text``, the accumulated corpus
can be improved retroactively without re-fetching anything:

* :func:`reextract_store` -- re-run extraction over stored sources with a
  (possibly better) engine, writing new drop *versions*.
* :func:`promote_reliability` -- facts corroborated across multiple
  independent sources are surfaced for a reliability bump.
* :func:`find_contradictions` -- same subject+predicate with different
  objects across the corpus are flagged rather than silently kept.

These are pure-Python and operate over a :class:`KnowledgeStore`.
"""

from collections import defaultdict
from typing import Any, Dict, List

# Ordered reliability ladder for promotion.
_RELIABILITY_ORDER = ["UNVERIFIED", "POSSIBLY_TRUE", "LIKELY_TRUE", "VERIFIED"]


def _norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def promote_reliability(store, min_sources: int = 2) -> List[Dict[str, Any]]:
    """Surface facts corroborated across >= ``min_sources`` distinct sources.

    Returns a list of promotion suggestions: each names the statement, how
    many sources asserted it, and the proposed bumped reliability (one rung
    up the ladder, capped at VERIFIED). Does not mutate the store -- it
    reports what should be promoted, so the decision stays auditable.
    """
    by_statement: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"sources": set(), "max_rel": "UNVERIFIED", "statement": ""}
    )
    for fact in store.iter_facts():
        key = _norm(fact.get("fact_statement"))
        if not key:
            continue
        rec = by_statement[key]
        rec["statement"] = fact.get("fact_statement")
        rec["sources"].add(fact.get("_source"))
        rel = fact.get("reliability_rating", "UNVERIFIED")
        if _RELIABILITY_ORDER.index(rel) > _RELIABILITY_ORDER.index(rec["max_rel"]):
            rec["max_rel"] = rel

    promotions = []
    for rec in by_statement.values():
        n = len(rec["sources"])
        if n >= min_sources:
            cur = _RELIABILITY_ORDER.index(rec["max_rel"])
            new = min(cur + 1, len(_RELIABILITY_ORDER) - 1)
            if new > cur:
                promotions.append({
                    "statement": rec["statement"],
                    "sources": n,
                    "old_reliability": rec["max_rel"],
                    "new_reliability": _RELIABILITY_ORDER[new],
                })
    return promotions


def find_contradictions(store) -> List[Dict[str, Any]]:
    """Flag (subject, predicate) pairs asserted with conflicting objects.

    Returns one record per conflicting pair, listing the distinct objects
    and which sources asserted each.
    """
    by_pair: Dict[tuple, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    obj_display: Dict[str, str] = {}
    for fact in store.iter_facts():
        subj = _norm(fact.get("subject"))
        pred = _norm(fact.get("predicate"))
        obj = fact.get("object")
        if not subj or not pred or obj is None:
            continue
        obj_key = _norm(obj)
        obj_display[obj_key] = obj  # keep original-case display form
        by_pair[(fact.get("subject"), fact.get("predicate"))][obj_key].add(
            fact.get("_source")
        )

    conflicts = []
    for (subj, pred), objmap in by_pair.items():
        if len(objmap) > 1:
            conflicts.append({
                "subject": subj,
                "predicate": pred,
                "objects": sorted(obj_display[k] for k in objmap),
                "sources": {obj_display[o]: sorted(s) for o, s in objmap.items()},
            })
    return conflicts


def reextract_store(store, engine: str = "svo", resolve_coref: bool = True,
                    filter_name: str = "standard") -> Dict[str, Any]:
    """Re-extract every stored source that carries raw text, with ``engine``.

    Writes a NEW drop version per re-extracted source (drops are immutable;
    re-extraction never edits in place). Drops lacking ``source_text`` are
    skipped. Returns counts of reextracted / skipped sources.
    """
    # Lazy imports to avoid a heavy import chain at module load.
    from .core import KnowledgeGraph, ReliabilityRating
    from .semantic import SemanticKnowledgeGraph
    from .distillation import KnowledgeDistiller
    from .quality import FactQualityFilter
    from .store import Drop, content_hash

    extractor = None
    if engine != "svo":
        from .extractor_base import get_extractor
        extractor = get_extractor(engine)

    qf = FactQualityFilter(max_object_len=80) if filter_name == "standard" else (
        FactQualityFilter(max_object_len=60, require_entity_subject=True)
        if filter_name == "strict" else None
    )

    reextracted = skipped = 0
    # Snapshot current drops (we'll be appending new ones).
    for entry in list(store.list_drops()):
        # Read the shard header to recover source_text.
        import os, json
        shard = os.path.join(store.root, entry["shard"])
        source_text = None
        with open(shard, "r", encoding="utf-8") as fh:
            header = json.loads(fh.readline())
            source_text = header.get("source_text")
        if not source_text:
            skipped += 1
            continue

        kg = KnowledgeGraph()
        skg = SemanticKnowledgeGraph(kg)
        skg.create_facts_from_text(source_text, source_id=entry["source"],
                                   reliability=ReliabilityRating.LIKELY_TRUE,
                                   resolve_coref=resolve_coref, extractor=extractor)
        distiller = KnowledgeDistiller(kg, min_reliability=ReliabilityRating.LIKELY_TRUE,
                                       dedup_threshold=0.9, quality_filter=qf)
        facts = distiller.select_facts()

        src_hash = entry.get("source_hash") or content_hash(source_text)
        new_drop = Drop(
            drop_id=f"{entry['drop_id']}-reextract-{engine}",
            source=entry["source"],
            source_hash=src_hash,
            facts=facts,
            engine=engine,
            filter_name=filter_name,
            coref=resolve_coref,
            source_text=source_text,
            meta={"reextracted_from": entry["drop_id"]},
        )
        store.write_drop(new_drop)
        reextracted += 1

    return {"reextracted": reextracted, "skipped": skipped}
