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
import json
import os
from typing import Any, Dict, List, Optional

# Ordered reliability ladder for promotion.
_RELIABILITY_ORDER = ["UNVERIFIED", "POSSIBLY_TRUE", "LIKELY_TRUE", "VERIFIED"]


def _norm(s: str) -> str:
    return " ".join(str(s or "").lower().split())


def _normalize_reliability(val: Any) -> str:
    """Normalize Enum, string, or missing reliability into a canonical rung."""
    if hasattr(val, "name"):
        val = val.name
    candidate = str(val or "").strip().upper()
    if candidate in _RELIABILITY_ORDER:
        return candidate
    return "UNVERIFIED"


def _is_within(base: str, target: str) -> bool:
    base_abs = os.path.abspath(base)
    target_abs = os.path.abspath(target)
    try:
        return os.path.commonpath([base_abs, target_abs]) == base_abs
    except ValueError:
        return False


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
        stmt = fact.get("fact_statement") or fact.get("statement")
        key = _norm(stmt)
        if not key:
            continue
        rec = by_statement[key]
        rec["statement"] = stmt
        src = fact.get("_source")
        if src is not None:
            rec["sources"].add(src)
        rel = _normalize_reliability(
            fact.get("reliability_rating") or fact.get("reliability")
        )
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
    pair_display: Dict[tuple, tuple] = {}
    obj_display: Dict[str, str] = {}
    for fact in store.iter_facts():
        subj = _norm(fact.get("subject"))
        pred = _norm(fact.get("predicate"))
        obj = fact.get("object")
        if not subj or not pred or obj is None:
            continue
        pair_key = (subj, pred)
        if pair_key not in pair_display:
            pair_display[pair_key] = (
                str(fact.get("subject") or "").strip(),
                str(fact.get("predicate") or "").strip(),
            )
        obj_key = _norm(obj)
        obj_display[obj_key] = str(obj)  # keep original-case display form
        src = fact.get("_source")
        if src is not None:
            by_pair[pair_key][obj_key].add(src)
        else:
            # Track distinct object occurrence even when _source is absent.
            _ = by_pair[pair_key][obj_key]

    conflicts = []
    for pair_key, objmap in by_pair.items():
        if len(objmap) > 1:
            subj_disp, pred_disp = pair_display[pair_key]
            conflicts.append({
                "subject": subj_disp,
                "predicate": pred_disp,
                "objects": sorted(obj_display[k] for k in objmap),
                "sources": {
                    obj_display[o]: sorted(str(x) for x in s if x is not None)
                    for o, s in objmap.items()
                },
            })
    return conflicts


def reextract_store(store, engine: str = "svo", resolve_coref: bool = True,
                    filter_name: str = "standard",
                    reliability: Optional[Any] = None) -> Dict[str, Any]:
    """Re-extract every stored source that carries raw text, with ``engine``.

    Writes a NEW drop version per re-extracted source (drops are immutable;
    re-extraction never edits in place). Drops lacking ``source_text`` or
    referencing missing shards are skipped; previously re-extracted derived
    drops are ignored so repeat runs do not multiply drops exponentially.
    """
    # Lazy imports to avoid a heavy import chain at module load.
    from .core import KnowledgeGraph, ReliabilityRating
    from .semantic import SemanticKnowledgeGraph
    from .distillation import KnowledgeDistiller
    from .quality import FactQualityFilter
    from .store import Drop, content_hash

    rel_enum = reliability if reliability is not None else ReliabilityRating.LIKELY_TRUE

    extractor = None
    if engine != "svo":
        from .extractor_base import get_extractor
        extractor = get_extractor(engine)

    qf = FactQualityFilter(max_object_len=80) if filter_name == "standard" else (
        FactQualityFilter(max_object_len=60, require_entity_subject=True)
        if filter_name == "strict" else None
    )

    reextracted = skipped = 0
    existing_drops = list(store.list_drops())
    existing_ids = {str(d.get("drop_id") or "") for d in existing_drops}

    for entry in existing_drops:
        drop_id = str(entry.get("drop_id") or "")
        if "-reextract-" in drop_id:
            continue

        rel_shard = entry.get("shard") or ""
        shard = os.path.join(store.root, rel_shard)
        if not rel_shard or not _is_within(store.root, shard) or not os.path.isfile(shard):
            skipped += 1
            continue

        with open(shard, "r", encoding="utf-8") as fh:
            first_line = fh.readline().strip()
        if not first_line:
            skipped += 1
            continue
        header = json.loads(first_line)
        meta = header.get("meta") or {}
        if isinstance(meta, dict) and meta.get("reextracted_from"):
            continue
        source_text = header.get("source_text")
        if not source_text:
            skipped += 1
            continue

        kg = KnowledgeGraph()
        skg = SemanticKnowledgeGraph(kg)
        skg.create_facts_from_text(
            source_text,
            source_id=entry["source"],
            reliability=rel_enum,
            resolve_coref=resolve_coref,
            extractor=extractor,
        )
        distiller = KnowledgeDistiller(
            kg, min_reliability=rel_enum, dedup_threshold=0.9, quality_filter=qf
        )
        facts = distiller.select_facts()

        base_id = f"{drop_id}-reextract-{engine}"
        new_drop_id = base_id
        version = 2
        while new_drop_id in existing_ids:
            new_drop_id = f"{base_id}-v{version}"
            version += 1
        existing_ids.add(new_drop_id)

        src_hash = entry.get("source_hash") or content_hash(source_text)
        new_drop = Drop(
            drop_id=new_drop_id,
            source=entry["source"],
            source_hash=src_hash,
            facts=facts,
            engine=engine,
            filter_name=filter_name,
            coref=resolve_coref,
            source_text=source_text,
            meta={"reextracted_from": drop_id},
        )
        store.write_drop(new_drop)
        reextracted += 1

    return {"reextracted": reextracted, "skipped": skipped}
