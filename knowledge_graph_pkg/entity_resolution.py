"""
Entity Resolution & Synonym Merging for Distilled Fact Graphs.

This module clusters similar entity names (subjects/objects) across facts,
collapses synonym references to a canonical name, and merges duplicate
nodes in the graph database while preserving relationships and metadata.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph_store_base import BaseGraphStore, RELIABILITY_RANK, _block_id, _s

_INITIALS_STOPWORDS: Set[str] = {
    "of",
    "the",
    "and",
    "for",
    "in",
    "on",
    "at",
    "to",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _norm_alnum(text: str) -> str:
    return "".join(_tokenize(text))


def _strip_parens(text: str) -> str:
    return re.sub(r"\([^)]*\)", "", text).strip()


def _strip_single_char_prefix(text: str) -> str:
    return re.sub(r"^[a-z0-9]-", "", text.strip().lower())


def _is_acronym_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped or " " in stripped:
        return False
    alnum = re.sub(r"[^A-Za-z0-9]", "", stripped)
    if len(alnum) < 2:
        return False
    upper_count = sum(1 for c in alnum if c.isupper())
    has_digit = any(c.isdigit() for c in alnum)
    return upper_count >= 2 or (upper_count >= 1 and has_digit)


def _matches_initials(short_raw: str, long_outer: str) -> bool:
    stripped = short_raw.strip()
    if not stripped or " " in stripped or stripped != stripped.upper():
        return False
    short_alnum = _norm_alnum(stripped)
    if len(short_alnum) < 2:
        return False
    words = _tokenize(long_outer)
    if len(words) < 2:
        return False
    all_initials = "".join(w[0] for w in words)
    if short_alnum == all_initials:
        return True
    content_words = [w for w in words if w not in _INITIALS_STOPWORDS]
    if len(content_words) >= 2:
        content_initials = "".join(w[0] for w in content_words)
        if short_alnum == content_initials:
            return True
    return False


def _jaccard_similarity(s1: str, s2: str) -> float:
    """Compute token-level Jaccard similarity with boundary-aware alias matching."""
    w1 = set(_tokenize(s1))
    w2 = set(_tokenize(s2))
    if not w1 or not w2:
        return 0.0

    if _norm_alnum(s1) == _norm_alnum(s2):
        return 1.0

    if len(s1.strip()) <= len(s2.strip()):
        short_s, long_s = s1, s2
    else:
        short_s, long_s = s2, s1

    long_outer = _strip_parens(long_s)
    short_outer = _strip_parens(short_s)
    short_norm = _norm_alnum(short_s)

    if short_norm and short_norm == _norm_alnum(long_outer):
        return 1.0

    short_pref_norm = _norm_alnum(_strip_single_char_prefix(short_outer))
    long_pref_norm = _norm_alnum(_strip_single_char_prefix(long_outer))
    if len(short_pref_norm) >= 4 and short_pref_norm == long_pref_norm:
        return 1.0

    for paren in re.findall(r"\(([^)]+)\)", long_s):
        if short_norm and short_norm == _norm_alnum(paren):
            if _is_acronym_like(paren) or _is_acronym_like(short_s) or _matches_initials(paren, long_outer):
                return 1.0

    if _matches_initials(short_s, long_outer):
        return 1.0

    intersection = w1.intersection(w2)
    union = w1.union(w2)
    return len(intersection) / len(union)


def _generate_block_id(subj: str, pred: str, obj: str) -> str:
    """Deterministic block ID based on the SVO triple."""
    return _block_id({"subject": subj, "predicate": pred, "object": obj})


def _get_blocking_keys(entity: str) -> Set[str]:
    """Generate cheap index blocking keys to avoid O(N^2) scans."""
    keys: Set[str] = set()
    for w_clean in _tokenize(entity):
        if w_clean:
            keys.add(w_clean[0])
            if len(w_clean) >= 2:
                keys.add(w_clean[:2])
    return keys


def _replace_entity_in_statement(stmt: str, old: str, new: str) -> str:
    """Replace standalone occurrences of ``old`` with ``new`` respecting word boundaries."""
    if not stmt or not old or old == new:
        return stmt
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(old)}(?![A-Za-z0-9])")
    return pattern.sub(lambda _m: new, stmt)


def _merge_source_models(group: List[Dict[str, Any]]) -> str:
    """Merge comma-delimited or list source_models deterministically without duplicates."""
    seen: Set[str] = set()
    merged: List[str] = []
    for item in group:
        raw = item.get("source_models")
        if raw is None:
            continue
        if isinstance(raw, (list, tuple)):
            parts = [_s(m).strip() for m in raw]
        else:
            parts = [p.strip() for p in _s(raw).split(",")]
        for part in parts:
            if not part:
                continue
            low = part.lower()
            if low not in seen:
                seen.add(low)
                merged.append(part)
    return ", ".join(merged)


def resolve_and_merge_entities(
    store: BaseGraphStore,
    threshold: float = 0.85,
    limit_to_concepts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Identify similar entities, collapse synonyms, and merge duplicate nodes safely."""
    facts = store.query(
        "MATCH (f:Fact) "
        "RETURN f.block_id AS block_id, f.statement AS statement, "
        "f.subject AS subject, f.predicate AS predicate, f.object AS object, "
        "f.domain AS domain, f.reliability AS reliability, f.agreement AS agreement, "
        "f.quality AS quality, f.source_models AS source_models"
    )
    if not facts:
        return {"resolved_clusters": 0, "merged_nodes": 0, "clusters": []}

    entities: Set[str] = set()
    for f in facts:
        subj = _s(f.get("subject")).strip()
        obj = _s(f.get("object")).strip()
        if subj:
            entities.add(subj)
        if obj:
            entities.add(obj)

    sorted_entities = sorted(entities, key=lambda s: (-len(s), s.lower(), s))

    blocks: Dict[str, List[str]] = {}
    for e in sorted_entities:
        for key in _get_blocking_keys(e):
            blocks.setdefault(key, []).append(e)

    limit_set: Set[str] = set()
    if limit_to_concepts:
        limit_set = {_s(c).lower().strip() for c in limit_to_concepts if _s(c).strip()}

    synonym_map: Dict[str, str] = {}
    clusters: List[List[str]] = []
    visited: Set[str] = set()

    for e in sorted_entities:
        if e in visited:
            continue
        cluster = [e]
        visited.add(e)

        is_e_target = (e.lower().strip() in limit_set) if limit_set else True

        candidates: Set[str] = set()
        for key in _get_blocking_keys(e):
            candidates.update(blocks.get(key, []))

        for other in sorted(candidates, key=lambda s: (-len(s), s.lower(), s)):
            if other == e or other in visited:
                continue
            is_other_target = (other.lower().strip() in limit_set) if limit_set else True
            if limit_set and not (is_e_target or is_other_target):
                continue
            if _jaccard_similarity(e, other) >= threshold:
                cluster.append(other)
                visited.add(other)
                synonym_map[other] = e

        if len(cluster) > 1:
            clusters.append(cluster)

    if not synonym_map:
        return {"resolved_clusters": 0, "merged_nodes": 0, "clusters": []}

    updated_facts: List[Dict[str, Any]] = []
    for f in facts:
        orig_subj = _s(f.get("subject")).strip()
        orig_obj = _s(f.get("object")).strip()
        pred = _s(f.get("predicate")).strip()
        orig_stmt = _s(f.get("statement"))
        old_bid = _s(f.get("block_id")).strip()

        new_subj = synonym_map.get(orig_subj, orig_subj)
        new_obj = synonym_map.get(orig_obj, orig_obj)

        stmt = orig_stmt
        if new_subj != orig_subj:
            stmt = _replace_entity_in_statement(stmt, orig_subj, new_subj)
        if new_obj != orig_obj:
            stmt = _replace_entity_in_statement(stmt, orig_obj, new_obj)

        new_bid = _generate_block_id(new_subj, pred, new_obj)
        changed = (
            new_bid != old_bid
            or new_subj != orig_subj
            or new_obj != orig_obj
            or stmt != orig_stmt
        )

        updated_facts.append(
            {
                "old_bid": old_bid,
                "new_bid": new_bid,
                "changed": changed,
                "subject": new_subj,
                "predicate": pred,
                "object": new_obj,
                "statement": stmt,
                "domain": _s(f.get("domain")),
                "reliability": _s(f.get("reliability")).strip(),
                "agreement": f.get("agreement"),
                "quality": f.get("quality"),
                "source_models": f.get("source_models"),
            }
        )

    merged_facts: Dict[str, List[Dict[str, Any]]] = {}
    old_to_new: Dict[str, str] = {}
    for uf in updated_facts:
        merged_facts.setdefault(uf["new_bid"], []).append(uf)
        if uf["old_bid"]:
            old_to_new[uf["old_bid"]] = uf["new_bid"]

    facts_to_upsert: List[Dict[str, Any]] = []
    modified_groups: List[List[Dict[str, Any]]] = []
    merged_count = 0

    for _new_bid, group in merged_facts.items():
        if len(group) == 1 and not group[0]["changed"]:
            continue

        modified_groups.append(group)
        if len(group) > 1:
            merged_count += len(group) - 1

        group_sorted = sorted(
            group,
            key=lambda x: (
                RELIABILITY_RANK.get(_s(x.get("reliability")).strip().upper(), 0),
                int(x.get("agreement") or 1),
                round(float(x.get("quality") or 0)),
            ),
            reverse=True,
        )
        keep = dict(group_sorted[0])
        raw_rel = _s(keep.get("reliability")).strip()
        upper_rel = raw_rel.upper()
        keep["reliability"] = upper_rel if upper_rel in RELIABILITY_RANK else raw_rel
        keep["agreement"] = max(int(x.get("agreement") or 1) for x in group_sorted)
        keep["quality"] = round(float(keep.get("quality") or 0))
        keep["source_models"] = _merge_source_models(group_sorted)
        facts_to_upsert.append(keep)

    active_new_bids = set(merged_facts.keys())
    obsolete_bids: Set[str] = {
        uf["old_bid"]
        for group in modified_groups
        for uf in group
        if uf["old_bid"] and uf["old_bid"] != uf["new_bid"] and uf["old_bid"] not in active_new_bids
    }

    edges_to_recreate: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if obsolete_bids:
        try:
            existing_edges = store.query(
                "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
                "RETURN a.block_id AS src, b.block_id AS dst, "
                "r.predicate AS predicate, r.weight AS weight"
            )
        except Exception:
            existing_edges = []

        for e in existing_edges or []:
            src = _s(e.get("src")).strip()
            dst = _s(e.get("dst")).strip()
            if not src or not dst:
                continue
            if src not in obsolete_bids and dst not in obsolete_bids:
                continue
            new_src = old_to_new.get(src, src)
            new_dst = old_to_new.get(dst, dst)
            if not new_src or not new_dst or new_src == new_dst:
                continue
            if (new_src, new_dst) not in edges_to_recreate:
                edges_to_recreate[(new_src, new_dst)] = {
                    "src": new_src,
                    "dst": new_dst,
                    "pred": _s(e.get("predicate")),
                    "weight": float(e.get("weight") if e.get("weight") is not None else 1.0),
                }

    # Write merged winners first so edge remapping and obsolete deletion are non-destructive.
    for f in facts_to_upsert:
        store.query(
            "MERGE (f:Fact {block_id: $bid}) "
            "SET f.statement = $stmt, f.subject = $subj, f.predicate = $pred, "
            "f.object = $obj, f.domain = $domain, f.reliability = $rel, "
            "f.agreement = $agree, f.quality = $quality, f.source_models = $models",
            {
                "bid": f["new_bid"],
                "stmt": f["statement"],
                "subj": f["subject"],
                "pred": f["predicate"],
                "obj": f["object"],
                "domain": f["domain"],
                "rel": f["reliability"],
                "agree": int(f["agreement"]),
                "quality": int(f["quality"]),
                "models": f["source_models"],
            },
        )

    for edge in edges_to_recreate.values():
        store.query(
            "MATCH (a:Fact {block_id: $src}), (b:Fact {block_id: $dst}) "
            "MERGE (a)-[r:RELATED]->(b) "
            "ON CREATE SET r.predicate = $pred, r.weight = $weight",
            edge,
        )

    for bid in sorted(obsolete_bids):
        store.query("MATCH (f:Fact) WHERE f.block_id = $bid DETACH DELETE f", {"bid": bid})

    return {
        "resolved_clusters": len(clusters),
        "merged_nodes": merged_count,
        "clusters": clusters,
    }
