"""
Entity Resolution & Synonym Merging for Distilled Fact Graphs.

This module clusters similar entity names (subjects/objects) across facts,
collapses synonym references to a canonical name, and merges duplicate
nodes in the graph database.
"""

import hashlib
from typing import Any, Dict, List, Optional, Set, Tuple
from .graph_store_base import BaseGraphStore

def _jaccard_similarity(s1: str, s2: str) -> float:
    """Compute word-level Jaccard similarity with substring fallback."""
    w1 = set(s1.lower().strip().split())
    w2 = set(s2.lower().strip().split())
    if not w1 or not w2:
        return 0.0
    
    # Exact or substring match (e.g. "ATP" in "Adenosine Triphosphate (ATP)")
    s1_clean = "".join(c for c in s1.lower() if c.isalnum())
    s2_clean = "".join(c for c in s2.lower() if c.isalnum())
    if s1_clean in s2_clean or s2_clean in s1_clean:
        return 1.0
        
    intersection = w1.intersection(w2)
    union = w1.union(w2)
    return len(intersection) / len(union)

def _generate_block_id(subj: str, pred: str, obj: str) -> str:
    """Deterministic block ID based on the SVO triple."""
    key = "\x00".join(str(v).strip() for v in (subj, pred, obj))
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def _get_blocking_keys(entity: str) -> Set[str]:
    """Generate cheap index blocking keys to avoid O(N^2) scans."""
    keys = set()
    words = entity.lower().strip().split()
    for w in words:
        w_clean = "".join(c for c in w if c.isalnum())
        if w_clean:
            # First character of word
            keys.add(w_clean[0])
            # First 2 characters of word if long enough
            if len(w_clean) >= 2:
                keys.add(w_clean[:2])
    return keys

def resolve_and_merge_entities(store: BaseGraphStore, threshold: float = 0.85,
                               limit_to_concepts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Identify similar entities, collapse synonyms, and merge duplicate nodes in the graph store.
    
    Supports multi-key prefix blocking and incremental candidate scanning.
    """
    # 1. Fetch all facts from the store
    facts = store.query(
        "MATCH (f:Fact) "
        "RETURN f.block_id AS block_id, f.statement AS statement, "
        "f.subject AS subject, f.predicate AS predicate, f.object AS object, "
        "f.domain AS domain, f.reliability AS reliability, f.agreement AS agreement, "
        "f.quality AS quality, f.source_models AS source_models"
    )
    if not facts:
        return {"resolved_clusters": 0, "merged_nodes": 0}

    # 2. Extract distinct entity names (subjects + objects)
    entities = set()
    for f in facts:
        if f.get("subject"):
            entities.add(str(f["subject"]).strip())
        if f.get("object"):
            entities.add(str(f["object"]).strip())
            
    sorted_entities = sorted(list(entities), key=len, reverse=True)
    
    # 3. Build blocking partitions
    blocks: Dict[str, List[str]] = {}
    for e in sorted_entities:
        for key in _get_blocking_keys(e):
            if key not in blocks:
                blocks[key] = []
            blocks[key].append(e)

    # Resolve target concepts if limiting comparison scope
    limit_set = set()
    if limit_to_concepts:
        limit_set = {c.lower().strip() for c in limit_to_concepts if c}

    # 4. Cluster similar entity names within blocks
    synonym_map: Dict[str, str] = {}  # synonym -> canonical
    clusters = []
    
    visited = set()
    for e in sorted_entities:
        if e in visited:
            continue
        cluster = [e]
        visited.add(e)
        
        is_e_target = (e.lower().strip() in limit_set) if limit_set else True
        
        # Gather candidate matches across overlapping blocks
        candidates = set()
        for key in _get_blocking_keys(e):
            candidates.update(blocks.get(key, []))
            
        for other in candidates:
            if other != e and other not in visited:
                is_other_target = (other.lower().strip() in limit_set) if limit_set else True
                
                # If target limitation is active, skip if neither is in the target set
                if limit_set and not (is_e_target or is_other_target):
                    continue
                    
                if _jaccard_similarity(e, other) >= threshold:
                    cluster.append(other)
                    visited.add(other)
                    synonym_map[other] = e
        if len(cluster) > 1:
            clusters.append(cluster)

    # If no clusters found, exit early
    if not synonym_map:
        return {"resolved_clusters": 0, "merged_nodes": 0}

    # 5. Update the facts structure in memory
    updated_facts: List[Dict[str, Any]] = []
    for f in facts:
        subj = str(f["subject"]).strip()
        obj = str(f["object"]).strip()
        
        new_subj = synonym_map.get(subj, subj)
        new_obj = synonym_map.get(obj, obj)
        
        # Rewrite the statement if either changed
        stmt = str(f["statement"])
        if new_subj != subj:
            stmt = stmt.replace(subj, new_subj)
        if new_obj != obj:
            stmt = stmt.replace(obj, new_obj)
            
        updated_facts.append({
            "old_bid": f["block_id"],
            "subject": new_subj,
            "predicate": f["predicate"],
            "object": new_obj,
            "statement": stmt,
            "domain": f["domain"],
            "reliability": f["reliability"],
            "agreement": f["agreement"],
            "quality": f["quality"],
            "source_models": f["source_models"]
        })

    # 6. Group by new block_id to detect duplicates
    merged_facts: Dict[str, List[Dict[str, Any]]] = {}
    for uf in updated_facts:
        new_bid = _generate_block_id(uf["subject"], uf["predicate"], uf["object"])
        uf["new_bid"] = new_bid
        if new_bid not in merged_facts:
            merged_facts[new_bid] = []
        merged_facts[new_bid].append(uf)

    # 7. Rebuild the database node list
    facts_to_keep: List[Dict[str, Any]] = []
    bids_to_delete = set()
    
    merged_count = 0
    for new_bid, group in merged_facts.items():
        group_sorted = sorted(group, key=lambda x: (int(x["agreement"] or 1), x["reliability"]), reverse=True)
        keep = group_sorted[0]
        facts_to_keep.append(keep)
        
        if len(group) > 1:
            merged_count += (len(group) - 1)
            
        for uf in group_sorted:
            bids_to_delete.add(uf["old_bid"])

    # 8. Execute graph transactions: delete old nodes & re-insert merged facts
    for bid in bids_to_delete:
        store.query("MATCH (f:Fact) WHERE f.block_id = $bid DETACH DELETE f", {"bid": bid})

    for f in facts_to_keep:
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
                "agree": int(f["agreement"] or 1),
                "quality": float(f["quality"] or 0.0),
                "models": f["source_models"]
            }
        )

    return {
        "resolved_clusters": len(clusters),
        "merged_nodes": merged_count,
        "clusters": clusters
    }
