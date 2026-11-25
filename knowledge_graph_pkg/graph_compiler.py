"""
Graph Instruction Compiler.

Walks relationship chains in the graph database to generate multi-hop
reasoning instruction sets for fine-tuning.
"""

import json
import os
from typing import Any, Dict, List

from .graph_store_base import BaseGraphStore, RELIABILITY_RANK, _core_predicates_match, _s


def compile_subgraph_instructions(
    store: BaseGraphStore, max_chains: int = 100, auto_link: bool = True
) -> List[Dict[str, Any]]:
    """Query 1-hop RELATED chains from the graph and compile them into instruction pairs."""
    safe_limit = max(0, int(max_chains))
    if safe_limit == 0:
        return []

    if auto_link and hasattr(store, "auto_link_relations"):
        store.auto_link_relations()

    chains = store.query(
        "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
        "WHERE a.reliability <> 'UNVERIFIED' AND b.reliability <> 'UNVERIFIED' "
        "AND lower(a.subject) <> lower(b.object) "
        "RETURN a.statement AS stmt1, a.subject AS subj1, a.object AS obj1, "
        "b.statement AS stmt2, b.subject AS subj2, b.object AS obj2, "
        "a.reliability AS rel1, b.reliability AS rel2 "
        f"LIMIT {safe_limit}"
    )

    if not chains:
        return []

    instructions: List[Dict[str, Any]] = []
    for c in chains[:safe_limit]:
        s1 = _s(c.get("subj1")).strip()
        o1 = _s(c.get("obj1")).strip()
        s2 = _s(c.get("subj2")).strip()
        o2 = _s(c.get("obj2")).strip()
        stmt1 = _s(c.get("stmt1")).strip()
        stmt2 = _s(c.get("stmt2")).strip()
        rel1 = _s(c.get("rel1")).strip().upper() or "UNVERIFIED"
        rel2 = _s(c.get("rel2")).strip().upper() or "UNVERIFIED"

        if rel1 == "UNVERIFIED" or rel2 == "UNVERIFIED":
            continue
        if not s1 or not o1 or not o2 or s1.lower() == o2.lower():
            continue

        instructions.append(
            {
                "instruction": f"Trace the relationship connection starting from {s1} to {o2}.",
                "response": (
                    f"According to the knowledge graph, {s1} connects to {o2} through the intermediate concept {o1}. "
                    f"Specifically, we know that: \n"
                    f"- {stmt1} (Reliability: {rel1})\n"
                    f"- {stmt2} (Reliability: {rel2})"
                ),
            }
        )

        instructions.append(
            {
                "instruction": f"Explain the sequence of connections linking {s1} to {o2}.",
                "response": (
                    f"The logical path connecting {s1} to {o2} is:\n"
                    f"1. {stmt1}\n"
                    f"2. {stmt2}"
                ),
            }
        )

        instructions.append(
            {
                "instruction": (
                    f"If {stmt1} and we know that {s2} connects to {o2}, "
                    "what is the second step in the path?"
                ),
                "response": f"The second step in the path is: {stmt2}",
            }
        )

    return instructions


def compile_contradiction_instructions(store: BaseGraphStore) -> List[Dict[str, Any]]:
    """Query contradictory claim pairs and synthesize source evaluation instructions."""
    pairs = store.query(
        "MATCH (a:Fact), (b:Fact) "
        "WHERE a.subject <> '' AND a.object <> '' "
        "AND lower(a.subject) = lower(b.subject) AND lower(a.object) = lower(b.object) "
        "AND lower(a.predicate) <> lower(b.predicate) AND a.block_id < b.block_id "
        "RETURN a.statement AS stmt1, a.predicate AS pred1, a.reliability AS rel1, a.agreement AS agree1, "
        "b.statement AS stmt2, b.predicate AS pred2, b.reliability AS rel2, b.agreement AS agree2"
    )

    if not pairs:
        return []

    instructions: List[Dict[str, Any]] = []
    for p in pairs:
        if not _core_predicates_match(_s(p.get("pred1")), _s(p.get("pred2"))):
            continue

        stmt1 = _s(p.get("stmt1"))
        stmt2 = _s(p.get("stmt2"))
        rel1 = _s(p.get("rel1")).strip().upper() or "UNVERIFIED"
        rel2 = _s(p.get("rel2")).strip().upper() or "UNVERIFIED"
        agree1 = int(p.get("agree1") or 1)
        agree2 = int(p.get("agree2") or 1)

        r1 = RELIABILITY_RANK.get(rel1, 0)
        r2 = RELIABILITY_RANK.get(rel2, 0)

        if r1 > r2:
            verdict = (
                f"Claim 1 ('{stmt1}') is more reliable. It is rated {rel1} "
                f"(Agreement count: {agree1}), whereas Claim 2 ('{stmt2}') is rated {rel2} "
                f"(Agreement count: {agree2})."
            )
        elif r2 > r1:
            verdict = (
                f"Claim 2 ('{stmt2}') is more reliable. It is rated {rel2} "
                f"(Agreement count: {agree2}), whereas Claim 1 ('{stmt1}') is rated {rel1} "
                f"(Agreement count: {agree1})."
            )
        elif agree1 > agree2:
            verdict = (
                f"Claim 1 ('{stmt1}') is more reliable. It has higher model agreement count "
                f"({agree1}) compared to Claim 2 ({agree2})."
            )
        elif agree2 > agree1:
            verdict = (
                f"Claim 2 ('{stmt2}') is more reliable. It has higher model agreement count "
                f"({agree2}) compared to Claim 1 ({agree1})."
            )
        else:
            verdict = (
                f"Both claims have equal reliability ratings ({rel1}) and agreement counts "
                f"({agree1}). Neither claim can be conclusively preferred without additional context."
            )

        instructions.append(
            {
                "instruction": (
                    "Evaluate the following two conflicting claims and determine which is more reliable:\n"
                    f"1. {stmt1}\n2. {stmt2}"
                ),
                "response": verdict,
            }
        )

    return instructions


def save_compiled_instructions(instructions: List[Dict[str, Any]], output_path: str) -> None:
    """Save the compiled instructions as a standard instruction JSONL dataset atomically."""
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        for inst in instructions:
            fh.write(json.dumps(inst) + "\n")
    os.replace(tmp_path, output_path)
