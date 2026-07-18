"""
Graph Instruction Compiler.

Walks relationship chains in the graph database to generate multi-hop
reasoning instruction sets for fine-tuning.
"""

import json
from typing import Any, Dict, List
from .graph_store_base import BaseGraphStore

def compile_subgraph_instructions(store: BaseGraphStore, max_chains: int = 100) -> List[Dict[str, Any]]:
    """Query 2-hop or 1-hop chains from the graph and compile them into instruction pairs."""
    # First, make sure auto-linking is completed so RELATED edges exist
    if hasattr(store, "auto_link_relations"):
        store.auto_link_relations()
        
    # Query RELATED edges (which represent 1-hop transitions)
    # Filter out UNVERIFIED nodes so we do not train on contradiction chains
    chains = store.query(
        "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
        "WHERE a.reliability <> 'UNVERIFIED' AND b.reliability <> 'UNVERIFIED' "
        "RETURN a.statement AS stmt1, a.subject AS subj1, a.object AS obj1, "
        "b.statement AS stmt2, b.subject AS subj2, b.object AS obj2, "
        "a.reliability AS rel1, b.reliability AS rel2"
    )
    
    if not chains:
        return []

    instructions: List[Dict[str, Any]] = []
    
    # Cap the number of processed chains
    processed_chains = chains[:max_chains]
    
    for c in processed_chains:
        s1, o1 = str(c["subj1"]), str(c["obj1"])
        s2, o2 = str(c["subj2"]), str(c["obj2"])
        stmt1, stmt2 = str(c["stmt1"]), str(c["stmt2"])
        rel1, rel2 = str(c["rel1"] or "VERIFIED"), str(c["rel2"] or "VERIFIED")
        
        # 1. Template: Trace Connection
        instructions.append({
            "instruction": f"Trace the relationship connection starting from {s1} to {o2}.",
            "response": (
                f"According to the knowledge graph, {s1} connects to {o2} through the intermediate concept {o1}. "
                f"Specifically, we know that: \n"
                f"- {stmt1} (Reliability: {rel1})\n"
                f"- {stmt2} (Reliability: {rel2})"
            )
        })
        
        # 2. Template: Chain Sequence
        instructions.append({
            "instruction": f"Explain the sequence of connections linking {s1} to {o2}.",
            "response": (
                f"The logical path connecting {s1} to {o2} is:\n"
                f"1. {stmt1}\n"
                f"2. {stmt2}"
            )
        })
        
        # 3. Template: Inference
        instructions.append({
            "instruction": f"If {stmt1} and we know that {s2} connects to {o2}, what is the second step in the path?",
            "response": f"The second step in the path is: {stmt2}"
        })
        
    return instructions

def compile_contradiction_instructions(store: BaseGraphStore) -> List[Dict[str, Any]]:
    """Query contradiction pairs and synthesize source evaluation instructions."""
    # Find contradictions: facts with same subject & object, but different predicates
    pairs = store.query(
        "MATCH (a:Fact), (b:Fact) "
        "WHERE lower(a.subject) = lower(b.subject) AND lower(a.object) = lower(b.object) "
        "AND lower(a.predicate) <> lower(b.predicate) AND a.block_id < b.block_id "
        "RETURN a.statement AS stmt1, a.reliability AS rel1, a.agreement AS agree1, "
        "b.statement AS stmt2, b.reliability AS rel2, b.agreement AS agree2"
    )
    
    if not pairs:
        return []
        
    instructions: List[Dict[str, Any]] = []
    for p in pairs:
        stmt1, stmt2 = str(p["stmt1"]), str(p["stmt2"])
        rel1, rel2 = str(p["rel1"] or "UNVERIFIED"), str(p["rel2"] or "UNVERIFIED")
        agree1, agree2 = int(p["agree1"] or 1), int(p["agree2"] or 1)
        
        # Decide which claim is more reliable
        # Order: VERIFIED > LIKELY_TRUE > POSSIBLY_TRUE > UNVERIFIED
        rank = {"VERIFIED": 4, "LIKELY_TRUE": 3, "POSSIBLY_TRUE": 2, "UNVERIFIED": 1}
        r1 = rank.get(rel1, 1)
        r2 = rank.get(rel2, 1)
        
        if r1 > r2:
            verdict = f"Claim 1 ('{stmt1}') is more reliable. It is rated {rel1} (Agreement count: {agree1}), whereas Claim 2 ('{stmt2}') is rated {rel2} (Agreement count: {agree2})."
        elif r2 > r1:
            verdict = f"Claim 2 ('{stmt2}') is more reliable. It is rated {rel2} (Agreement count: {agree2}), whereas Claim 1 ('{stmt1}') is rated {rel1} (Agreement count: {agree1})."
        else:
            # If ratings are equal, look at agreement count
            if agree1 > agree2:
                verdict = f"Claim 1 ('{stmt1}') is more reliable. It has higher model agreement count ({agree1}) compared to Claim 2 ({agree2})."
            elif agree2 > agree1:
                verdict = f"Claim 2 ('{stmt2}') is more reliable. It has higher model agreement count ({agree2}) compared to Claim 1 ({agree1})."
            else:
                verdict = f"Both claims have equal reliability ratings ({rel1}) and agreement counts ({agree1}). Neither claim can be conclusively preferred without additional context."
                
        instructions.append({
            "instruction": f"Evaluate the following two conflicting claims and determine which is more reliable:\n1. {stmt1}\n2. {stmt2}",
            "response": verdict
        })
        
    return instructions

def save_compiled_instructions(instructions: List[Dict[str, Any]], output_path: str) -> None:
    """Save the compiled instructions as a standard instruction JSONL dataset."""
    with open(output_path, "w", encoding="utf-8") as fh:
        for inst in instructions:
            fh.write(json.dumps(inst) + "\n")
