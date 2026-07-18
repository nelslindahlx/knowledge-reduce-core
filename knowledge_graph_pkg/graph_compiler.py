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
    chains = store.query(
        "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
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

def save_compiled_instructions(instructions: List[Dict[str, Any]], output_path: str) -> None:
    """Save the compiled instructions as a standard instruction JSONL dataset."""
    with open(output_path, "w", encoding="utf-8") as fh:
        for inst in instructions:
            fh.write(json.dumps(inst) + "\n")
