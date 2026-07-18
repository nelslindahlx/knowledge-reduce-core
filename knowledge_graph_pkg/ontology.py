"""
Ontology & Graph Schema Distillation.

Extracts concept class hierarchies and aggregates instance-level facts
into high-level relationship schemas.
"""

from typing import Any, Dict, List, Set, Tuple
from .graph_store_base import BaseGraphStore

class OntologyDistiller:
    """Analyze a facts graph to distill a high-level concept taxonomy and relationship schema."""

    def __init__(self, store: BaseGraphStore):
        self.store = store

    def distill_taxonomy(self) -> Dict[str, List[str]]:
        """Extract taxonomic hierarchies from 'is a' / 'type of' facts.
        
        Returns a dictionary mapping parent classes to child classes.
        """
        # Query subclass/instance relations
        subclasses = self.store.query(
            "MATCH (a:Fact) "
            "WHERE lower(a.predicate) IN ['is a', 'type of', 'subclass of', 'is a type of'] "
            "RETURN a.subject AS child, a.object AS parent"
        )
        
        taxonomy: Dict[str, List[str]] = {}
        for row in subclasses:
            child = str(row["child"]).strip()
            parent = str(row["parent"]).strip()
            if parent not in taxonomy:
                taxonomy[parent] = []
            if child not in taxonomy[parent]:
                taxonomy[parent].append(child)
                
        return taxonomy

    def infer_semantic_types(self) -> Dict[str, str]:
        """Infer semantic types (e.g. ENTITY, PROCESS, ATTRIBUTE) for concept nodes.
        
        Uses common predicate patterns and lexical endings to categorize nodes.
        """
        facts = self.store.query(
            "MATCH (f:Fact) RETURN f.subject AS subject, f.predicate AS predicate, f.object AS object"
        )
        
        concept_predicates: Dict[str, Set[str]] = {}
        for f in facts:
            s, p, o = str(f["subject"]), str(f["predicate"]), str(f["object"])
            for concept in (s, o):
                if concept not in concept_predicates:
                    concept_predicates[concept] = set()
            concept_predicates[s].add(f"out:{p}")
            concept_predicates[o].add(f"in:{p}")
            
        semantic_types: Dict[str, str] = {}
        for concept, preds in concept_predicates.items():
            # Heuristics
            concept_lower = concept.lower()
            if (concept_lower.endswith("ation") or 
                concept_lower.endswith("ing") or 
                concept_lower.endswith("sis") or
                concept_lower.endswith("process")):
                semantic_types[concept] = "PROCESS"
                continue
                
            out_preds = {p.split(":", 1)[1] for p in preds if p.startswith("out:")}
            in_preds = {p.split(":", 1)[1] for p in preds if p.startswith("in:")}
            
            if any(p in out_preds for p in ["produces", "fuels", "synthesizes", "contains", "causes", "produce"]):
                semantic_types[concept] = "ENTITY"
            elif any(p in in_preds for p in ["is located in", "is part of", "occurs in", "located in", "part of"]):
                semantic_types[concept] = "LOCATION"
            elif any(p in out_preds for p in ["is", "has", "exhibits"]):
                semantic_types[concept] = "ATTRIBUTE"
            else:
                semantic_types[concept] = "CONCEPT"
                
        return semantic_types

    def infer_relation_schema(self) -> List[Dict[str, Any]]:
        """Summarize all instance-level connections into a high-level schema graph.
        
        E.g. (ENTITY) -[produces]-> (ENTITY)
        """
        sem_types = self.infer_semantic_types()
        
        facts = self.store.query(
            "MATCH (f:Fact) RETURN f.subject AS subject, f.predicate AS predicate, f.object AS object"
        )
        
        schema_triplets: Set[Tuple[str, str, str]] = set()
        for f in facts:
            s, p, o = str(f["subject"]), str(f["predicate"]), str(f["object"])
            type_s = sem_types.get(s, "CONCEPT")
            type_o = sem_types.get(o, "CONCEPT")
            schema_triplets.add((type_s, p, type_o))
            
        return [
            {"subject_type": t[0], "predicate": t[1], "object_type": t[2]}
            for t in sorted(schema_triplets)
        ]
