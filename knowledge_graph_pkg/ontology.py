"""
Ontology & Graph Schema Distillation.

Extracts concept class hierarchies and aggregates instance-level facts
into high-level relationship schemas.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from .graph_store_base import BaseGraphStore

class OntologyDistiller:
    """Analyze a facts graph to distill a high-level concept taxonomy and relationship schema."""

    def __init__(self, store: BaseGraphStore, rules: Optional[Dict[str, Any]] = None):
        self.store = store
        
        # Default heuristics rules
        self.rules = {
            "process_suffixes": ["ation", "ing", "sis", "process"],
            "entity_outgoing_predicates": ["produces", "fuels", "synthesizes", "contains", "causes", "produce"],
            "location_incoming_predicates": ["is located in", "is part of", "occurs in", "located in", "part of"],
            "attribute_outgoing_predicates": ["is", "has", "exhibits"],
            "taxonomy_predicates": ["is a", "type of", "subclass of", "is a type of"]
        }
        if rules:
            self.rules.update(rules)

    def distill_taxonomy(self) -> Dict[str, List[str]]:
        """Extract taxonomic hierarchies from 'is a' / 'type of' facts.
        
        Returns a dictionary mapping parent classes to child classes.
        """
        # Query subclass/instance relations securely using parameterized query
        subclasses = self.store.query(
            "MATCH (a:Fact) "
            "WHERE lower(a.predicate) IN $predicates "
            "RETURN a.subject AS child, a.object AS parent",
            {"predicates": [p.lower() for p in self.rules["taxonomy_predicates"]]}
        )
        
        taxonomy: Dict[str, List[str]] = {}
        for row in subclasses:
            child = str(row.get("child", "")).strip()
            parent = str(row.get("parent", "")).strip()
            if not child or not parent:
                continue
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
            s = str(f.get("subject", "")).strip()
            p = str(f.get("predicate", "")).strip()
            o = str(f.get("object", "")).strip()
            if not s or not o:
                continue
            for concept in (s, o):
                if concept not in concept_predicates:
                    concept_predicates[concept] = set()
            if p:
                concept_predicates[s].add(f"out:{p}")
                concept_predicates[o].add(f"in:{p}")
            
        semantic_types: Dict[str, str] = {}
        for concept, preds in concept_predicates.items():
            concept_lower = concept.lower()
            
            # Suffix checks
            if any(concept_lower.endswith(sfx) for sfx in self.rules["process_suffixes"]):
                semantic_types[concept] = "PROCESS"
                continue
                
            out_preds = {p.split(":", 1)[1] for p in preds if p.startswith("out:")}
            in_preds = {p.split(":", 1)[1] for p in preds if p.startswith("in:")}
            
            if any(p in out_preds for p in self.rules["entity_outgoing_predicates"]):
                semantic_types[concept] = "ENTITY"
            elif any(p in in_preds for p in self.rules["location_incoming_predicates"]):
                semantic_types[concept] = "LOCATION"
            elif any(p in out_preds for p in self.rules["attribute_outgoing_predicates"]):
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
            s = str(f.get("subject", "")).strip()
            p = str(f.get("predicate", "")).strip()
            o = str(f.get("object", "")).strip()
            if not s or not o or not p:
                continue
            type_s = sem_types.get(s, "CONCEPT")
            type_o = sem_types.get(o, "CONCEPT")
            schema_triplets.add((type_s, p, type_o))
            
        return [
            {"subject_type": t[0], "predicate": t[1], "object_type": t[2]}
            for t in sorted(schema_triplets)
        ]
