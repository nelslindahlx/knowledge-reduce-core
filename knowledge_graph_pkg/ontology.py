"""
Ontology & Graph Schema Distillation.

Extracts concept class hierarchies and aggregates instance-level facts
into high-level relationship schemas.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from .graph_store_base import BaseGraphStore, _s


class OntologyDistiller:
    """Analyze a facts graph to distill a high-level concept taxonomy and relationship schema."""

    def __init__(self, store: BaseGraphStore, rules: Optional[Dict[str, Any]] = None):
        self.store = store
        self.rules = {
            "process_suffixes": ["ation", "ing", "sis", "process"],
            "entity_outgoing_predicates": [
                "produces",
                "fuels",
                "synthesizes",
                "contains",
                "causes",
                "produce",
            ],
            "location_incoming_predicates": [
                "is located in",
                "is part of",
                "occurs in",
                "located in",
                "part of",
            ],
            "attribute_outgoing_predicates": ["is", "has", "exhibits"],
            "taxonomy_predicates": ["is a", "type of", "subclass of", "is a type of"],
        }
        if rules:
            self.rules.update(rules)

    def _fetch_facts(self) -> List[Dict[str, Any]]:
        return self.store.query(
            "MATCH (f:Fact) RETURN f.subject AS subject, f.predicate AS predicate, f.object AS object"
        )

    def distill_taxonomy(
        self, facts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, List[str]]:
        """Extract taxonomic hierarchies from 'is a' / 'type of' facts."""
        tax_preds = [
            _s(p).strip().lower()
            for p in self.rules.get("taxonomy_predicates", [])
            if _s(p).strip()
        ]
        if not tax_preds:
            return {}

        if facts is None:
            subclasses = self.store.query(
                "MATCH (a:Fact) "
                "WHERE lower(a.predicate) IN $predicates "
                "RETURN a.subject AS child, a.object AS parent",
                {"predicates": tax_preds},
            )
        else:
            tax_set = set(tax_preds)
            subclasses = [
                f for f in facts if _s(f.get("predicate")).strip().lower() in tax_set
            ]

        taxonomy: Dict[str, List[str]] = {}
        for row in subclasses:
            raw_child = row["child"] if "child" in row else row.get("subject")
            raw_parent = row["parent"] if "parent" in row else row.get("object")
            child = _s(raw_child).strip()
            parent = _s(raw_parent).strip()
            if not child or not parent:
                continue
            children = taxonomy.setdefault(parent, [])
            if child not in children:
                children.append(child)

        return taxonomy

    def infer_semantic_types(
        self, facts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, str]:
        """Infer semantic types (ENTITY, LOCATION, PROCESS, ATTRIBUTE, CONCEPT) for nodes."""
        rows = self._fetch_facts() if facts is None else facts

        concept_predicates: Dict[str, Set[str]] = {}
        for f in rows:
            s = _s(f.get("subject")).strip()
            p = _s(f.get("predicate")).strip().lower()
            o = _s(f.get("object")).strip()
            if not s or not o:
                continue
            concept_predicates.setdefault(s, set())
            concept_predicates.setdefault(o, set())
            if p:
                concept_predicates[s].add(f"out:{p}")
                concept_predicates[o].add(f"in:{p}")

        entity_out = {
            _s(p).strip().lower()
            for p in self.rules.get("entity_outgoing_predicates", [])
            if _s(p).strip()
        }
        loc_in = {
            _s(p).strip().lower()
            for p in self.rules.get("location_incoming_predicates", [])
            if _s(p).strip()
        }
        attr_out = {
            _s(p).strip().lower()
            for p in self.rules.get("attribute_outgoing_predicates", [])
            if _s(p).strip()
        }
        proc_suffixes = [
            _s(sfx).strip().lower()
            for sfx in self.rules.get("process_suffixes", [])
            if _s(sfx).strip()
        ]

        semantic_types: Dict[str, str] = {}
        for concept, preds in concept_predicates.items():
            concept_lower = concept.lower()
            out_preds = {p.split(":", 1)[1] for p in preds if p.startswith("out:")}
            in_preds = {p.split(":", 1)[1] for p in preds if p.startswith("in:")}

            if out_preds & entity_out:
                semantic_types[concept] = "ENTITY"
            elif in_preds & loc_in:
                semantic_types[concept] = "LOCATION"
            elif proc_suffixes and any(
                concept_lower.endswith(sfx) for sfx in proc_suffixes
            ):
                semantic_types[concept] = "PROCESS"
            elif out_preds & attr_out:
                semantic_types[concept] = "ATTRIBUTE"
            else:
                semantic_types[concept] = "CONCEPT"

        return semantic_types

    def infer_relation_schema(
        self,
        facts: Optional[List[Dict[str, Any]]] = None,
        semantic_types: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """Summarize instance-level connections into a high-level schema graph."""
        rows = self._fetch_facts() if facts is None else facts
        sem_types = (
            self.infer_semantic_types(facts=rows)
            if semantic_types is None
            else semantic_types
        )

        schema_triplets: Set[Tuple[str, str, str]] = set()
        for f in rows:
            s = _s(f.get("subject")).strip()
            p = _s(f.get("predicate")).strip()
            o = _s(f.get("object")).strip()
            if not s or not o or not p:
                continue
            type_s = sem_types.get(s, "CONCEPT")
            type_o = sem_types.get(o, "CONCEPT")
            schema_triplets.add((type_s, p, type_o))

        return [
            {"subject_type": t[0], "predicate": t[1], "object_type": t[2]}
            for t in sorted(schema_triplets)
        ]

    def distill_all(
        self, facts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Distill taxonomy, semantic types, and relation schema in a single graph scan."""
        rows = self._fetch_facts() if facts is None else facts
        taxonomy = self.distill_taxonomy(facts=rows)
        semantic_types = self.infer_semantic_types(facts=rows)
        relation_schema = self.infer_relation_schema(
            facts=rows, semantic_types=semantic_types
        )
        return {
            "taxonomy": taxonomy,
            "semantic_types": semantic_types,
            "relation_schema": relation_schema,
        }
