import os
import unittest
import tempfile
import shutil
import json
from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.ontology import OntologyDistiller

class TestOntologyDistiller(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_ontology_")
        self.store = get_graph_store(os.path.join(self.temp_dir, "kdb"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def test_ontology_distillation_taxonomy_and_schema(self):
        # 1. Ingest taxonomy facts
        facts = [
            {
                "subject": "ATP",
                "predicate": "is a",
                "object": "ChemicalCompound",
                "fact_statement": "ATP is a ChemicalCompound.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4"
            },
            {
                "subject": "Glucose",
                "predicate": "is a type of",
                "object": "ChemicalCompound",
                "fact_statement": "Glucose is a type of ChemicalCompound.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4"
            },
            # Ingest relationship fact that classifies semantic types
            {
                "subject": "Mitochondria",
                "predicate": "produces",
                "object": "ATP",
                "fact_statement": "Mitochondria produces ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4"
            },
            {
                "subject": "Respiration",
                "predicate": "occurs in",
                "object": "Cytoplasm",
                "fact_statement": "Respiration occurs in Cytoplasm.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4"
            }
        ]
        self.store.ingest_facts(facts)

        distiller = OntologyDistiller(self.store)

        # 2. Test taxonomy extraction
        taxonomy = distiller.distill_taxonomy()
        self.assertIn("ChemicalCompound", taxonomy)
        self.assertIn("ATP", taxonomy["ChemicalCompound"])
        self.assertIn("Glucose", taxonomy["ChemicalCompound"])

        # 3. Test semantic type inference
        sem_types = distiller.infer_semantic_types()
        self.assertEqual(sem_types.get("Mitochondria"), "ENTITY")
        self.assertEqual(sem_types.get("Respiration"), "PROCESS")  # ends in -ation
        self.assertEqual(sem_types.get("Cytoplasm"), "LOCATION")   # in:occurs_in pattern

        # 4. Test relation schema inference
        schema = distiller.infer_relation_schema()
        self.assertTrue(len(schema) > 0)
        
        # Verify entity produces chemical compound (concept) relation is mapped
        has_produces = False
        for s in schema:
            if s["subject_type"] == "ENTITY" and s["predicate"] == "produces" and s["object_type"] == "CONCEPT":
                has_produces = True
        self.assertTrue(has_produces)

    def test_custom_rules_override(self):
        facts = [
            {
                "subject": "Respiration",
                "predicate": "part_of_pathway",
                "object": "Metabolism",
                "fact_statement": "Respiration is part_of_pathway Metabolism.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4"
            }
        ]
        self.store.ingest_facts(facts)

        # 1. Custom taxonomy predicate 'part_of_pathway'
        custom_rules = {
            "taxonomy_predicates": ["part_of_pathway"]
        }
        distiller = OntologyDistiller(self.store, rules=custom_rules)
        taxonomy = distiller.distill_taxonomy()
        
        self.assertIn("Metabolism", taxonomy)
        self.assertIn("Respiration", taxonomy["Metabolism"])
