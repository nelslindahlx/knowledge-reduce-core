import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.ontology import OntologyDistiller

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None


class TestOntologyDistillerPure(unittest.TestCase):

    def test_case_insensitive_predicates_and_none_safety(self):
        mock_store = MagicMock()
        rows = [
            {"subject": "MITO", "predicate": "PRODUCES", "object": "ATP"},
            {"subject": "Respiration", "predicate": "OCCURS IN", "object": "CYTO"},
            {"subject": "ATP", "predicate": "IS A", "object": "Energy"},
            {"subject": None, "predicate": "IS A", "object": "Ghost"},
        ]
        mock_store.query.return_value = rows
        distiller = OntologyDistiller(mock_store)

        sem_types = distiller.infer_semantic_types()
        self.assertEqual(sem_types["MITO"], "ENTITY")
        self.assertEqual(sem_types["CYTO"], "LOCATION")
        self.assertEqual(sem_types["Respiration"], "PROCESS")
        self.assertNotIn("None", sem_types)

        taxonomy = distiller.distill_taxonomy(facts=rows)
        self.assertEqual(taxonomy, {"Energy": ["ATP"]})
        self.assertNotIn("Ghost", taxonomy)

    def test_predicate_priority_before_process_suffix(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"subject": "Research", "predicate": "occurs in", "object": "Building"},
            {"subject": "Station", "predicate": "produces", "object": "Power"},
            {"subject": "Respiration", "predicate": "occurs in", "object": "Cytoplasm"},
        ]
        distiller = OntologyDistiller(mock_store)
        sem_types = distiller.infer_semantic_types()
        self.assertEqual(sem_types["Building"], "LOCATION")
        self.assertEqual(sem_types["Station"], "ENTITY")
        self.assertEqual(sem_types["Respiration"], "PROCESS")

    def test_distill_all_single_pass_query(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"subject": "ATP", "predicate": "is a", "object": "Molecule"},
            {"subject": "Mitochondria", "predicate": "produces", "object": "ATP"},
        ]
        distiller = OntologyDistiller(mock_store)
        summary = distiller.distill_all()
        self.assertEqual(mock_store.query.call_count, 1)
        self.assertEqual(summary["taxonomy"], {"Molecule": ["ATP"]})
        self.assertEqual(summary["semantic_types"]["Mitochondria"], "ENTITY")
        self.assertGreater(len(summary["relation_schema"]), 0)


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu not installed")
class TestOntologyDistiller(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_ontology_")
        self.store = get_graph_store(os.path.join(self.temp_dir, "kdb"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def test_ontology_distillation_taxonomy_and_schema(self):
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
                "source_models": "gpt-4",
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
                "source_models": "gpt-4",
            },
            {
                "subject": "Mitochondria",
                "predicate": "produces",
                "object": "ATP",
                "fact_statement": "Mitochondria produces ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
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
                "source_models": "gpt-4",
            },
        ]
        self.store.ingest_facts(facts)

        distiller = OntologyDistiller(self.store)

        taxonomy = distiller.distill_taxonomy()
        self.assertIn("ChemicalCompound", taxonomy)
        self.assertIn("ATP", taxonomy["ChemicalCompound"])
        self.assertIn("Glucose", taxonomy["ChemicalCompound"])

        sem_types = distiller.infer_semantic_types()
        self.assertEqual(sem_types.get("Mitochondria"), "ENTITY")
        self.assertEqual(sem_types.get("Respiration"), "PROCESS")
        self.assertEqual(sem_types.get("Cytoplasm"), "LOCATION")

        schema = distiller.infer_relation_schema()
        self.assertTrue(len(schema) > 0)

        has_produces = any(
            s["subject_type"] == "ENTITY"
            and s["predicate"] == "produces"
            and s["object_type"] == "CONCEPT"
            for s in schema
        )
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
                "source_models": "gpt-4",
            }
        ]
        self.store.ingest_facts(facts)

        custom_rules = {"taxonomy_predicates": ["part_of_pathway"]}
        distiller = OntologyDistiller(self.store, rules=custom_rules)
        taxonomy = distiller.distill_taxonomy()

        self.assertIn("Metabolism", taxonomy)
        self.assertIn("Respiration", taxonomy["Metabolism"])
