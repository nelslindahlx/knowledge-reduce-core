import os
import unittest
import tempfile
import shutil
from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.entity_resolution import resolve_and_merge_entities, _get_blocking_keys

class TestEntityResolution(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_er_")
        self.store = get_graph_store(os.path.join(self.temp_dir, "kdb"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def test_blocking_keys(self):
        keys = _get_blocking_keys("Adenosine Triphosphate (ATP)")
        # Should include first characters of words: 'a', 't', and 'at'
        self.assertIn("a", keys)
        self.assertIn("t", keys)
        self.assertIn("at", keys)

    def test_entity_resolution_with_blocking(self):
        facts = [
            {
                "subject": "Adenosine Triphosphate (ATP)",
                "predicate": "fuels",
                "object": "CellularProcess",
                "fact_statement": "Adenosine Triphosphate (ATP) fuels CellularProcess.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1.0,
                "source_models": "gpt-4"
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "CellularProcess",
                "fact_statement": "ATP fuels CellularProcess.",
                "domain": "biology",
                "reliability_rating": "LIKELY_TRUE",
                "cross_model_agreement": 2,
                "quality_score": 1.0,
                "source_models": "gpt-3.5"
            }
        ]
        self.store.ingest_facts(facts)
        self.assertEqual(self.store.count(), 2)

        # Run resolution
        report = resolve_and_merge_entities(self.store, threshold=0.85)
        self.assertEqual(report["resolved_clusters"], 1)
        self.assertEqual(report["merged_nodes"], 1)

        # Verify duplicate was merged and name collapsed to longest: "Adenosine Triphosphate (ATP)"
        self.assertEqual(self.store.count(), 1)
        res = self.store.query("MATCH (f:Fact) RETURN f.subject AS subject")
        self.assertEqual(res[0]["subject"], "Adenosine Triphosphate (ATP)")

    def test_incremental_resolution_with_limits(self):
        facts = [
            {
                "subject": "Adenosine Triphosphate (ATP)",
                "predicate": "fuels",
                "object": "CellularProcess",
                "fact_statement": "Adenosine Triphosphate (ATP) fuels CellularProcess.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1.0,
                "source_models": "gpt-4"
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "CellularProcess",
                "fact_statement": "ATP fuels CellularProcess.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1.0,
                "source_models": "gpt-4"
            },
            {
                "subject": "Glucose",
                "predicate": "is a type of",
                "object": "Sugar",
                "fact_statement": "Glucose is a type of Sugar.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1.0,
                "source_models": "gpt-4"
            },
            {
                "subject": "D-Glucose (Sugar)",
                "predicate": "is a type of",
                "object": "Sugar",
                "fact_statement": "D-Glucose (Sugar) is a type of Sugar.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1.0,
                "source_models": "gpt-4"
            }
        ]
        self.store.ingest_facts(facts)

        # Run resolution limiting ONLY to "ATP"
        report = resolve_and_merge_entities(self.store, threshold=0.85, limit_to_concepts=["ATP"])
        
        # Should resolve ATP cluster, but NOT Glucose cluster!
        self.assertEqual(report["resolved_clusters"], 1)
        self.assertEqual(report["merged_nodes"], 1)
        
        # Verify ATP facts merged, Glucose facts remained separate
        res = self.store.query("MATCH (f:Fact) RETURN f.subject AS subject")
        subjects = [r["subject"] for r in res]
        
        self.assertIn("Adenosine Triphosphate (ATP)", subjects)
        self.assertIn("Glucose", subjects)
        self.assertIn("D-Glucose (Sugar)", subjects)
