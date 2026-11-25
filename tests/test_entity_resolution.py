import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

import pytest

from knowledge_graph_pkg.entity_resolution import (
    _get_blocking_keys,
    _jaccard_similarity,
    _replace_entity_in_statement,
    resolve_and_merge_entities,
)
from knowledge_graph_pkg.graph_store_factory import get_graph_store

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None


class TestEntityResolutionPure(unittest.TestCase):

    def test_no_false_positive_substring_merges(self):
        self.assertLess(_jaccard_similarity("AI", "Brain Activation"), 0.85)
        self.assertLess(_jaccard_similarity("Fe", "Infection"), 0.85)
        self.assertLess(_jaccard_similarity("Cell", "Fuel Cell"), 0.85)
        self.assertLess(_jaccard_similarity("Ion", "Respiration"), 0.85)
        self.assertLess(_jaccard_similarity("DNA", "Mitochondrial DNA"), 0.85)
        self.assertLess(_jaccard_similarity("Sugar", "D-Glucose (Sugar)"), 0.85)
        self.assertLess(_jaccard_similarity("ray", "X-ray"), 0.85)

    def test_valid_alias_and_acronym_merges(self):
        self.assertEqual(_jaccard_similarity("ATP", "Adenosine Triphosphate (ATP)"), 1.0)
        self.assertEqual(_jaccard_similarity("Glucose", "D-Glucose (Sugar)"), 1.0)
        self.assertEqual(_jaccard_similarity("LLM", "Large Language Model"), 1.0)
        self.assertEqual(_jaccard_similarity("USA", "United States of America"), 1.0)
        self.assertEqual(_jaccard_similarity("CO2", "Carbon Dioxide (CO2)"), 1.0)

    def test_statement_word_boundary_rewrite(self):
        stmt = "ATP powers ATPase enzymes via ATP hydrolysis."
        rewritten = _replace_entity_in_statement(
            stmt, "ATP", "Adenosine Triphosphate (ATP)"
        )
        self.assertEqual(
            rewritten,
            "Adenosine Triphosphate (ATP) powers ATPase enzymes via "
            "Adenosine Triphosphate (ATP) hydrolysis.",
        )

    def test_no_clusters_performs_no_mutations(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "block_id": "b1",
                "statement": None,
                "subject": "AI",
                "predicate": "helps",
                "object": "Brain Activation",
                "domain": None,
                "reliability": "VERIFIED",
                "agreement": 1,
                "quality": 5,
                "source_models": None,
            }
        ]
        res = resolve_and_merge_entities(mock_store, threshold=0.85)
        self.assertEqual(res["resolved_clusters"], 0)
        self.assertEqual(res["merged_nodes"], 0)
        self.assertEqual(mock_store.query.call_count, 1)


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu not installed")
class TestEntityResolution(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_er_")
        self.store = get_graph_store(os.path.join(self.temp_dir, "kdb"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def test_blocking_keys(self):
        keys = _get_blocking_keys("Adenosine Triphosphate (ATP)")
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
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "CellularProcess",
                "fact_statement": "ATP fuels CellularProcess.",
                "domain": "biology",
                "reliability_rating": "LIKELY_TRUE",
                "cross_model_agreement": 2,
                "quality_score": 1,
                "source_models": "gpt-3.5",
            },
        ]
        self.store.ingest_facts(facts)
        self.assertEqual(self.store.count(), 2)

        report = resolve_and_merge_entities(self.store, threshold=0.85)
        self.assertEqual(report["resolved_clusters"], 1)
        self.assertEqual(report["merged_nodes"], 1)

        self.assertEqual(self.store.count(), 1)
        res = self.store.query(
            "MATCH (f:Fact) RETURN f.subject AS subject, f.reliability AS reliability, "
            "f.agreement AS agreement, f.source_models AS source_models"
        )
        self.assertEqual(res[0]["subject"], "Adenosine Triphosphate (ATP)")
        self.assertEqual(res[0]["reliability"], "VERIFIED")
        self.assertEqual(res[0]["agreement"], 3)
        self.assertIn("gpt-4", res[0]["source_models"])
        self.assertIn("gpt-3.5", res[0]["source_models"])

    def test_reliability_tier_outranks_unverified_with_higher_agreement(self):
        facts = [
            {
                "subject": "ATP",
                "predicate": "powers",
                "object": "ATPase",
                "fact_statement": "ATP powers ATPase activity.",
                "domain": "biology",
                "reliability_rating": "UNVERIFIED",
                "cross_model_agreement": 9,
                "quality_score": 2,
                "source_models": "weak-model",
            },
            {
                "subject": "Adenosine Triphosphate (ATP)",
                "predicate": "powers",
                "object": "ATPase",
                "fact_statement": "Adenosine Triphosphate (ATP) powers ATPase activity.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 2,
                "quality_score": 9,
                "source_models": "strong-model",
            },
        ]
        self.store.ingest_facts(facts)
        report = resolve_and_merge_entities(self.store, threshold=0.85)
        self.assertEqual(report["merged_nodes"], 1)

        rows = self.store.query(
            "MATCH (f:Fact) RETURN f.reliability AS rel, f.quality AS quality, "
            "f.agreement AS agree, f.statement AS stmt, f.source_models AS models"
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rel"], "VERIFIED")
        self.assertEqual(rows[0]["quality"], 9)
        self.assertEqual(rows[0]["agree"], 9)
        self.assertIn("ATPase", rows[0]["stmt"])
        self.assertIn("strong-model", rows[0]["models"])
        self.assertIn("weak-model", rows[0]["models"])

    def test_preserves_untouched_nodes_and_remaps_related_edges(self):
        facts = [
            {
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "fact_statement": "Mitochondria produce ATP.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 9,
                "source_models": "m1",
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "cells",
                "fact_statement": "ATP fuels cells.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 9,
                "source_models": "m1",
            },
            {
                "subject": "Adenosine Triphosphate (ATP)",
                "predicate": "fuels",
                "object": "muscles",
                "fact_statement": "Adenosine Triphosphate (ATP) fuels muscles.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 9,
                "source_models": "m1",
            },
            {
                "subject": "Nucleus",
                "predicate": "stores",
                "object": "Genomic DNA",
                "fact_statement": "Nucleus stores Genomic DNA.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 9,
                "source_models": "m1",
            },
            {
                "subject": "Genomic DNA",
                "predicate": "encodes",
                "object": "Proteins",
                "fact_statement": "Genomic DNA encodes Proteins.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 9,
                "source_models": "m1",
            },
        ]
        self.store.ingest_facts(facts)
        self.store.auto_link_relations()
        self.assertGreaterEqual(self.store.edge_count(), 2)

        report = resolve_and_merge_entities(
            self.store, threshold=0.85, limit_to_concepts=["ATP"]
        )
        self.assertEqual(report["resolved_clusters"], 1)

        edges = self.store.query(
            "MATCH (a:Fact)-[r:RELATED]->(b:Fact) "
            "RETURN a.subject AS a_subj, a.object AS a_obj, "
            "b.subject AS b_subj, b.object AS b_obj"
        )
        edge_pairs = {(e["a_subj"], e["a_obj"], e["b_subj"], e["b_obj"]) for e in edges}
        self.assertIn(("Nucleus", "Genomic DNA", "Genomic DNA", "Proteins"), edge_pairs)
        self.assertIn(
            (
                "Mitochondria",
                "Adenosine Triphosphate (ATP)",
                "Adenosine Triphosphate (ATP)",
                "cells",
            ),
            edge_pairs,
        )

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
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "CellularProcess",
                "fact_statement": "ATP fuels CellularProcess.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "Glucose",
                "predicate": "is a type of",
                "object": "Sugar",
                "fact_statement": "Glucose is a type of Sugar.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
            {
                "subject": "D-Glucose (Sugar)",
                "predicate": "is a type of",
                "object": "Sugar",
                "fact_statement": "D-Glucose (Sugar) is a type of Sugar.",
                "domain": "biology",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": "gpt-4",
            },
        ]
        self.store.ingest_facts(facts)

        report = resolve_and_merge_entities(
            self.store, threshold=0.85, limit_to_concepts=["ATP"]
        )
        self.assertEqual(report["resolved_clusters"], 1)
        self.assertEqual(report["merged_nodes"], 1)

        res = self.store.query("MATCH (f:Fact) RETURN f.subject AS subject")
        subjects = [r["subject"] for r in res]

        self.assertIn("Adenosine Triphosphate (ATP)", subjects)
        self.assertIn("Glucose", subjects)
        self.assertIn("D-Glucose (Sugar)", subjects)
