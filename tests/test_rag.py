import importlib.util
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.rag import GraphRAGRetriever

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None


class TestGraphRAGRetriever(unittest.TestCase):

    def test_retrieve_seeds_keyword_fallback(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "block_id": "b1",
                "statement": "Mitochondria produce ATP",
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "reliability": "verified",
                "quality": 9,
            },
            {
                "block_id": "b2",
                "statement": "Nucleus stores DNA",
                "subject": "Nucleus",
                "predicate": "stores",
                "object": "DNA",
                "reliability": "verified",
                "quality": 8,
            },
        ]

        retriever = GraphRAGRetriever(store=mock_store)

        seeds = retriever.retrieve_seeds("Where is ATP produced?", limit=2)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["block_id"], "b1")

        seeds_all = retriever.retrieve_seeds("", limit=2)
        self.assertEqual(len(seeds_all), 2)

        seeds_none = retriever.retrieve_seeds("XYZ non-matching query", limit=2)
        self.assertEqual(len(seeds_none), 0)

    def test_exclude_unverified_filters_seeds_before_limit(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "block_id": "u1",
                "statement": "ATP is fake energy.",
                "subject": "ATP",
                "predicate": "is",
                "object": "fake energy",
                "reliability": "UNVERIFIED",
                "quality": 1,
            },
            {
                "block_id": "v1",
                "statement": "ATP powers cellular work.",
                "subject": "ATP",
                "predicate": "powers",
                "object": "cellular work",
                "reliability": "VERIFIED",
                "quality": 9,
            },
        ]
        retriever = GraphRAGRetriever(store=mock_store)
        seeds = retriever.retrieve_seeds("ATP", limit=1, exclude_unverified=True)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["block_id"], "v1")

    def test_short_keyword_token_boundary(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "block_id": "brain",
                "statement": "Brain activation consumes oxygen",
                "subject": "Brain activation",
                "predicate": "consumes",
                "object": "oxygen",
                "reliability": "VERIFIED",
            },
            {
                "block_id": "ai1",
                "statement": "AI systems summarize documents",
                "subject": "AI",
                "predicate": "summarizes",
                "object": "documents",
                "reliability": "VERIFIED",
            },
        ]
        retriever = GraphRAGRetriever(store=mock_store)
        seeds = retriever.retrieve_seeds("AI", limit=5)
        self.assertEqual([s["block_id"] for s in seeds], ["ai1"])

    def test_none_fields_and_format_helpers(self):
        mock_store = MagicMock()
        mock_store.count.return_value = 1
        mock_store.query.return_value = [
            {
                "block_id": "n1",
                "statement": None,
                "subject": None,
                "predicate": None,
                "object": None,
                "reliability": None,
            }
        ]
        retriever = GraphRAGRetriever(store=mock_store)
        self.assertEqual(retriever.retrieve_seeds("anything", limit=2), [])
        formatted = retriever.format_facts(mock_store.query.return_value)
        self.assertIn("UNVERIFIED", formatted)

        unverified_only_store = MagicMock()
        unverified_only_store.query.return_value = [
            {
                "block_id": "u1",
                "statement": "ATP claim",
                "subject": "ATP",
                "predicate": "is",
                "object": "energy",
                "reliability": "UNVERIFIED",
            }
        ]
        r2 = GraphRAGRetriever(store=unverified_only_store)
        self.assertIn(
            "No relevant facts",
            r2.format_context("ATP", exclude_unverified=True),
        )
        self.assertIn(
            "ATP claim",
            r2.format_context("ATP", exclude_unverified=False),
        )

    @patch("knowledge_graph_pkg.rag.get_embedder")
    def test_retrieve_seeds_vector_and_cache(self, mock_get_embedder):
        mock_embedder_inst = MagicMock()
        mock_embedder_inst.embed_one.side_effect = (
            lambda text: [1.0, 0.0]
            if "ATP" in text or "Mitochondria" in text
            else [0.0, 1.0]
        )
        mock_get_embedder.return_value = mock_embedder_inst

        facts = [
            {
                "block_id": "b1",
                "statement": "Mitochondria produce ATP",
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "reliability": "verified",
                "quality": 9,
            },
            {
                "block_id": "b2",
                "statement": "Nucleus stores DNA",
                "subject": "Nucleus",
                "predicate": "stores",
                "object": "DNA",
                "reliability": "verified",
                "quality": 8,
            },
        ]
        mock_store = MagicMock()
        mock_store.count.return_value = 2
        mock_store.query.side_effect = (
            lambda cypher, params=None: facts if "f.statement" in cypher else []
        )

        retriever = GraphRAGRetriever(
            store=mock_store, embedder_type="sentence-transformers"
        )

        seeds = retriever.retrieve_seeds("Mitochondria ATP generation", limit=1)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["block_id"], "b1")
        calls_after_first = mock_embedder_inst.embed_one.call_count

        # Repeat with same query: statement & query vectors are cached, zero extra embed calls.
        seeds2 = retriever.retrieve_seeds("Mitochondria ATP generation", limit=1)
        self.assertEqual(seeds2[0]["block_id"], "b1")
        self.assertEqual(mock_embedder_inst.embed_one.call_count, calls_after_first)

        # invalidate_cache forces re-embedding on next query.
        retriever.invalidate_cache()
        retriever.retrieve_seeds("Mitochondria ATP generation", limit=1)
        self.assertGreater(mock_embedder_inst.embed_one.call_count, calls_after_first)

    def test_retrieve_graph_connections(self):
        mock_store = MagicMock()

        def mock_query(cypher, params=None):
            if "MATCH (f:Fact)" in cypher:
                return [
                    {
                        "block_id": "b1",
                        "statement": "Mitochondria produce ATP",
                        "subject": "Mitochondria",
                        "predicate": "produce",
                        "object": "ATP",
                        "reliability": "verified",
                        "quality": 9,
                    }
                ]
            elif "MATCH (a:Fact)-[r:RELATED]" in cypher:
                return [
                    {
                        "block_id": "b2",
                        "statement": "ATP is the cell energy currency",
                        "subject": "ATP",
                        "predicate": "is",
                        "object": "cell energy currency",
                        "reliability": "verified",
                        "quality": 8,
                        "rel_predicate": "RELATED",
                    }
                ]
            return []

        mock_store.query.side_effect = mock_query

        retriever = GraphRAGRetriever(store=mock_store)
        results = retriever.retrieve("Mitochondria", top_k=1)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["block_id"], "b1")
        self.assertEqual(results[1]["block_id"], "b2")

    def test_format_context(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "block_id": "b1",
                "statement": "Mitochondria produce ATP",
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "reliability": "verified",
                "quality": 9,
            }
        ]

        retriever = GraphRAGRetriever(store=mock_store)
        context = retriever.format_context("Mitochondria", top_k=1)
        self.assertIn("Mitochondria", context)
        self.assertIn("produce", context)
        self.assertIn("ATP", context)

    def test_pagerank_calculation(self):
        mock_store = MagicMock()
        mock_store.query.side_effect = [
            [{"block_id": "b1"}, {"block_id": "b2"}],
            [{"src": "b1", "dst": "b2"}],
        ]

        retriever = GraphRAGRetriever(store=mock_store)
        scores = retriever.calculate_pagerank()
        self.assertIn("b1", scores)
        self.assertIn("b2", scores)
        self.assertGreater(scores["b2"], scores["b1"])

    def test_multihop_traversal(self):
        mock_store = MagicMock()

        def mock_query(cypher, params=None):
            if "MATCH (f:Fact) RETURN f.block_id" in cypher:
                return [{"block_id": "b1"}, {"block_id": "b2"}, {"block_id": "b3"}]
            if "MATCH (a:Fact)-[r:RELATED]->(b:Fact)" in cypher:
                return [{"src": "b1", "dst": "b2"}, {"src": "b2", "dst": "b3"}]
            if "MATCH (f:Fact) " in cypher:
                return [
                    {
                        "block_id": "b1",
                        "statement": "A relates to B",
                        "subject": "A",
                        "predicate": "relates to",
                        "object": "B",
                    }
                ]
            if "MATCH (a:Fact)-[r:RELATED]-(b:Fact)" in cypher:
                if params and params.get("bid") == "b1":
                    return [
                        {
                            "block_id": "b2",
                            "statement": "B relates to C",
                            "subject": "B",
                            "predicate": "relates to",
                            "object": "C",
                        }
                    ]
                if params and params.get("bid") == "b2":
                    return [
                        {
                            "block_id": "b3",
                            "statement": "C relates to D",
                            "subject": "C",
                            "predicate": "relates to",
                            "object": "D",
                        }
                    ]
            return []

        mock_store.query.side_effect = mock_query
        retriever = GraphRAGRetriever(store=mock_store)
        results = retriever.retrieve("A", top_k=2, hops=2)
        bids = [r["block_id"] for r in results]
        self.assertIn("b1", bids)
        self.assertIn("b2", bids)
        self.assertIn("b3", bids)

    def test_exclude_unverified_filtering(self):
        mock_store = MagicMock()

        def mock_query(cypher, params=None):
            if "MATCH (f:Fact)" in cypher and "f.statement" not in cypher:
                return [{"block_id": "b1"}, {"block_id": "b2"}]
            if "MATCH (a:Fact)-[r:RELATED]->(b:Fact)" in cypher:
                return [{"src": "b1", "dst": "b2"}]
            if "MATCH (f:Fact)" in cypher and "f.statement" in cypher:
                return [
                    {
                        "block_id": "b1",
                        "statement": "A relates to B",
                        "subject": "A",
                        "predicate": "relates to",
                        "object": "B",
                        "reliability": "VERIFIED",
                    },
                    {
                        "block_id": "b2",
                        "statement": "B relates to C",
                        "subject": "B",
                        "predicate": "relates to",
                        "object": "C",
                        "reliability": "UNVERIFIED",
                    },
                ]
            if "MATCH (a:Fact)-[r:RELATED]-(b:Fact)" in cypher:
                if "reliability <> 'UNVERIFIED'" in cypher:
                    return []
                return [
                    {
                        "block_id": "b2",
                        "statement": "B relates to C",
                        "subject": "B",
                        "predicate": "relates to",
                        "object": "C",
                        "reliability": "UNVERIFIED",
                    }
                ]
            return []

        mock_store.query.side_effect = mock_query
        retriever = GraphRAGRetriever(store=mock_store)

        results_filtered = retriever.retrieve(
            "A", top_k=2, hops=1, exclude_unverified=True
        )
        bids_filtered = [r["block_id"] for r in results_filtered]
        self.assertIn("b1", bids_filtered)
        self.assertNotIn("b2", bids_filtered)

        results_all = retriever.retrieve("A", top_k=2, hops=1, exclude_unverified=False)
        bids_all = [r["block_id"] for r in results_all]
        self.assertIn("b1", bids_all)
        self.assertIn("b2", bids_all)

    def test_graph_rag_retrieve_tool(self):
        from knowledge_graph_pkg.graph_tool import TOOL_SCHEMAS, GraphTools

        mock_store = MagicMock()
        mock_store.count.return_value = 1
        mock_store.query.return_value = [
            {
                "block_id": "b1",
                "statement": "A relates to B",
                "subject": "A",
                "predicate": "relates to",
                "object": "B",
                "reliability": "VERIFIED",
            }
        ]

        tools = GraphTools(store=mock_store)
        results = tools.graph_rag_retrieve(query="A", top_k=1, hops=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["block_id"], "b1")

        tool_names = [t["name"] for t in TOOL_SCHEMAS]
        self.assertIn("graph_rag_retrieve", tool_names)

    def test_conflict_aware_pruning(self):
        mock_store = MagicMock()

        def mock_query(cypher, params=None):
            if "MATCH (a:Fact), (b:Fact)" in cypher:
                return [{"subject": "Mitochondria", "object": "ATP"}]
            if "MATCH (f:Fact)" in cypher and "f.statement" in cypher:
                return [
                    {
                        "block_id": "b1",
                        "statement": "Mitochondria produces ATP",
                        "subject": "Mitochondria",
                        "predicate": "produces",
                        "object": "ATP",
                        "reliability": "VERIFIED",
                    }
                ]
            if "MATCH (a:Fact)-[r:RELATED]-(b:Fact)" in cypher:
                return [
                    {
                        "block_id": "b2",
                        "statement": "Mitochondria produces ATP",
                        "subject": "Mitochondria",
                        "predicate": "produces",
                        "object": "ATP",
                        "reliability": "VERIFIED",
                    }
                ]
            return []

        mock_store.query.side_effect = mock_query
        retriever = GraphRAGRetriever(store=mock_store)

        results = retriever.retrieve("Mitochondria", top_k=2, hops=1)
        bids = [r["block_id"] for r in results]
        self.assertIn("b1", bids)
        self.assertNotIn("b2", bids)

    def test_graph_distill_ontology_tool(self):
        from knowledge_graph_pkg.graph_tool import TOOL_SCHEMAS, GraphTools

        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "subject": "ATP",
                "predicate": "is a type of",
                "object": "ChemicalCompound",
                "child": "ATP",
                "parent": "ChemicalCompound",
            }
        ]

        tools = GraphTools(store=mock_store)
        res = tools.graph_distill_ontology()

        self.assertIn("taxonomy", res)
        self.assertIn("ChemicalCompound", res["taxonomy"])
        self.assertIn("ATP", res["taxonomy"]["ChemicalCompound"])

        tool_names = [t["name"] for t in TOOL_SCHEMAS]
        self.assertIn("graph_distill_ontology", tool_names)


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu not installed")
class TestGraphRAGKuzuIntegration(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_rag_kuzu_")
        self.store = get_graph_store(os.path.join(self.temp_dir, "kdb"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.temp_dir)

    def test_multi_predicate_and_reconciled_contradiction_not_falsely_pruned(self):
        self.store.ingest_facts(
            [
                {
                    "subject": "Cells",
                    "predicate": "contain",
                    "object": "Mitochondria",
                    "fact_statement": "Cells contain Mitochondria.",
                    "reliability_rating": "VERIFIED",
                    "quality_score": 9,
                },
                {
                    "subject": "Mitochondria",
                    "predicate": "produces",
                    "object": "ATP",
                    "fact_statement": "Mitochondria produces ATP.",
                    "reliability_rating": "VERIFIED",
                    "quality_score": 9,
                },
                {
                    "subject": "Mitochondria",
                    "predicate": "stores",
                    "object": "ATP",
                    "fact_statement": "Mitochondria stores ATP.",
                    "reliability_rating": "VERIFIED",
                    "quality_score": 8,
                },
                {
                    "subject": "Mitochondria",
                    "predicate": "do not produce",
                    "object": "ATP",
                    "fact_statement": "Mitochondria do not produce ATP.",
                    "reliability_rating": "UNVERIFIED",
                    "quality_score": 2,
                },
            ]
        )
        self.store.auto_link_relations()
        retriever = GraphRAGRetriever(store=self.store)
        results = retriever.retrieve("Cells", top_k=2, hops=1, exclude_unverified=True)
        stmts = {r["statement"] for r in results}
        self.assertIn("Cells contain Mitochondria.", stmts)
        self.assertIn("Mitochondria produces ATP.", stmts)
        self.assertIn("Mitochondria stores ATP.", stmts)
        self.assertNotIn("Mitochondria do not produce ATP.", stmts)

    def test_active_unresolved_contradiction_pruned(self):
        self.store.ingest_facts(
            [
                {
                    "subject": "Cells",
                    "predicate": "contain",
                    "object": "Mitochondria",
                    "fact_statement": "Cells contain Mitochondria.",
                    "reliability_rating": "VERIFIED",
                    "quality_score": 9,
                },
                {
                    "subject": "Mitochondria",
                    "predicate": "produce",
                    "object": "ATP",
                    "fact_statement": "Mitochondria produce ATP.",
                    "reliability_rating": "LIKELY_TRUE",
                    "quality_score": 7,
                },
                {
                    "subject": "Mitochondria",
                    "predicate": "do not produce",
                    "object": "ATP",
                    "fact_statement": "Mitochondria do not produce ATP.",
                    "reliability_rating": "LIKELY_TRUE",
                    "quality_score": 7,
                },
            ]
        )
        self.store.auto_link_relations()
        retriever = GraphRAGRetriever(store=self.store)
        results = retriever.retrieve("Cells", top_k=2, hops=1, exclude_unverified=True)
        stmts = {r["statement"] for r in results}
        self.assertIn("Cells contain Mitochondria.", stmts)
        self.assertNotIn("Mitochondria produce ATP.", stmts)
        self.assertNotIn("Mitochondria do not produce ATP.", stmts)

    def test_pagerank_cache_invalidates_on_new_edges(self):
        self.store.ingest_facts(
            [
                {
                    "subject": "Mitochondria",
                    "predicate": "produce",
                    "object": "ATP",
                    "fact_statement": "Mitochondria produce ATP.",
                    "reliability_rating": "VERIFIED",
                },
                {
                    "subject": "ATP",
                    "predicate": "fuels",
                    "object": "Cells",
                    "fact_statement": "ATP fuels Cells.",
                    "reliability_rating": "VERIFIED",
                },
            ]
        )
        retriever = GraphRAGRetriever(store=self.store)
        scores_before = retriever.calculate_pagerank()
        vals_before = list(scores_before.values())
        self.assertAlmostEqual(vals_before[0], vals_before[1], places=5)

        self.store.auto_link_relations()
        self.assertEqual(self.store.count(), 2)
        self.assertEqual(self.store.edge_count(), 1)

        scores_after = retriever.calculate_pagerank()
        self.assertGreater(len(set(round(v, 5) for v in scores_after.values())), 1)
