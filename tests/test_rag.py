import unittest
from unittest.mock import MagicMock, patch
from knowledge_graph_pkg.rag import GraphRAGRetriever

class TestGraphRAGRetriever(unittest.TestCase):

    def test_retrieve_seeds_keyword_fallback(self):
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"block_id": "b1", "statement": "Mitochondria produce ATP", "subject": "Mitochondria", "predicate": "produce", "object": "ATP", "reliability": "verified", "quality": 9},
            {"block_id": "b2", "statement": "Nucleus stores DNA", "subject": "Nucleus", "predicate": "stores", "object": "DNA", "reliability": "verified", "quality": 8},
        ]
        
        retriever = GraphRAGRetriever(store=mock_store)
        
        # 1. Search for mitochondria
        seeds = retriever.retrieve_seeds("Where is ATP produced?", limit=2)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["block_id"], "b1")

        # 2. Search with empty query (returns default list slice)
        seeds_all = retriever.retrieve_seeds("", limit=2)
        self.assertEqual(len(seeds_all), 2)

        # 3. Search with no matching keyword (returns empty list)
        seeds_none = retriever.retrieve_seeds("XYZ non-matching query", limit=2)
        self.assertEqual(len(seeds_none), 0)

    @patch('knowledge_graph_pkg.rag.get_embedder')
    def test_retrieve_seeds_vector(self, mock_get_embedder):
        mock_embedder_inst = MagicMock()
        mock_embedder_inst.embed_one.side_effect = lambda text: [1.0, 0.0] if "ATP" in text or "Mitochondria" in text else [0.0, 1.0]
        mock_get_embedder.return_value = mock_embedder_inst
        
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"block_id": "b1", "statement": "Mitochondria produce ATP", "subject": "Mitochondria", "predicate": "produce", "object": "ATP", "reliability": "verified", "quality": 9},
            {"block_id": "b2", "statement": "Nucleus stores DNA", "subject": "Nucleus", "predicate": "stores", "object": "DNA", "reliability": "verified", "quality": 8},
        ]
        
        retriever = GraphRAGRetriever(store=mock_store, embedder_type="sentence-transformers")
        
        seeds = retriever.retrieve_seeds("Mitochondria ATP generation", limit=1)
        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0]["block_id"], "b1")

    def test_retrieve_graph_connections(self):
        mock_store = MagicMock()
        
        def mock_query(cypher, params=None):
            if "MATCH (f:Fact)" in cypher:
                return [
                    {"block_id": "b1", "statement": "Mitochondria produce ATP", "subject": "Mitochondria", "predicate": "produce", "object": "ATP", "reliability": "verified", "quality": 9}
                ]
            elif "MATCH (a:Fact)-[r:RELATED]" in cypher:
                return [
                    {"block_id": "b2", "statement": "ATP is the cell energy currency", "subject": "ATP", "predicate": "is", "object": "cell energy currency", "reliability": "verified", "quality": 8, "rel_predicate": "RELATED"}
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
            {"block_id": "b1", "statement": "Mitochondria produce ATP", "subject": "Mitochondria", "predicate": "produce", "object": "ATP", "reliability": "verified", "quality": 9}
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
            [{"src": "b1", "dst": "b2"}]
        ]
        
        retriever = GraphRAGRetriever(store=mock_store)
        scores = retriever.calculate_pagerank()
        self.assertIn("b1", scores)
        self.assertIn("b2", scores)
        self.assertTrue(scores["b2"] > 0)

    def test_multihop_traversal(self):
        mock_store = MagicMock()
        
        def mock_query(cypher, params=None):
            if "MATCH (f:Fact) RETURN f.block_id" in cypher:
                return [{"block_id": "b1"}, {"block_id": "b2"}, {"block_id": "b3"}]
            if "MATCH (a:Fact)-[r:RELATED]->(b:Fact)" in cypher:
                return [{"src": "b1", "dst": "b2"}, {"src": "b2", "dst": "b3"}]
            if "MATCH (f:Fact) " in cypher:
                return [{"block_id": "b1", "statement": "A relates to B", "subject": "A", "predicate": "relates to", "object": "B"}]
            if "MATCH (a:Fact)-[r:RELATED]-(b:Fact)" in cypher:
                if params and params.get("bid") == "b1":
                    return [{"block_id": "b2", "statement": "B relates to C", "subject": "B", "predicate": "relates to", "object": "C"}]
                if params and params.get("bid") == "b2":
                    return [{"block_id": "b3", "statement": "C relates to D", "subject": "C", "predicate": "relates to", "object": "D"}]
            return []
            
        mock_store.query.side_effect = mock_query
        retriever = GraphRAGRetriever(store=mock_store)
        results = retriever.retrieve("A", top_k=2, hops=2)
        bids = [r["block_id"] for r in results]
        self.assertIn("b1", bids)
        self.assertIn("b2", bids)
        self.assertIn("b3", bids)

