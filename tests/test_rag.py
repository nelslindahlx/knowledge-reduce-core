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
