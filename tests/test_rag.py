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

    def test_exclude_unverified_filtering(self):
        mock_store = MagicMock()
        def mock_query(cypher, params=None):
            if "MATCH (f:Fact)" in cypher and "f.statement" not in cypher:
                return [{"block_id": "b1"}, {"block_id": "b2"}]
            if "MATCH (a:Fact)-[r:RELATED]->(b:Fact)" in cypher:
                return [{"src": "b1", "dst": "b2"}]
            if "MATCH (f:Fact)" in cypher and "f.statement" in cypher:
                return [
                    {"block_id": "b1", "statement": "A relates to B", "subject": "A", "predicate": "relates to", "object": "B", "reliability": "VERIFIED"},
                    {"block_id": "b2", "statement": "B relates to C", "subject": "B", "predicate": "relates to", "object": "C", "reliability": "UNVERIFIED"}
                ]
            if "MATCH (a:Fact)-[r:RELATED]-(b:Fact)" in cypher:
                if "reliability <> 'UNVERIFIED'" in cypher:
                    return []
                return [{"block_id": "b2", "statement": "B relates to C", "subject": "B", "predicate": "relates to", "object": "C", "reliability": "UNVERIFIED"}]
            return []
            
        mock_store.query.side_effect = mock_query
        retriever = GraphRAGRetriever(store=mock_store)
        
        # When exclude_unverified=True, b2 (UNVERIFIED) should be filtered out
        results_filtered = retriever.retrieve("A", top_k=2, hops=1, exclude_unverified=True)
        bids_filtered = [r["block_id"] for r in results_filtered]
        self.assertIn("b1", bids_filtered)
        self.assertNotIn("b2", bids_filtered)

        # When exclude_unverified=False, b2 should be included
        results_all = retriever.retrieve("A", top_k=2, hops=1, exclude_unverified=False)
        bids_all = [r["block_id"] for r in results_all]
        self.assertIn("b1", bids_all)
        self.assertIn("b2", bids_all)

    def test_graph_rag_retrieve_tool(self):
        from knowledge_graph_pkg.graph_tool import GraphTools, TOOL_SCHEMAS
        
        mock_store = MagicMock()
        mock_store.count.return_value = 1
        mock_store.query.return_value = [
            {"block_id": "b1", "statement": "A relates to B", "subject": "A", "predicate": "relates to", "object": "B", "reliability": "VERIFIED"}
        ]
        
        tools = GraphTools(store=mock_store)
        results = tools.graph_rag_retrieve(query="A", top_k=1, hops=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["block_id"], "b1")
        
        # Verify schema is in TOOL_SCHEMAS
        tool_names = [t["name"] for t in TOOL_SCHEMAS]
        self.assertIn("graph_rag_retrieve", tool_names)

    def test_conflict_aware_pruning(self):
        mock_store = MagicMock()
        def mock_query(cypher, params=None):
            # Return active conflict between Mitochondria and ATP
            if "MATCH (a:Fact), (b:Fact)" in cypher:
                return [{"subject": "Mitochondria", "object": "ATP"}]
            if "MATCH (f:Fact)" in cypher and "f.statement" in cypher:
                return [
                    {"block_id": "b1", "statement": "Mitochondria produces ATP", "subject": "Mitochondria", "predicate": "produces", "object": "ATP", "reliability": "VERIFIED"}
                ]
            if "MATCH (a:Fact)-[r:RELATED]-(b:Fact)" in cypher:
                return [
                    {"block_id": "b2", "statement": "Mitochondria produces ATP", "subject": "Mitochondria", "predicate": "produces", "object": "ATP", "reliability": "VERIFIED"}
                ]
            return []
            
        mock_store.query.side_effect = mock_query
        retriever = GraphRAGRetriever(store=mock_store)
        
        # Walk should prune b2 because Mitochondria -> ATP is in active conflict
        results = retriever.retrieve("Mitochondria", top_k=2, hops=1)
        bids = [r["block_id"] for r in results]
        self.assertIn("b1", bids)
        self.assertNotIn("b2", bids)

    def test_graph_distill_ontology_tool(self):
        from knowledge_graph_pkg.graph_tool import GraphTools, TOOL_SCHEMAS
        
        mock_store = MagicMock()
        mock_store.query.return_value = [
            {"subject": "ATP", "predicate": "is a type of", "object": "ChemicalCompound",
             "child": "ATP", "parent": "ChemicalCompound"}
        ]
        
        tools = GraphTools(store=mock_store)
        res = tools.graph_distill_ontology()
        
        self.assertIn("taxonomy", res)
        self.assertIn("ChemicalCompound", res["taxonomy"])
        self.assertIn("ATP", res["taxonomy"]["ChemicalCompound"])
        
        # Verify schema is in TOOL_SCHEMAS
        tool_names = [t["name"] for t in TOOL_SCHEMAS]
        self.assertIn("graph_distill_ontology", tool_names)


