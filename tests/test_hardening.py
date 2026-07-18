import sys
from unittest.mock import MagicMock, patch

import unittest
from knowledge_graph_pkg.rag import GraphRAGRetriever

# Setup dummy modules before class definition
mock_openai = MagicMock()
mock_genai = MagicMock()

from knowledge_graph_pkg.model_probe import OpenAICompatibleBackend, GeminiBackend

class TestSystemHardening(unittest.TestCase):

    def setUp(self):
        self.sys_modules_patcher = patch.dict(sys.modules, {
            'openai': mock_openai,
            'google.generativeai': mock_genai
        })
        self.sys_modules_patcher.start()

    def tearDown(self):
        self.sys_modules_patcher.stop()

    @patch('knowledge_graph_pkg.rag.get_embedder')
    def test_pagerank_caching(self, mock_get_embedder):
        mock_store = MagicMock()
        # Mock count returning same value
        mock_store.count.return_value = 5
        mock_store.query.side_effect = [
            [{"block_id": "f1"}, {"block_id": "f2"}],  # facts query
            [{"src": "f1", "dst": "f2"}]  # edges query
        ]

        retriever = GraphRAGRetriever(store=mock_store)
        
        # First call calculates Page-Rank
        scores1 = retriever.calculate_pagerank()
        self.assertIn("f1", scores1)
        self.assertEqual(mock_store.query.call_count, 2)

        # Second call returns cached dictionary directly without executing DB queries
        scores2 = retriever.calculate_pagerank()
        self.assertEqual(scores1, scores2)
        self.assertEqual(mock_store.query.call_count, 2)

        # Update count to trigger recalculation
        mock_store.count.return_value = 6
        mock_store.query.side_effect = [
            [{"block_id": "f1"}, {"block_id": "f2"}, {"block_id": "f3"}],
            [{"src": "f1", "dst": "f2"}]
        ]
        
        scores3 = retriever.calculate_pagerank()
        self.assertEqual(mock_store.query.call_count, 4)

    @patch.dict('os.environ', {}, clear=True)
    def test_openai_api_key_missing(self):
        with self.assertRaises(ValueError) as context:
            OpenAICompatibleBackend(model="gpt-4")
        self.assertIn("OpenAI API key not found", str(context.exception))

    @patch.dict('os.environ', {}, clear=True)
    def test_gemini_api_key_missing(self):
        with self.assertRaises(ValueError) as context:
            GeminiBackend(model="gemini-1.5-flash")
        self.assertIn("Gemini API key not found", str(context.exception))
