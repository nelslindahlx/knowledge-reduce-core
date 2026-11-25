import sys
import unittest
from unittest.mock import MagicMock, patch

from knowledge_graph_pkg.model_probe import GeminiBackend, OpenAICompatibleBackend
from knowledge_graph_pkg.rag import GraphRAGRetriever


class TestSystemHardening(unittest.TestCase):

    def setUp(self):
        self.sys_modules_patcher = patch.dict(
            sys.modules,
            {
                "openai": MagicMock(),
                "google": MagicMock(),
                "google.generativeai": MagicMock(),
            },
        )
        self.sys_modules_patcher.start()

    def tearDown(self):
        self.sys_modules_patcher.stop()

    @patch("knowledge_graph_pkg.rag.get_embedder")
    def test_pagerank_caching(self, mock_get_embedder):
        mock_store = MagicMock()
        mock_store.count.return_value = 5
        mock_store.query.side_effect = [
            [{"block_id": "f1"}, {"block_id": "f2"}],
            [{"src": "f1", "dst": "f2"}],
        ]

        retriever = GraphRAGRetriever(store=mock_store)

        scores1 = retriever.calculate_pagerank()
        self.assertIn("f1", scores1)
        self.assertEqual(mock_store.query.call_count, 2)

        scores2 = retriever.calculate_pagerank()
        self.assertEqual(scores1, scores2)
        self.assertEqual(mock_store.query.call_count, 2)

        mock_store.count.return_value = 6
        mock_store.query.side_effect = [
            [{"block_id": "f1"}, {"block_id": "f2"}, {"block_id": "f3"}],
            [{"src": "f1", "dst": "f2"}],
        ]

        retriever.calculate_pagerank()
        self.assertEqual(mock_store.query.call_count, 4)

    @patch.dict("os.environ", {}, clear=True)
    def test_openai_api_key_missing(self):
        with self.assertRaises(ValueError) as context:
            OpenAICompatibleBackend(model="gpt-4")
        self.assertIn("OpenAI API key not found", str(context.exception))

    @patch.dict("os.environ", {}, clear=True)
    def test_gemini_api_key_missing(self):
        with self.assertRaises(ValueError) as context:
            GeminiBackend(model="gemini-2.0-flash")
        self.assertIn("Gemini API key not found", str(context.exception))
