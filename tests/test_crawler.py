import unittest
from unittest.mock import MagicMock, patch
from knowledge_graph_pkg.crawler import ModelCrawler
from knowledge_graph_pkg.cli import main

class TestModelCrawler(unittest.TestCase):

    def test_crawl_recursive_flow(self):
        mock_backend = MagicMock()
        
        responses = {
            "Provide a concise list of factual assertions about 'AI'.": ("AI leverages Machine Learning. AI builds Neural Networks.", 0.0),
            "Provide a concise list of factual assertions about 'Machine Learning'.": ("Machine Learning processes Data.", 0.0),
            "Provide a concise list of factual assertions about 'Neural Networks'.": ("Neural Networks mimic Neurons.", 0.0),
            "Provide a concise list of factual assertions about 'Data'.": ("Data is informative.", 0.0),
            "Provide a concise list of factual assertions about 'Neurons'.": ("Neurons carry signals.", 0.0),
        }
        
        def mock_generate(prompt, **kwargs):
            return responses.get(prompt, ("", 0.0))
            
        mock_backend.generate_text_with_logprobs.side_effect = mock_generate
        mock_backend.model = "crawler-model"
        
        crawler = ModelCrawler(backend=mock_backend)
        facts = crawler.crawl(seed_topic="AI", max_depth=1, concepts_per_level=2)
        
        self.assertTrue(len(facts) > 0)
        
        topics_crawled = [call[0][0] for call in mock_backend.generate_text_with_logprobs.call_args_list]
        self.assertIn("Provide a concise list of factual assertions about 'AI'.", topics_crawled)
        self.assertTrue(any("Machine Learning" in t for t in topics_crawled))
        self.assertTrue(any("Neural Networks" in t for t in topics_crawled))

    def test_crawl_entropy_threshold_filtering(self):
        mock_backend = MagicMock()
        
        mock_backend.generate_text_with_logprobs.return_value = ("AI is magical.", -3.0)
        mock_backend.model = "entropy-model"
        
        crawler = ModelCrawler(backend=mock_backend)
        facts = crawler.crawl(seed_topic="AI", max_depth=1, logprob_threshold=-1.5)
        
        self.assertEqual(len(facts), 0)

    @patch('knowledge_graph_pkg.model_probe.get_backend')
    @patch('knowledge_graph_pkg.crawler.ModelCrawler')
    @patch('knowledge_graph_pkg.store.KnowledgeStore')
    def test_cli_crawl_routing(self, mock_store_cls, mock_crawler_cls, mock_get_backend):
        mock_crawler_inst = MagicMock()
        mock_crawler_inst.crawl.return_value = [
            {"block_id": "b1", "statement": "AI leverages ML", "subject": "AI", "predicate": "leverages", "object": "ML"}
        ]
        mock_crawler_cls.return_value = mock_crawler_inst
        
        with patch("sys.argv", ["knowledgereduce", "crawl", "--seed", "AI", "--model", "dummy_model", "--backend", "openai", "--store", "dummy_store"]):
            code = main()
            self.assertEqual(code, 0)
            mock_crawler_inst.crawl.assert_called_once()
