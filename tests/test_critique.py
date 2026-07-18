import unittest
from unittest.mock import MagicMock, patch
from knowledge_graph_pkg.critique import FactCritic

class TestFactCritic(unittest.TestCase):

    @patch('knowledge_graph_pkg.critique.get_backend')
    def test_critique_fact_positive(self, mock_make_backend):
        mock_backend = MagicMock()
        mock_backend.generate_text.return_value = '{"reasoning": "This is accurate.", "is_factual": true, "confidence_score": 0.95}'
        mock_backend.generate_structured_json.return_value = '{"reasoning": "This is accurate.", "is_factual": true, "confidence_score": 0.95}'
        mock_make_backend.return_value = mock_backend

        critic = FactCritic("openai", "gpt-4o")
        fact = {"fact_statement": "Paris is the capital of France.", "fact_id": "f1"}
        report = critic.critique_fact(fact)

        self.assertEqual(report["block_id"], "f1")
        self.assertTrue(report["is_factual"])
        self.assertEqual(report["confidence_score"], 0.95)
        self.assertEqual(report["reasoning"], "This is accurate.")

    @patch('knowledge_graph_pkg.critique.get_backend')
    def test_critique_fact_negative(self, mock_make_backend):
        mock_backend = MagicMock()
        # Mock backend returning markdown code blocks (standard LLM output)
        mock_backend.generate_text.return_value = '```json\n{"reasoning": "This is false. Paris is not in Germany.", "is_factual": false, "confidence_score": 0.99}\n```'
        mock_backend.generate_structured_json.return_value = '{"reasoning": "This is false. Paris is not in Germany.", "is_factual": false, "confidence_score": 0.99}'
        mock_make_backend.return_value = mock_backend

        critic = FactCritic("openai", "gpt-4o")
        fact = {"fact_statement": "Paris is in Germany.", "fact_id": "f2"}
        report = critic.critique_fact(fact)

        self.assertEqual(report["block_id"], "f2")
        self.assertFalse(report["is_factual"])
        self.assertEqual(report["confidence_score"], 0.99)
        self.assertEqual(report["reasoning"], "This is false. Paris is not in Germany.")

    @patch('knowledge_graph_pkg.critique.get_backend')
    def test_critique_facts_batch(self, mock_make_backend):
        mock_backend = MagicMock()
        mock_backend.generate_text.return_value = '{"reasoning": "Ok.", "is_factual": true, "confidence_score": 1.0}'
        mock_backend.generate_structured_json.return_value = '{"reasoning": "Ok.", "is_factual": true, "confidence_score": 1.0}'
        mock_make_backend.return_value = mock_backend

        critic = FactCritic("openai", "gpt-4o")
        facts = [
            {"fact_statement": "Water boils at 100C.", "fact_id": "w1"},
            {"fact_statement": "Gravity pulls down.", "fact_id": "g1"}
        ]
        reports = critic.critique_facts(facts)

        self.assertEqual(len(reports), 2)
        self.assertTrue(reports[0]["is_factual"])
        self.assertTrue(reports[1]["is_factual"])
