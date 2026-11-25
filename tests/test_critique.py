import unittest
from unittest.mock import MagicMock, patch

from knowledge_graph_pkg.critique import FactCritic
from knowledge_graph_pkg.model_probe import OpenAICompatibleBackend


class TestFactCritic(unittest.TestCase):

    @patch("knowledge_graph_pkg.critique.get_backend")
    def test_critique_fact_positive_structured_dict(self, mock_make_backend):
        mock_backend = MagicMock(spec=OpenAICompatibleBackend)
        mock_backend.generate_structured.return_value = {
            "reasoning": "This is accurate.",
            "is_factual": True,
            "confidence_score": 0.95,
        }
        mock_make_backend.return_value = mock_backend

        critic = FactCritic("openai", "gpt-4o")
        fact = {"fact_statement": "Paris is the capital of France.", "fact_id": "f1"}
        report = critic.critique_fact(fact)

        self.assertEqual(report["block_id"], "f1")
        self.assertTrue(report["is_factual"])
        self.assertEqual(report["confidence_score"], 0.95)
        self.assertEqual(report["reasoning"], "This is accurate.")

    @patch("knowledge_graph_pkg.critique.get_backend")
    def test_critique_fact_negative_string_false_and_logprob_fallback(self, mock_make_backend):
        class LogprobOnlyBackend:
            def generate_text_with_logprobs(self, prompt):
                return (
                    '```json\n{"reasoning": "Paris is not in Germany.", '
                    '"is_factual": "false", "confidence_score": 0.99}\n```',
                    -0.1,
                )

        mock_make_backend.return_value = LogprobOnlyBackend()

        critic = FactCritic("openai", "gpt-4o")
        fact = {"fact_statement": "Paris is in Germany.", "fact_id": "f2"}
        report = critic.critique_fact(fact)

        self.assertEqual(report["block_id"], "f2")
        self.assertFalse(report["is_factual"])
        self.assertEqual(report["confidence_score"], 0.99)
        self.assertEqual(report["reasoning"], "Paris is not in Germany.")

    @patch("knowledge_graph_pkg.critique.get_backend")
    def test_critique_fails_closed_on_malformed_json(self, mock_make_backend):
        class BrokenBackend:
            def generate_text_with_logprobs(self, prompt):
                return ("not valid json", 0.0)

        mock_make_backend.return_value = BrokenBackend()
        critic = FactCritic("openai", "gpt-4o")
        report = critic.critique_fact({"fact_statement": "Water boils.", "fact_id": "bad"})
        self.assertFalse(report["is_factual"])
        self.assertEqual(report["confidence_score"], 0.0)
        self.assertTrue(report.get("error"))

    @patch("knowledge_graph_pkg.critique.get_backend")
    def test_critique_facts_batch(self, mock_make_backend):
        mock_backend = MagicMock(spec=OpenAICompatibleBackend)
        mock_backend.generate_structured.return_value = {
            "reasoning": "Ok.",
            "is_factual": True,
            "confidence_score": 1.0,
        }
        mock_make_backend.return_value = mock_backend

        critic = FactCritic("openai", "gpt-4o")
        facts = [
            {"fact_statement": "Water boils at 100C.", "fact_id": "w1"},
            {"fact_statement": "Gravity pulls down.", "fact_id": "g1"},
        ]
        reports = critic.critique_facts(facts)

        self.assertEqual(len(reports), 2)
        self.assertTrue(reports[0]["is_factual"])
        self.assertTrue(reports[1]["is_factual"])

    def test_invalid_backend_or_missing_api_key_raises(self):
        with self.assertRaises(ValueError):
            FactCritic("nonexistent-backend", "m")
        with patch.dict("os.environ", {}, clear=True):
            with patch.dict("sys.modules", {"openai": MagicMock()}):
                with self.assertRaises(ValueError):
                    FactCritic("openai", "gpt-4o")

    def test_heuristic_critique_pronoun_subject(self):
        critic = FactCritic("none")
        fact = {
            "subject": "it",
            "predicate": "heats",
            "object": "water",
            "statement": "it heats water.",
            "fact_id": "f_pronoun",
        }
        report = critic.critique_fact(fact)
        self.assertFalse(report["is_factual"])
        self.assertIn("Pronoun subject", report["reasoning"])

    def test_heuristic_critique_pronoun_object(self):
        critic = FactCritic("none")
        fact = {
            "subject": "Fire",
            "predicate": "heats",
            "object": "them",
            "statement": "Fire heats them.",
            "fact_id": "f_pronoun_obj",
        }
        report = critic.critique_fact(fact)
        self.assertFalse(report["is_factual"])
        self.assertIn("Pronoun object", report["reasoning"])

    def test_heuristic_critique_stub_or_empty(self):
        critic = FactCritic("none")
        fact_empty = {
            "subject": "",
            "predicate": "heats",
            "object": "water",
            "statement": "heats water.",
            "fact_id": "f_empty",
        }
        report_empty = critic.critique_fact(fact_empty)
        self.assertFalse(report_empty["is_factual"])
        self.assertIn("too short or has empty SVO", report_empty["reasoning"])

    def test_heuristic_critique_tautology_subj_equals_obj(self):
        critic = FactCritic("none")
        fact = {
            "subject": "Water",
            "predicate": "is",
            "object": "water",
            "statement": "Water is water.",
            "fact_id": "f_tautology",
        }
        report = critic.critique_fact(fact)
        self.assertFalse(report["is_factual"])
        self.assertIn("Redundant SVO components", report["reasoning"])

    def test_heuristic_critique_identical_components(self):
        critic = FactCritic("none")
        fact = {
            "subject": "water",
            "predicate": "water",
            "object": "water",
            "statement": "water water water.",
            "fact_id": "f_redundant",
        }
        report = critic.critique_fact(fact)
        self.assertFalse(report["is_factual"])
        self.assertIn("Redundant SVO components", report["reasoning"])

    def test_heuristic_critique_valid(self):
        critic = FactCritic("none")
        fact = {
            "subject": "Fire",
            "predicate": "heats",
            "object": "water",
            "statement": "Fire heats water.",
            "fact_id": "f_valid",
        }
        report = critic.critique_fact(fact)
        self.assertTrue(report["is_factual"])
        self.assertIn("Passed offline heuristic validations", report["reasoning"])
