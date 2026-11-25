import unittest
from unittest.mock import MagicMock, patch

from knowledge_graph_pkg.cross_model import CrossModelVerifier, reliability_for_agreement


class TestWeightedConsensus(unittest.TestCase):

    def test_weighted_consensus_reliability_ratings(self):
        self.assertEqual(reliability_for_agreement(1, ["gpt-4"]), "VERIFIED")
        self.assertEqual(reliability_for_agreement(1, ["gpt-4o"]), "VERIFIED")
        self.assertEqual(reliability_for_agreement(1, ["gpt-4o-mini"]), "LIKELY_TRUE")
        self.assertEqual(reliability_for_agreement(1, ["qwen2.5:14b"]), "LIKELY_TRUE")
        self.assertEqual(
            reliability_for_agreement(2, ["qwen2.5-0.5b", "qwen2.5-0.5b"]),
            "POSSIBLY_TRUE",
        )
        self.assertEqual(reliability_for_agreement(1, ["llama3-8b"]), "LIKELY_TRUE")
        self.assertEqual(
            reliability_for_agreement(2, ["unknown-model-A", "unknown-model-B"]),
            "LIKELY_TRUE",
        )
        # Non-matching substring must not get capability boost
        self.assertEqual(reliability_for_agreement(1, ["pro"]), "POSSIBLY_TRUE")

    def test_verifier_weighted_verify(self):
        mock_probe = MagicMock()
        mock_probe.model = "gpt-4"

        verifier = CrossModelVerifier([mock_probe])

        outputs_by_model = {
            "gpt-4": [
                {
                    "prompt": "prompt1",
                    "model": "gpt-4",
                    "structured_response": {
                        "facts": [
                            {
                                "subject": "Mitochondria",
                                "predicate": "make",
                                "object": "ATP",
                                "statement": "Mitochondria make ATP.",
                            }
                        ]
                    },
                }
            ],
            "qwen2.5-0.5b": [
                {
                    "prompt": "prompt1",
                    "model": "qwen2.5-0.5b",
                    "structured_response": {
                        "facts": [
                            {
                                "subject": "Mitochondria",
                                "predicate": "make",
                                "object": "ATP",
                                "statement": "Mitochondria make ATP.",
                            }
                        ]
                    },
                }
            ],
        }

        report = verifier.verify(outputs_by_model)

        self.assertEqual(report["n_clusters"], 1)
        cluster = report["clusters"][0]
        self.assertEqual(cluster["reliability"], "VERIFIED")
        self.assertEqual(cluster["cross_model_agreement"], 2)

    def test_graph_store_factory_parsing(self):
        with patch("knowledge_graph_pkg.neo4j_store.Neo4jStore") as mock_neo4j:
            from knowledge_graph_pkg.graph_store_factory import get_graph_store

            get_graph_store("bolt://user1:pass2@localhost:7687?database=db1")
            mock_neo4j.assert_called_with(
                "bolt://localhost:7687", user="user1", password="pass2", database="db1"
            )

            get_graph_store("neo4j://user3:pass4@host:9999/db2")
            mock_neo4j.assert_called_with(
                "neo4j://host:9999", user="user3", password="pass4", database="db2"
            )
