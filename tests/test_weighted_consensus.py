import unittest
from knowledge_graph_pkg.cross_model import (
    reliability_for_agreement,
    CrossModelVerifier,
    MODEL_CAPABILITY_WEIGHTS
)

class TestWeightedConsensus(unittest.TestCase):

    def test_weighted_consensus_reliability_ratings(self):
        # 1. Single high-capability model (weight 3.0 >= 3.0) -> VERIFIED
        self.assertEqual(reliability_for_agreement(1, ["gpt-4"]), "VERIFIED")
        
        # 2. Two low-capability models (total weight 0.5 + 0.5 = 1.0 < 1.5) -> POSSIBLY_TRUE
        self.assertEqual(reliability_for_agreement(2, ["qwen2.5-0.5b", "qwen2.5-0.5b"]), "POSSIBLY_TRUE")
        
        # 3. One medium model (weight 1.5) -> LIKELY_TRUE
        self.assertEqual(reliability_for_agreement(1, ["llama3-8b"]), "LIKELY_TRUE")
        
        # 4. Unknown model (default weight 1.0) -> LIKELY_TRUE if combined with another unknown model
        self.assertEqual(reliability_for_agreement(2, ["unknown-model-A", "unknown-model-B"]), "LIKELY_TRUE")

    def test_verifier_weighted_verify(self):
        # Mock probes and verifier setup
        from unittest.mock import MagicMock
        mock_probe = MagicMock()
        mock_probe.model = "gpt-4"
    
        verifier = CrossModelVerifier([mock_probe])
    
        # Set up outputs representing 2 distinct models (one high weight, one low weight)
        outputs_by_model = {
            "gpt-4": [
                {
                    "prompt": "prompt1",
                    "model": "gpt-4",
                    "structured_response": {"facts": [{"subject": "Mitochondria", "predicate": "make", "object": "ATP", "statement": "Mitochondria make ATP."}]}
                }
            ],
            "qwen2.5-0.5b": [
                {
                    "prompt": "prompt1",
                    "model": "qwen2.5-0.5b",
                    "structured_response": {"facts": [{"subject": "Mitochondria", "predicate": "make", "object": "ATP", "statement": "Mitochondria make ATP."}]}
                }
            ]
        }
    
        report = verifier.verify(outputs_by_model)
    
        # Verify cluster is promoted to VERIFIED because gpt-4 + qwen2.5-0.5b weight is 3.5 >= 3.0
        self.assertEqual(report["n_clusters"], 1)
        cluster = report["clusters"][0]
        self.assertEqual(cluster["reliability"], "VERIFIED")
        self.assertEqual(cluster["cross_model_agreement"], 2)

    def test_graph_store_factory_parsing(self):
        from unittest.mock import patch
        with patch("knowledge_graph_pkg.neo4j_store.Neo4jStore") as mock_neo4j:
            from knowledge_graph_pkg.graph_store_factory import get_graph_store
            
            # 1. Parse database name from query string parameter
            get_graph_store("bolt://user1:pass2@localhost:7687?database=db1")
            mock_neo4j.assert_called_with("bolt://user1:pass2@localhost:7687", user="user1", password="pass2", database="db1")
            
            # 2. Parse database name from path segment
            get_graph_store("neo4j://user3:pass4@host:9999/db2")
            mock_neo4j.assert_called_with("neo4j://user3:pass4@host:9999", user="user3", password="pass4", database="db2")

