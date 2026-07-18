import unittest
from unittest.mock import MagicMock, patch
from knowledge_graph_pkg.consensus import ConsensusEngine

class TestConsensusEngine(unittest.TestCase):

    @patch('knowledge_graph_pkg.consensus.batch_drop')
    @patch('knowledge_graph_pkg.consensus.KnowledgeStore')
    @patch('knowledge_graph_pkg.consensus.ModelKnowledgeDistiller')
    @patch('knowledge_graph_pkg.consensus.KuzuStore')
    def test_process_with_consensus(self, mock_kuzu_store_cls, mock_distiller_cls, mock_store_cls, mock_batch_drop):
        mock_kstore = MagicMock()
        mock_kstore.validate_and_reconcile.return_value = {
            "demoted": [{"block_id": "b1", "statement": "Conflicting statement"}]
        }
        mock_kuzu_store_cls.return_value = mock_kstore
        
        mock_batch_drop.return_value = {"dropped": 1, "errors": 0}
        
        engine = ConsensusEngine(store_dir="mock_store", graph_db_path="mock_graph_db")
        result = engine.process_with_consensus(
            file_path="dummy.txt",
            engines=["svo", "spacy"]
        )
        
        self.assertEqual(mock_batch_drop.call_count, 2)
        mock_kstore.validate_and_reconcile.assert_called_once()
        self.assertEqual(len(result["demoted"]), 1)
        self.assertEqual(result["demoted"][0]["block_id"], "b1")
