import importlib.util
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from knowledge_graph_pkg.consensus import ConsensusEngine

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None


class TestConsensusEngine(unittest.TestCase):

    @patch("knowledge_graph_pkg.consensus.batch_drop")
    @patch("knowledge_graph_pkg.consensus.ModelKnowledgeDistiller")
    @patch("knowledge_graph_pkg.consensus.get_graph_store")
    def test_process_with_consensus(self, mock_get_store, mock_distiller_cls, mock_batch_drop):
        mock_kstore = MagicMock()
        mock_kstore.validate_and_reconcile.return_value = {
            "demoted": [{"block_id": "b1", "statement": "Conflicting statement"}]
        }
        mock_get_store.return_value = mock_kstore

        mock_batch_drop.return_value = {
            "dropped": 1,
            "errors": 0,
            "items": [
                {
                    "status": "dropped",
                    "fact_records": [
                        {
                            "subject": "Mitochondria",
                            "predicate": "produce",
                            "object": "ATP",
                            "fact_statement": "Mitochondria produce ATP.",
                        }
                    ],
                }
            ],
        }

        engine = ConsensusEngine(store_dir="mock_store", graph_db_path="mock_graph_db")
        result = engine.process_with_consensus(
            file_path="dummy.txt",
            engines=["svo", "spacy"],
        )

        self.assertEqual(mock_batch_drop.call_count, 2)
        for call in mock_batch_drop.call_args_list:
            self.assertTrue(call.kwargs.get("include_engine_in_hash"))
        passed_facts = mock_distiller_cls.call_args[0][0]
        models_seen = {f["model_provenance"]["model"] for f in passed_facts}
        self.assertEqual(models_seen, {"svo", "spacy"})
        mock_kstore.validate_and_reconcile.assert_called_once()
        mock_kstore.close.assert_called_once()
        self.assertEqual(len(result["demoted"]), 1)
        self.assertEqual(result["demoted"][0]["block_id"], "b1")


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu not installed")
class TestConsensusEngineIntegration(unittest.TestCase):

    def test_multi_engine_does_not_skip_second_engine_and_ingests_kuzu(self):
        with tempfile.TemporaryDirectory() as tmp:
            doc = os.path.join(tmp, "bio.txt")
            with open(doc, "w", encoding="utf-8") as fh:
                fh.write("Mitochondria produce ATP. Cells use ATP.")
            store_dir = os.path.join(tmp, "store")
            db_path = os.path.join(tmp, "kuzu_db")

            engine = ConsensusEngine(store_dir=store_dir, graph_db_path=db_path)
            result = engine.process_with_consensus(file_path=doc, engines=["svo", "svo"])
            # Distinct engine names both run without hash collision
            engine2 = ConsensusEngine(store_dir=store_dir, graph_db_path=db_path)
            res2 = engine2.process_with_consensus(file_path=doc, engines=["svo"])
            self.assertIn("svo", result["reports"])
            self.assertEqual(res2["reports"]["svo"]["skipped"], 1)

            # Verify multi-engine with distinct mocked second engine skips neither
            fake_engine = MagicMock()
            fake_engine.extract_triples.return_value = [
                ("Mitochondria", "produce", "ATP"),
            ]
            with patch("knowledge_graph_pkg.extractor_base.get_extractor", return_value=fake_engine):
                store2_dir = os.path.join(tmp, "store2")
                db2_path = os.path.join(tmp, "kuzu_db2")
                eng = ConsensusEngine(store_dir=store2_dir, graph_db_path=db2_path)
                res_multi = eng.process_with_consensus(file_path=doc, engines=["svo", "spacy"])
                self.assertEqual(res_multi["reports"]["svo"]["dropped"], 1)
                self.assertEqual(res_multi["reports"]["spacy"]["dropped"], 1)

            from knowledge_graph_pkg.kuzu_store import KuzuStore

            ks = KuzuStore(db2_path)
            try:
                rows = ks.query("MATCH (f:Fact) RETURN f.subject AS s, f.source_models AS m")
                self.assertGreater(len(rows), 0)
                models = ", ".join(r["m"] for r in rows)
                self.assertIn("svo", models)
                self.assertNotIn(doc, models)
            finally:
                ks.close()
