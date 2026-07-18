import os
import json
import unittest
from unittest.mock import MagicMock, patch
from knowledge_graph_pkg.cli import main

class TestCompileSFTOptions(unittest.TestCase):

    @patch('knowledge_graph_pkg.store.KnowledgeStore')
    def test_compile_sft_from_store_alpaca(self, mock_store_cls):
        # We handle tmp_path manually or use a standard path
        out_file = "test_sft_dataset.json"
        if os.path.exists(out_file):
            os.remove(out_file)

        mock_store = MagicMock()
        mock_store.iter_facts.return_value = [
            {"statement": "Mitochondria produce ATP.", "subject": "Mitochondria"},
            {"statement": "ATP is energy.", "subject": "ATP"}
        ]
        mock_store_cls.return_value = mock_store

        try:
            rc = main([
                "compile-sft",
                "-o", out_file,
                "--store", "dummy_store",
                "--format", "alpaca"
            ])
            
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_file))
            
            with open(out_file, "r") as fh:
                data = json.load(fh)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["instruction"], "State a verified fact about Mitochondria.")
            self.assertEqual(data[0]["output"], "Mitochondria produce ATP.")
        finally:
            if os.path.exists(out_file):
                os.remove(out_file)

    @patch('knowledge_graph_pkg.store.KnowledgeStore')
    def test_compile_sft_from_store_sharegpt(self, mock_store_cls):
        out_file = "test_sft_dataset.jsonl"
        if os.path.exists(out_file):
            os.remove(out_file)

        mock_store = MagicMock()
        mock_store.iter_facts.return_value = [
            {"statement": "Mitochondria produce ATP.", "subject": "Mitochondria"}
        ]
        mock_store_cls.return_value = mock_store

        try:
            rc = main([
                "compile-sft",
                "-o", out_file,
                "--store", "dummy_store",
                "--format", "sharegpt"
            ])
            
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out_file))
            
            with open(out_file, "r") as fh:
                line = fh.readline()
                data = json.loads(line)
                
            self.assertEqual(len(data["conversations"]), 2)
            self.assertEqual(data["conversations"][0]["from"], "human")
            self.assertEqual(data["conversations"][1]["from"], "gpt")
            self.assertEqual(data["conversations"][1]["value"], "Mitochondria produce ATP.")
        finally:
            if os.path.exists(out_file):
                os.remove(out_file)
