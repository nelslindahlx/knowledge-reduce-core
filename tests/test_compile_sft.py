"""Tests for compile-sft CLI subcommand using isolated temporary directories."""
import json
import os
import tempfile
import unittest

from knowledge_graph_pkg.cli import main
from knowledge_graph_pkg.store import Drop, KnowledgeStore, content_hash


class TestCompileSFTOptions(unittest.TestCase):

    def test_compile_sft_from_real_store_formats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_dir = os.path.join(tmpdir, "store")
            store = KnowledgeStore(store_dir)
            store.write_drop(
                Drop(
                    drop_id="d1",
                    source="doc.txt",
                    source_hash=content_hash("doc"),
                    facts=[
                        {"statement": "Mitochondria produce ATP.", "subject": "Mitochondria"},
                        {"statement": "ATP is energy.", "subject": "ATP"},
                    ],
                )
            )

            alpaca_out = os.path.join(tmpdir, "sft_alpaca.json")
            rc = main(["compile-sft", "-o", alpaca_out, "--store", store_dir, "--format", "alpaca"])
            self.assertEqual(rc, 0)
            data = json.loads(open(alpaca_out, encoding="utf-8").read())
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["instruction"], "State a verified fact about Mitochondria.")
            self.assertEqual(data[0]["output"], "Mitochondria produce ATP.")

            sharegpt_out = os.path.join(tmpdir, "sft_sharegpt.jsonl")
            rc2 = main(["compile-sft", "-o", sharegpt_out, "--store", store_dir, "--format", "sharegpt"])
            self.assertEqual(rc2, 0)
            lines = open(sharegpt_out, encoding="utf-8").read().splitlines()
            row = json.loads(lines[0])
            self.assertEqual(row["conversations"][0]["from"], "human")
            self.assertEqual(row["conversations"][1]["value"], "Mitochondria produce ATP.")

            plain_out = os.path.join(tmpdir, "sft_plain.jsonl")
            rc3 = main(["compile-sft", "-o", plain_out, "--store", store_dir, "--format", "plain"])
            self.assertEqual(rc3, 0)
            plain_row = json.loads(open(plain_out, encoding="utf-8").read().splitlines()[0])
            self.assertEqual(plain_row["text"], "Mitochondria produce ATP.")

    def test_compile_sft_missing_store_returns_2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            rc = main([
                "compile-sft",
                "-o", os.path.join(tmpdir, "out.json"),
                "--store", os.path.join(tmpdir, "missing_store"),
            ])
            self.assertEqual(rc, 2)

    def test_compile_sft_from_kuzu_store(self):
        import importlib.util
        if importlib.util.find_spec("kuzu") is None:
            self.skipTest("kuzu not installed")
        from knowledge_graph_pkg.kuzu_store import KuzuStore

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "kuzu_db")
            ks = KuzuStore(db_path)
            ks.ingest_facts([{
                "subject": "Marie Curie",
                "predicate": "discovered",
                "object": "radium",
                "fact_statement": "Marie Curie discovered radium.",
                "reliability_rating": "VERIFIED",
            }])
            ks.close()

            out_file = os.path.join(tmpdir, "kuzu_sft.json")
            rc = main(["compile-sft", "-o", out_file, "--graph-db", db_path, "--format", "alpaca"])
            self.assertEqual(rc, 0)
            items = json.loads(open(out_file, encoding="utf-8").read())
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["output"], "Marie Curie discovered radium.")
