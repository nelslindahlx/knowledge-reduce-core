import os
import json
import tempfile
import unittest
from unittest.mock import patch
from knowledge_graph_pkg.store import KnowledgeStore, Drop
from knowledge_graph_pkg.cli import main

class TestStoreAudit(unittest.TestCase):

    def test_audit_summary_calculates_correct_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            
            # 1. Add first drop: valid fact
            drop1 = Drop(
                drop_id="drop_001",
                source="doc1.txt",
                source_hash="hash1",
                facts=[
                    {
                        "subject": "Alpha",
                        "predicate": "activates",
                        "object": "Beta",
                        "statement": "Alpha activates Beta.",
                        "reliability": "LIKELY_TRUE"
                    }
                ]
            )
            store.write_drop(drop1)
            
            # 2. Add second drop: contains a duplicate SVO and a missing field SVO
            drop2 = Drop(
                drop_id="drop_002",
                source="doc2.txt",
                source_hash="hash2",
                facts=[
                    # Duplicate
                    {
                        "subject": "Alpha",
                        "predicate": "activates",
                        "object": "Beta",
                        "statement": "Alpha activates Beta.",
                        "reliability": "LIKELY_TRUE"
                    },
                    # Missing subject/object/statement/predicate
                    {
                        "subject": "",
                        "predicate": "inhibits",
                        "object": "Gamma",
                        "statement": "inhibits Gamma.",
                        "reliability": "UNVERIFIED"
                    }
                ]
            )
            store.write_drop(drop2)
            
            report = store.audit_summary()
            
            self.assertEqual(report["total_drops"], 2)
            self.assertEqual(report["total_facts"], 3)
            self.assertEqual(report["missing_fields_count"], 1)
            self.assertEqual(report["duplicate_svo_count"], 1)
            self.assertEqual(report["reliability_tier_distribution"]["LIKELY_TRUE"], 2)
            self.assertEqual(report["reliability_tier_distribution"]["UNVERIFIED"], 1)

    def test_audit_cli_command(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = KnowledgeStore(tmpdir)
            drop1 = Drop(
                drop_id="drop_001",
                source="doc1.txt",
                source_hash="hash1",
                facts=[
                    {
                        "subject": "Alpha",
                        "predicate": "activates",
                        "object": "Beta",
                        "statement": "Alpha activates Beta.",
                        "reliability": "VERIFIED"
                    }
                ]
            )
            store.write_drop(drop1)
            
            with patch("sys.stdout") as mock_stdout:
                exit_code = main(["audit-store", "--store", tmpdir])
                self.assertEqual(exit_code, 0)
