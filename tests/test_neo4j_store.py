import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.neo4j_store import Neo4jStore


class TestNeo4jStore(unittest.TestCase):
    def setUp(self):
        self.mock_neo4j = MagicMock()
        self.mock_driver = MagicMock()
        self.mock_neo4j.GraphDatabase.driver.return_value = self.mock_driver
        self.mock_session = MagicMock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session
        self.sys_modules_patcher = patch.dict(sys.modules, {"neo4j": self.mock_neo4j})
        self.sys_modules_patcher.start()

    def tearDown(self):
        self.sys_modules_patcher.stop()

    def test_factory_routing_kuzu(self):
        pytest.importorskip("kuzu")
        with tempfile.TemporaryDirectory() as tmpdir:
            store = get_graph_store(os.path.join(tmpdir, "kdb"))
            self.assertEqual(store.__class__.__name__, "KuzuStore")
            store.close()

    def test_factory_routing_neo4j_and_tls_uri_cleaning(self):
        with patch.object(Neo4jStore, "_init_constraints", return_value=None):
            with patch.dict(os.environ, {"NEO4J_PASSWORD": "env-secret-password"}):
                store_neo = get_graph_store("neo4j://localhost:7687")
                self.assertEqual(store_neo.__class__.__name__, "Neo4jStore")
                self.assertEqual(store_neo._uri, "neo4j://localhost:7687")

            store_tls = get_graph_store("bolt+s://alice:p%40ss%2F1@db.example.com:7687/tenant_db")
            self.assertEqual(store_tls.__class__.__name__, "Neo4jStore")
            self.assertEqual(store_tls._uri, "bolt+s://db.example.com:7687")
            self.assertEqual(store_tls._database, "tenant_db")
            self.mock_neo4j.GraphDatabase.driver.assert_called_with(
                "bolt+s://db.example.com:7687",
                auth=("alice", "p@ss/1"),
            )

    def test_missing_password_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                Neo4jStore("neo4j://localhost:7687")

    def test_with_database_preserves_driver_without_password_leak(self):
        with patch.object(Neo4jStore, "_init_constraints", return_value=None):
            store = Neo4jStore("neo4j://localhost:7687", user="alice", password="pw", database="db1")
            ws_store = store.with_database("ws2")
            self.assertIs(ws_store.driver, store.driver)
            self.assertEqual(ws_store._database, "ws2")
            self.assertEqual(ws_store._uri, "neo4j://localhost:7687")
            self.assertFalse(hasattr(store, "_password"))

    def test_query_mapping(self):
        mock_record1 = MagicMock()
        mock_record1.data.return_value = {"a": "value1"}
        mock_record2 = MagicMock()
        mock_record2.data.return_value = {"a": "value2"}
        self.mock_session.run.return_value = [mock_record1, mock_record2]

        with patch.object(Neo4jStore, "_init_constraints", return_value=None):
            store = Neo4jStore("neo4j://localhost:7687", password="test-password")
            results = store.query("MATCH (n) RETURN n")
            self.assertEqual(results, [{"a": "value1"}, {"a": "value2"}])

    def test_ingest_facts_canonical_and_legacy_parity(self):
        with patch.object(Neo4jStore, "_init_constraints", return_value=None):
            store = Neo4jStore("neo4j://localhost:7687", password="test-password")
            canonical_fact = {
                "fact_statement": "Mitochondria produce ATP.",
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "category": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 10,
                "source_models": ["gpt-4", "qwen2.5:7b"],
            }
            legacy_fact = {
                "statement": "Antigravity helps developers",
                "subject": "Antigravity",
                "predicate": "helps",
                "object": "developers",
                "domain": "coding",
                "reliability": "LIKELY_TRUE",
                "agreement": 2,
                "quality": 5,
                "source_models": "model_a",
            }
            count = store.ingest_facts([canonical_fact, legacy_fact])
            self.assertEqual(count, 2)

            first_cypher, first_params = self.mock_session.run.call_args_list[0].args
            self.assertIn("MERGE", first_cypher)
            self.assertIn("SET", first_cypher)
            self.assertNotIn("ON CREATE SET", first_cypher)
            self.assertEqual(first_params["statement"], "Mitochondria produce ATP.")
            self.assertEqual(first_params["domain"], "biochemistry")
            self.assertEqual(first_params["reliability"], "VERIFIED")
            self.assertEqual(first_params["agreement"], 3)
            self.assertEqual(first_params["quality"], 10)
            self.assertEqual(first_params["source_models"], "gpt-4, qwen2.5:7b")

            second_params = self.mock_session.run.call_args_list[1].args[1]
            self.assertEqual(second_params["statement"], "Antigravity helps developers")
            self.assertEqual(second_params["reliability"], "LIKELY_TRUE")

    def test_count(self):
        mock_record = MagicMock()
        mock_record.data.return_value = {"n": 42}
        self.mock_session.run.return_value = [mock_record]

        with patch.object(Neo4jStore, "_init_constraints", return_value=None):
            store = Neo4jStore("neo4j://localhost:7687", password="test-password")
            self.assertEqual(store.count(), 42)
