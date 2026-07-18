import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock the entire neo4j module
mock_neo4j = MagicMock()
sys.modules["neo4j"] = mock_neo4j

from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.neo4j_store import Neo4jStore


class TestNeo4jStore(unittest.TestCase):

    def setUp(self):
        mock_neo4j.reset_mock()
        self.mock_driver = MagicMock()
        mock_neo4j.GraphDatabase.driver.return_value = self.mock_driver
        self.mock_session = MagicMock()
        self.mock_driver.session.return_value.__enter__.return_value = self.mock_session

    def test_factory_routing(self):
        import pytest
        pytest.importorskip("kuzu")
        store = get_graph_store("local_path")
        self.assertEqual(store.__class__.__name__, "KuzuStore")
        store.close()

        with patch.object(Neo4jStore, '_init_constraints', return_value=None):
            store_neo = get_graph_store("neo4j://localhost:7687")
            self.assertEqual(store_neo.__class__.__name__, "Neo4jStore")

    def test_query_mapping(self):
        mock_record1 = MagicMock()
        mock_record1.data.return_value = {"a": "value1"}
        mock_record2 = MagicMock()
        mock_record2.data.return_value = {"a": "value2"}
        self.mock_session.run.return_value = [mock_record1, mock_record2]

        with patch.object(Neo4jStore, '_init_constraints', return_value=None):
            store = Neo4jStore("neo4j://localhost:7687")
            results = store.query("MATCH (n) RETURN n")
            self.assertEqual(results, [{"a": "value1"}, {"a": "value2"}])

    def test_ingest_facts(self):
        with patch.object(Neo4jStore, '_init_constraints', return_value=None):
            store = Neo4jStore("neo4j://localhost:7687")
            fact = {
                "statement": "Antigravity helps developers",
                "subject": "Antigravity",
                "predicate": "helps",
                "object": "developers",
                "domain": "coding",
                "reliability": "VERIFIED",
                "agreement": 3,
                "quality": 0.95,
                "source_models": "model_a"
            }
            count = store.ingest_facts([fact])
            self.assertEqual(count, 1)
            args, kwargs = self.mock_session.run.call_args
            self.assertIn("MERGE", args[0])
            params = args[1]
            self.assertEqual(params["subject"], "Antigravity")
            self.assertEqual(params["predicate"], "helps")

    def test_count(self):
        mock_record = MagicMock()
        mock_record.data.return_value = {"n": 42}
        self.mock_session.run.return_value = [mock_record]

        with patch.object(Neo4jStore, '_init_constraints', return_value=None):
            store = Neo4jStore("neo4j://localhost:7687")
            self.assertEqual(store.count(), 42)
