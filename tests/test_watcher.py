import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import pytest

from knowledge_graph_pkg.cli import main
from knowledge_graph_pkg.watcher import WatcherDaemon

pytest.importorskip("watchdog")


class TestWatcherDaemon(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_watcher_")
        self.watch_dir = os.path.join(self.temp_dir, "watch")
        self.store_dir = os.path.join(self.temp_dir, "store")
        self.db_log = os.path.join(self.temp_dir, "watcher.db")
        self.graph_db = os.path.join(self.temp_dir, "graph_db")
        self.distill_dir = os.path.join(self.temp_dir, "distill")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        WatcherDaemon(
            watch_dir=self.watch_dir, store_dir=self.store_dir, db_log_path=self.db_log
        )

        self.assertTrue(os.path.isdir(self.watch_dir))
        self.assertTrue(os.path.isdir(self.store_dir))
        self.assertTrue(os.path.isfile(self.db_log))

        conn = sqlite3.connect(self.db_log)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='watched_files'"
        )
        self.assertIsNotNone(cursor.fetchone())
        conn.close()

    def test_neo4j_uri_not_mangled_by_abspath(self):
        daemon = WatcherDaemon(
            watch_dir=self.watch_dir,
            store_dir=self.store_dir,
            db_log_path=self.db_log,
            graph_db="neo4j://localhost:7687/tenant1",
        )
        self.assertEqual(daemon.graph_db, "neo4j://localhost:7687/tenant1")

    def test_should_process_and_log(self):
        daemon = WatcherDaemon(
            watch_dir=self.watch_dir, store_dir=self.store_dir, db_log_path=self.db_log
        )
        sample_file = os.path.join(self.watch_dir, "test.txt")

        self.assertTrue(daemon.should_process(sample_file, 100.0))
        daemon.log_file_state(sample_file, 100.0, "PROCESSED")
        self.assertFalse(daemon.should_process(sample_file, 100.0))
        self.assertTrue(daemon.should_process(sample_file, 101.0))

        daemon.log_file_state(sample_file, 101.0, "FAILED")
        self.assertTrue(daemon.should_process(sample_file, 101.0))

    @patch("knowledge_graph_pkg.watcher.batch_drop")
    def test_process_file_success(self, mock_batch_drop):
        daemon = WatcherDaemon(
            watch_dir=self.watch_dir, store_dir=self.store_dir, db_log_path=self.db_log
        )
        mock_batch_drop.return_value = {
            "dropped": 1,
            "errors": 0,
            "items": [{"status": "dropped", "facts": 1, "fact_records": []}],
        }

        test_file = os.path.join(self.watch_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Some text content")

        daemon.process_file(test_file)

        conn = sqlite3.connect(self.db_log)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status FROM watched_files WHERE file_path = ?",
            (os.path.abspath(test_file),),
        )
        row = cursor.fetchone()
        self.assertEqual(row[0], "PROCESSED")
        conn.close()

    @patch("knowledge_graph_pkg.watcher.batch_drop")
    def test_process_file_failure(self, mock_batch_drop):
        daemon = WatcherDaemon(
            watch_dir=self.watch_dir, store_dir=self.store_dir, db_log_path=self.db_log
        )
        mock_batch_drop.return_value = {
            "dropped": 0,
            "errors": 1,
            "items": [{"status": "error", "error": "Extraction failed", "file": "test.txt"}],
        }

        test_file = os.path.join(self.watch_dir, "test.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Some text content")

        daemon.process_file(test_file)

        conn = sqlite3.connect(self.db_log)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT status, error FROM watched_files WHERE file_path = ?",
            (os.path.abspath(test_file),),
        )
        row = cursor.fetchone()
        self.assertEqual(row[0], "FAILED")
        self.assertEqual(row[1], "Extraction failed")
        conn.close()

    def test_cli_watch_daemon(self):
        with patch(
            "sys.argv",
            [
                "knowledgereduce",
                "watch-daemon",
                "--dir",
                self.watch_dir,
                "--store",
                self.store_dir,
                "--db-log",
                self.db_log,
            ],
        ):
            with patch("knowledge_graph_pkg.watcher.WatcherDaemon.run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()
                code = main()
                self.assertEqual(code, 0)
                mock_run.assert_called_once()

    def test_watcher_daemon_graph_failure_logs_failed(self):
        daemon = WatcherDaemon(
            watch_dir=self.watch_dir,
            store_dir=self.store_dir,
            db_log_path=self.db_log,
            graph_db=self.graph_db,
        )
        test_file = os.path.join(self.watch_dir, "doc.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("Mitochondria produce ATP. ATP fuels cells.")

        with patch.object(daemon, "_sync_graph_db", side_effect=RuntimeError("graph db boom")):
            daemon.process_file(test_file)

        conn = sqlite3.connect(self.db_log)
        row = conn.cursor().execute(
            "SELECT status, error FROM watched_files WHERE file_path = ?",
            (os.path.abspath(test_file),),
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], "FAILED")
        self.assertIn("graph db boom", row[1])

    def test_watcher_daemon_end_to_end_real_pipeline(self):
        pytest.importorskip("kuzu")
        from knowledge_graph_pkg.graph_store_factory import get_graph_store

        daemon = WatcherDaemon(
            watch_dir=self.watch_dir,
            store_dir=self.store_dir,
            db_log_path=self.db_log,
            graph_db=self.graph_db,
            distill_dir=self.distill_dir,
        )

        test_file = os.path.join(self.watch_dir, "bio.txt")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(
                "Mitochondria produce ATP. ATP fuels cellular respiration. "
                "ATP is a nucleotide."
            )

        daemon.process_file(test_file)

        # Verify SQLite marked PROCESSED
        conn = sqlite3.connect(self.db_log)
        row = conn.cursor().execute(
            "SELECT status FROM watched_files WHERE file_path = ?",
            (os.path.abspath(test_file),),
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], "PROCESSED")

        # Verify KùzuDB actually received facts and RELATED edges (NO mocking batch_drop)
        kstore = get_graph_store(self.graph_db)
        try:
            self.assertGreaterEqual(kstore.count(), 2)
            edges = kstore.query("MATCH (a:Fact)-[r:RELATED]->(b:Fact) RETURN count(r) AS n")
            self.assertGreaterEqual(edges[0]["n"], 1)
        finally:
            kstore.close()

        # Verify ontology summary JSON was written atomically
        summary_path = os.path.join(self.distill_dir, "ontology_summary.json")
        self.assertTrue(os.path.isfile(summary_path))
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
            self.assertIn("taxonomy", summary)
            self.assertIn("semantic_types", summary)
            self.assertIn("relation_schema", summary)
