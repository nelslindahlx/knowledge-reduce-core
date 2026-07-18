import sys
from unittest.mock import MagicMock

# Setup dummy watchdog modules before imports occur
mock_watchdog = MagicMock()
mock_observers = MagicMock()
mock_events = MagicMock()

mock_Observer_class = MagicMock()
mock_observers.Observer = mock_Observer_class

mock_FileSystemEventHandler = MagicMock()
mock_events.FileSystemEventHandler = mock_FileSystemEventHandler

sys.modules['watchdog'] = mock_watchdog
sys.modules['watchdog.observers'] = mock_observers
sys.modules['watchdog.events'] = mock_events

import unittest
import os
import sqlite3
import pytest
from unittest.mock import patch
from knowledge_graph_pkg.watcher import WatcherDaemon
from knowledge_graph_pkg.cli import main

pytest.importorskip("kuzu")

class TestWatcherDaemon(unittest.TestCase):

    def test_initialization(self):
        watch_dir = "test_watch"
        store_dir = "test_store"
        db_log = "test_watcher.db"
        
        # Clean up files if they exist
        if os.path.exists(db_log):
            os.remove(db_log)
        if os.path.exists(watch_dir):
            os.rmdir(watch_dir)
        if os.path.exists(store_dir):
            os.rmdir(store_dir)
            
        daemon = WatcherDaemon(watch_dir=watch_dir, store_dir=store_dir, db_log_path=db_log)
        
        self.assertTrue(os.path.isdir(watch_dir))
        self.assertTrue(os.path.isdir(store_dir))
        self.assertTrue(os.path.isfile(db_log))
        
        # Verify db structure
        conn = sqlite3.connect(db_log)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='watched_files'")
        self.assertIsNotNone(cursor.fetchone())
        conn.close()
        
        # Clean up
        os.remove(db_log)
        os.rmdir(watch_dir)
        os.rmdir(store_dir)

    def test_should_process_and_log(self):
        watch_dir = "test_watch"
        store_dir = "test_store"
        db_log = "test_watcher.db"
        
        daemon = WatcherDaemon(watch_dir=watch_dir, store_dir=store_dir, db_log_path=db_log)
        
        # 1. New file should process
        self.assertTrue(daemon.should_process("test.txt", 100.0))
        
        # 2. Log state as processed
        daemon.log_file_state("test.txt", 100.0, "PROCESSED")
        
        # 3. Same modification time should NOT process
        self.assertFalse(daemon.should_process("test.txt", 100.0))
        
        # 4. Newer modification time should process
        self.assertTrue(daemon.should_process("test.txt", 101.0))
        
        # 5. Failed status should process
        daemon.log_file_state("test.txt", 101.0, "FAILED")
        self.assertTrue(daemon.should_process("test.txt", 101.0))
        
        # Clean up
        os.remove(db_log)
        os.rmdir(watch_dir)
        os.rmdir(store_dir)

    @patch('knowledge_graph_pkg.watcher.batch_drop')
    def test_process_file_success(self, mock_batch_drop):
        watch_dir = "test_watch"
        store_dir = "test_store"
        db_log = "test_watcher.db"
        
        daemon = WatcherDaemon(watch_dir=watch_dir, store_dir=store_dir, db_log_path=db_log)
        
        # Mock batch_drop return value
        mock_batch_drop.return_value = {
            "dropped": 1,
            "errors": 0,
            "items": [{"status": "dropped", "file": "test.txt"}]
        }
        
        # Create a dummy test file
        os.makedirs(watch_dir, exist_ok=True)
        test_file = os.path.join(watch_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Some text content")
            
        daemon.process_file(test_file)
        
        # Check that it's processed and logged
        conn = sqlite3.connect(db_log)
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM watched_files WHERE file_path = ?", (test_file,))
        row = cursor.fetchone()
        self.assertEqual(row[0], "PROCESSED")
        conn.close()
        
        # Clean up
        os.remove(test_file)
        os.remove(db_log)
        os.rmdir(watch_dir)
        os.rmdir(store_dir)

    @patch('knowledge_graph_pkg.watcher.batch_drop')
    def test_process_file_failure(self, mock_batch_drop):
        watch_dir = "test_watch"
        store_dir = "test_store"
        db_log = "test_watcher.db"
        
        daemon = WatcherDaemon(watch_dir=watch_dir, store_dir=store_dir, db_log_path=db_log)
        
        # Mock batch_drop return value indicating an error
        mock_batch_drop.return_value = {
            "dropped": 0,
            "errors": 1,
            "items": [{"status": "error", "error": "Extraction failed", "file": "test.txt"}]
        }
        
        os.makedirs(watch_dir, exist_ok=True)
        test_file = os.path.join(watch_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Some text content")
            
        daemon.process_file(test_file)
        
        conn = sqlite3.connect(db_log)
        cursor = conn.cursor()
        cursor.execute("SELECT status, error FROM watched_files WHERE file_path = ?", (test_file,))
        row = cursor.fetchone()
        self.assertEqual(row[0], "FAILED")
        self.assertEqual(row[1], "Extraction failed")
        conn.close()
        
        # Clean up
        os.remove(test_file)
        os.remove(db_log)
        os.rmdir(watch_dir)
        os.rmdir(store_dir)

    def test_cli_watch_daemon(self):
        # We test routing of arguments in watch-daemon command
        with patch("sys.argv", ["knowledgereduce", "watch-daemon", "--dir", "test_watch", "--store", "test_store", "--db-log", "test_watcher.db"]):
            with patch("knowledge_graph_pkg.watcher.WatcherDaemon.run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()
                code = main()
                self.assertEqual(code, 0)
                mock_run.assert_called_once()
                
        # Clean up folders created by WatcherDaemon constructor
        if os.path.exists("test_watcher.db"):
            os.remove("test_watcher.db")
        if os.path.exists("test_watch"):
            os.rmdir("test_watch")
        if os.path.exists("test_store"):
            os.rmdir("test_store")

    @patch("knowledge_graph_pkg.watcher.batch_drop")
    def test_watcher_daemon_graph_pipeline(self, mock_batch_drop):
        import shutil
        watch_dir = "test_watch_g"
        store_dir = "test_store_g"
        db_log = "test_watcher_g.db"
        graph_db = "test_graph_db_g"
        
        daemon = WatcherDaemon(
            watch_dir=watch_dir, 
            store_dir=store_dir, 
            db_log_path=db_log, 
            graph_db=graph_db
        )
        
        mock_batch_drop.return_value = {
            "dropped": 1,
            "errors": 0,
            "items": [{
                "status": "success",
                "facts": [{
                    "subject": "Mitochondria",
                    "predicate": "produce",
                    "object": "ATP",
                    "statement": "Mitochondria produce ATP.",
                    "domain": "biochemistry",
                    "reliability": "VERIFIED",
                    "agreement": 3,
                    "quality": 1
                }]
            }]
        }
        
        os.makedirs(watch_dir, exist_ok=True)
        test_file = os.path.join(watch_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Some text content")
            
        daemon.process_file(test_file)
        
        from knowledge_graph_pkg.graph_store_factory import get_graph_store
        kstore = get_graph_store(graph_db)
        try:
            self.assertEqual(kstore.count(), 1)
            facts = kstore.query("MATCH (f:Fact) RETURN f.statement AS statement")
            self.assertEqual(facts[0]["statement"], "Mitochondria produce ATP.")
        finally:
            kstore.close()
            
        # Clean up
        os.remove(test_file)
        os.remove(db_log)
        os.rmdir(watch_dir)
        os.rmdir(store_dir)
        if os.path.exists(daemon.graph_db):
            if os.path.isdir(daemon.graph_db):
                shutil.rmtree(daemon.graph_db)
            else:
                os.remove(daemon.graph_db)

    @patch("knowledge_graph_pkg.watcher.batch_drop")
    def test_watcher_daemon_distill_pipeline(self, mock_batch_drop):
        import shutil
        import json
        watch_dir = "test_watch_d"
        store_dir = "test_store_d"
        db_log = "test_watcher_d.db"
        graph_db = "test_graph_db_d"
        distill_dir = "test_distill_d"
        
        daemon = WatcherDaemon(
            watch_dir=watch_dir, 
            store_dir=store_dir, 
            db_log_path=db_log, 
            graph_db=graph_db,
            distill_dir=distill_dir
        )
        
        mock_batch_drop.return_value = {
            "dropped": 1,
            "errors": 0,
            "items": [{
                "status": "success",
                "facts": [{
                    "subject": "ATP",
                    "predicate": "is a",
                    "object": "ChemicalCompound",
                    "statement": "ATP is a ChemicalCompound.",
                    "domain": "biochemistry",
                    "reliability": "VERIFIED",
                    "agreement": 3,
                    "quality": 1
                }]
            }]
        }
        
        os.makedirs(watch_dir, exist_ok=True)
        test_file = os.path.join(watch_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("Some text content")
            
        daemon.process_file(test_file)
        
        # Verify distilled ontology exists
        summary_path = os.path.join(distill_dir, "ontology_summary.json")
        self.assertTrue(os.path.isfile(summary_path))
        
        with open(summary_path, "r") as f:
            summary = json.load(f)
            self.assertIn("taxonomy", summary)
            self.assertIn("semantic_types", summary)
            self.assertIn("relation_schema", summary)
            
        # Clean up
        os.remove(test_file)
        os.remove(db_log)
        os.rmdir(watch_dir)
        os.rmdir(store_dir)
        shutil.rmtree(distill_dir)
        if os.path.exists(daemon.graph_db):
            if os.path.isdir(daemon.graph_db):
                shutil.rmtree(daemon.graph_db)
            else:
                os.remove(daemon.graph_db)

