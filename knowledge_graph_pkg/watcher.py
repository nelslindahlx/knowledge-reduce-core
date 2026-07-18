import os
import sys
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from .factory import batch_drop

class WatcherDaemon:
    """Daemon to monitor a directory for new document files and automatically ingest them."""

    def __init__(self, watch_dir: str, store_dir: str, db_log_path: str = "watcher_state.db",
                 reliability: str = "likely_true", filter_name: str = "standard",
                 coref: bool = False, engine: str = "svo", graph_db: Optional[str] = None,
                 distill_dir: Optional[str] = None):
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError as exc:
            raise ImportError(
                "WatcherDaemon requires the watcher extra: pip install knowledgereduce[watcher]"
            ) from exc

        self.watch_dir = os.path.abspath(watch_dir)
        self.store_dir = os.path.abspath(store_dir)
        self.db_log_path = os.path.abspath(db_log_path)
        self.reliability = reliability
        self.filter_name = filter_name
        self.coref = coref
        self.engine = engine
        self.graph_db = os.path.abspath(graph_db) if graph_db else None
        self.distill_dir = os.path.abspath(distill_dir) if distill_dir else None

        # Ensure directories exist
        os.makedirs(self.watch_dir, exist_ok=True)
        os.makedirs(self.store_dir, exist_ok=True)

        self._init_db()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_log_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
        except Exception:
            pass
        return conn

    def _init_db(self):
        conn = self._connect_db()
        try:
            with conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS watched_files("
                    "file_path TEXT PRIMARY KEY, "
                    "last_modified REAL, "
                    "status TEXT, "
                    "last_processed_at TEXT, "
                    "error TEXT)"
                )
        finally:
            conn.close()

    def log_file_state(self, file_path: str, mtime: float, status: str, error: Optional[str] = None):
        conn = self._connect_db()
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO watched_files(file_path, last_modified, status, last_processed_at, error) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (file_path, mtime, status, now, error)
                )
        finally:
            conn.close()

    def should_process(self, file_path: str, mtime: float) -> bool:
        conn = self._connect_db()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT last_modified, status FROM watched_files WHERE file_path = ?", (file_path,))
            row = cursor.fetchone()
            if row:
                saved_mtime, status = row
                if mtime > saved_mtime or status == "FAILED":
                    return True
                return False
            return True
        finally:
            conn.close()

    def process_file(self, file_path: str):
        from .factory import _SUPPORTED_EXTS
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in _SUPPORTED_EXTS:
            return

        try:
            mtime = os.path.getmtime(file_path)
        except OSError:
            # File might have been deleted or moved
            return

        if not self.should_process(file_path, mtime):
            return

        # Wait briefly for file write completion to avoid reading partial file
        time.sleep(0.5)

        print(f"[Watcher] Processing new/modified file: {file_path}")
        try:
            report = batch_drop(
                sources=[file_path],
                store_dir=self.store_dir,
                reliability=self.reliability,
                filter_name=self.filter_name,
                coref=self.coref,
                engine=self.engine
            )

            if report.get("errors", 0) > 0:
                err_msg = "Unknown error"
                for item in report.get("items", []):
                    if item.get("status") == "error":
                        err_msg = item.get("error", "Unknown error")
                        break
                print(f"[Watcher] Failed processing: {file_path} - {err_msg}")
                self.log_file_state(file_path, mtime, "FAILED", error=err_msg)
            else:
                item_status = "PROCESSED"
                for item in report.get("items", []):
                    if item.get("status") == "skipped":
                        item_status = "SKIPPED"
                print(f"[Watcher] Success: {file_path} (Status: {item_status})")
                self.log_file_state(file_path, mtime, item_status)
                
                if self.graph_db:
                    try:
                        from .graph_store_factory import get_graph_store
                        from .entity_resolution import resolve_and_merge_entities
                        
                        print(f"[Watcher] Ingesting new facts into graph database at: {self.graph_db}")
                        kstore = get_graph_store(self.graph_db)
                        try:
                            new_items = []
                            for item in report.get("items", []):
                                if item.get("status") == "success":
                                    new_items.extend(item.get("facts", []))
                                    
                            if new_items:
                                formatted_items = []
                                for ni in new_items:
                                    formatted_items.append({
                                        "subject": ni.get("subject"),
                                        "predicate": ni.get("predicate"),
                                        "object": ni.get("object"),
                                        "fact_statement": ni.get("statement") or ni.get("fact_statement"),
                                        "domain": ni.get("domain") or ni.get("category"),
                                        "reliability_rating": ni.get("reliability") or ni.get("reliability_rating") or "POSSIBLY_TRUE",
                                        "cross_model_agreement": ni.get("agreement") or ni.get("cross_model_agreement") or 1,
                                        "quality_score": ni.get("quality") or ni.get("quality_score") or 1,
                                        "source_models": ni.get("source_models") or []
                                    })
                                kstore.ingest_facts(formatted_items)
                                kstore.auto_link_relations()
                                
                                print("[Watcher] Running graph entity resolution...")
                                new_concepts = []
                                for ni in formatted_items:
                                    if ni.get("subject"):
                                        new_concepts.append(ni["subject"])
                                    if ni.get("object"):
                                        new_concepts.append(ni["object"])
                                resolve_and_merge_entities(kstore, limit_to_concepts=new_concepts)
                                
                                print("[Watcher] Running path validation and contradiction reconciliation...")
                                kstore.validate_and_reconcile()
                                
                                if self.distill_dir:
                                    print(f"[Watcher] Distilling ontology to: {self.distill_dir}")
                                    try:
                                        from .ontology import OntologyDistiller
                                        import json
                                        os.makedirs(self.distill_dir, exist_ok=True)
                                        distiller = OntologyDistiller(kstore)
                                        taxonomy = distiller.distill_taxonomy()
                                        types = distiller.infer_semantic_types()
                                        schema = distiller.infer_relation_schema()
                                        
                                        summary = {
                                            "taxonomy": taxonomy,
                                            "semantic_types": types,
                                            "relation_schema": schema
                                        }
                                        
                                        out_path = os.path.join(self.distill_dir, "ontology_summary.json")
                                        with open(out_path, "w", encoding="utf-8") as fh:
                                            json.dump(summary, fh, indent=2)
                                        print(f"[Watcher] Saved distilled ontology to {out_path}")
                                    except Exception as d_exc:
                                        print(f"[Watcher] Ontology distillation failed: {d_exc}")
                        finally:
                            kstore.close()
                    except Exception as g_exc:
                        print(f"[Watcher] Error updating graph database: {g_exc}")
        except Exception as exc:
            print(f"[Watcher] Exception processing: {file_path} - {exc}")
            self.log_file_state(file_path, mtime, "FAILED", error=str(exc))

    def run(self):
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        watcher_self = self

        class IngestHandler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher_self.process_file(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    watcher_self.process_file(event.src_path)

        # Do a startup scan of existing files in directory
        print(f"[Watcher] Performing startup scan of {self.watch_dir}...")
        for root, _dirs, files in os.walk(self.watch_dir):
            for name in files:
                path = os.path.join(root, name)
                self.process_file(path)

        handler = IngestHandler()
        observer = Observer()
        observer.schedule(handler, path=self.watch_dir, recursive=True)
        observer.start()
        print(f"[Watcher] Watching directory {self.watch_dir} for new papers/articles...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
