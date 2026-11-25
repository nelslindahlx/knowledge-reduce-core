from datetime import datetime, timezone
import json
import os
import sqlite3
import threading
import time
from typing import Optional

from .factory import batch_drop
from .graph_store_factory import is_neo4j_uri


class WatcherDaemon:
    """Daemon to monitor a directory for new document files and automatically ingest them."""

    def __init__(
        self,
        watch_dir: str,
        store_dir: str,
        db_log_path: str = "watcher_state.db",
        reliability: str = "likely_true",
        filter_name: str = "standard",
        coref: bool = False,
        engine: str = "svo",
        graph_db: Optional[str] = None,
        distill_dir: Optional[str] = None,
    ):
        try:
            from watchdog.events import FileSystemEventHandler  # noqa: F401
            from watchdog.observers import Observer  # noqa: F401
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
        if graph_db:
            self.graph_db: Optional[str] = (
                graph_db if is_neo4j_uri(graph_db) else os.path.abspath(graph_db)
            )
        else:
            self.graph_db = None
        self.distill_dir = os.path.abspath(distill_dir) if distill_dir else None
        self._lock = threading.RLock()

        os.makedirs(self.watch_dir, exist_ok=True)
        os.makedirs(self.store_dir, exist_ok=True)

        self._init_db()

    def _connect_db(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_log_path, timeout=5.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return conn

    def _init_db(self) -> None:
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

    def log_file_state(
        self, file_path: str, mtime: float, status: str, error: Optional[str] = None
    ) -> None:
        norm_path = os.path.abspath(file_path)
        conn = self._connect_db()
        try:
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            with conn:
                conn.execute(
                    "INSERT OR REPLACE INTO watched_files"
                    "(file_path, last_modified, status, last_processed_at, error) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (norm_path, mtime, status, now, error),
                )
        finally:
            conn.close()

    def should_process(self, file_path: str, mtime: float) -> bool:
        norm_path = os.path.abspath(file_path)
        conn = self._connect_db()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT last_modified, status FROM watched_files WHERE file_path = ?",
                (norm_path,),
            )
            row = cursor.fetchone()
            if row:
                saved_mtime, status = row
                return bool(mtime > saved_mtime or status == "FAILED")
            return True
        finally:
            conn.close()

    def _sync_graph_db(self, report: dict) -> None:
        from .entity_resolution import resolve_and_merge_entities
        from .graph_store_factory import get_graph_store

        new_items = []
        for item in report.get("items", []):
            if item.get("status") in ("dropped", "success"):
                fact_records = item.get("fact_records")
                if isinstance(fact_records, list):
                    new_items.extend(fact_records)
                elif isinstance(item.get("facts"), list):
                    new_items.extend(item["facts"])

        print(f"[Watcher] Ingesting new facts into graph database at: {self.graph_db}")
        kstore = get_graph_store(self.graph_db)  # type: ignore[arg-type]
        try:
            if new_items:
                formatted_items = []
                for ni in new_items:
                    formatted_items.append(
                        {
                            "subject": ni.get("subject"),
                            "predicate": ni.get("predicate"),
                            "object": ni.get("object"),
                            "fact_statement": ni.get("fact_statement") or ni.get("statement"),
                            "domain": ni.get("domain") or ni.get("category"),
                            "reliability_rating": (
                                ni.get("reliability_rating")
                                or ni.get("reliability")
                                or "POSSIBLY_TRUE"
                            ),
                            "cross_model_agreement": (
                                ni.get("cross_model_agreement")
                                if ni.get("cross_model_agreement") is not None
                                else (
                                    ni.get("agreement")
                                    if ni.get("agreement") is not None
                                    else 1
                                )
                            ),
                            "quality_score": (
                                ni.get("quality_score")
                                if ni.get("quality_score") is not None
                                else (
                                    ni.get("quality") if ni.get("quality") is not None else 1
                                )
                            ),
                            "source_models": ni.get("source_models") or [],
                        }
                    )
                kstore.ingest_facts(formatted_items)

                print("[Watcher] Running graph entity resolution...")
                new_concepts = []
                for ni in formatted_items:
                    if ni.get("subject"):
                        new_concepts.append(ni["subject"])
                    if ni.get("object"):
                        new_concepts.append(ni["object"])
                resolve_and_merge_entities(kstore, limit_to_concepts=new_concepts)

                kstore.auto_link_relations()

                print("[Watcher] Running path validation and contradiction reconciliation...")
                kstore.validate_and_reconcile()

            if self.distill_dir:
                print(f"[Watcher] Distilling ontology to: {self.distill_dir}")
                from .ontology import OntologyDistiller

                os.makedirs(self.distill_dir, exist_ok=True)
                distiller = OntologyDistiller(kstore)
                summary = distiller.distill_all()

                out_path = os.path.join(self.distill_dir, "ontology_summary.json")
                tmp_path = f"{out_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as fh:
                    json.dump(summary, fh, indent=2)
                os.replace(tmp_path, out_path)
                print(f"[Watcher] Saved distilled ontology to {out_path}")
        finally:
            kstore.close()

    def process_file(self, file_path: str) -> None:
        from .factory import _SUPPORTED_EXTS

        norm_path = os.path.abspath(file_path)
        ext = os.path.splitext(norm_path)[1].lower()
        if ext not in _SUPPORTED_EXTS:
            return

        with self._lock:
            try:
                initial_mtime = os.path.getmtime(norm_path)
            except OSError:
                return

            if not self.should_process(norm_path, initial_mtime):
                return

            # Wait briefly for file write completion before sampling final mtime and reading.
            time.sleep(0.1)
            try:
                mtime = os.path.getmtime(norm_path)
            except OSError:
                return

            print(f"[Watcher] Processing new/modified file: {norm_path}")
            try:
                report = batch_drop(
                    sources=[norm_path],
                    store_dir=self.store_dir,
                    reliability=self.reliability,
                    filter_name=self.filter_name,
                    coref=self.coref,
                    engine=self.engine,
                )

                if report.get("errors", 0) > 0:
                    err_msg = "Unknown error"
                    for item in report.get("items", []):
                        if item.get("status") == "error":
                            err_msg = item.get("error", "Unknown error")
                            break
                    print(f"[Watcher] Failed processing: {norm_path} - {err_msg}")
                    self.log_file_state(norm_path, mtime, "FAILED", error=err_msg)
                    return

                item_status = "PROCESSED"
                for item in report.get("items", []):
                    if item.get("status") == "skipped":
                        item_status = "SKIPPED"

                if self.graph_db:
                    self._sync_graph_db(report)

                print(f"[Watcher] Success: {norm_path} (Status: {item_status})")
                self.log_file_state(norm_path, mtime, item_status)
            except Exception as exc:
                print(f"[Watcher] Exception processing: {norm_path} - {exc}")
                self.log_file_state(norm_path, mtime, "FAILED", error=str(exc))

    def run(self) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher_self = self

        class IngestHandler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher_self.process_file(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    watcher_self.process_file(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    watcher_self.process_file(event.dest_path)

        handler = IngestHandler()
        observer = Observer()
        observer.schedule(handler, path=self.watch_dir, recursive=True)
        observer.start()

        print(f"[Watcher] Performing startup scan of {self.watch_dir}...")
        for root, _dirs, files in os.walk(self.watch_dir):
            for name in files:
                path = os.path.join(root, name)
                self.process_file(path)

        print(f"[Watcher] Watching directory {self.watch_dir} for new papers/articles...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
