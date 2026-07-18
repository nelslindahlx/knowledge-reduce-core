import os
import sys
from typing import List, Dict, Any, Optional
from .store import KnowledgeStore
from .kuzu_store import KuzuStore
from .factory import batch_drop
from .model_distill import ModelKnowledgeDistiller

class ConsensusEngine:
    """Consensus engine to ingest documents via multiple extractors and flag conflicts."""

    def __init__(self, store_dir: str, graph_db_path: str = "graph_db"):
        self.store_dir = os.path.abspath(store_dir)
        self.graph_db_path = os.path.abspath(graph_db_path)

    def process_with_consensus(self, file_path: str, engines: List[str],
                               reliability: str = "likely_true",
                               filter_name: str = "standard",
                               coref: bool = False) -> Dict[str, Any]:
        """Ingest the same file using multiple extraction engines and resolve contradictions."""
        print(f"[Consensus] Ingesting {file_path} using engines: {engines}...")

        # Ingest using batch_drop for each engine
        reports = {}
        for engine in engines:
            try:
                report = batch_drop(
                    sources=[file_path],
                    store_dir=self.store_dir,
                    reliability=reliability,
                    filter_name=filter_name,
                    coref=coref,
                    engine=engine
                )
                reports[engine] = report
            except Exception as exc:
                print(f"[Consensus] Engine '{engine}' failed extraction: {exc}", file=sys.stderr)
                reports[engine] = {"errors": 1, "items": [{"status": "error", "error": str(exc)}]}

        # Load the store drops into the Kuzu Graph
        store = KnowledgeStore(self.store_dir)
        distiller = ModelKnowledgeDistiller.from_store(
            store, min_agreement=1, min_reliability="UNVERIFIED"
        )
        facts = distiller.select_facts()
        
        kstore = KuzuStore(self.graph_db_path)
        try:
            kstore.ingest_facts(facts)
            kstore.auto_link_relations()
            result = kstore.validate_and_reconcile()
            demoted = result.get("demoted", [])
            print(f"[Consensus] Completed graph validation: demoted {len(demoted)} contradictory facts.")
            return {
                "reports": reports,
                "demoted": demoted
            }
        finally:
            kstore.close()
