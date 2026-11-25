"""Multi-extractor consensus ingestion and graph contradiction reconciliation."""

import os
import sys
from typing import Any, Dict, List

from .factory import batch_drop
from .graph_store_factory import get_graph_store, is_neo4j_uri
from .model_distill import ModelKnowledgeDistiller
from .store import KnowledgeStore


class ConsensusEngine:
    """Consensus engine to ingest documents via multiple extractors and flag conflicts."""

    def __init__(self, store_dir: str, graph_db_path: str = "graph_db"):
        self.store_dir = os.path.abspath(store_dir)
        self.graph_db_path = (
            graph_db_path
            if is_neo4j_uri(graph_db_path)
            else os.path.abspath(graph_db_path)
        )

    def process_with_consensus(
        self,
        file_path: str,
        engines: List[str],
        reliability: str = "likely_true",
        filter_name: str = "standard",
        coref: bool = False,
    ) -> Dict[str, Any]:
        """Ingest the same file using multiple extraction engines and resolve contradictions."""
        print(f"[Consensus] Ingesting {file_path} using engines: {engines}...")

        reports: Dict[str, Any] = {}
        all_facts: List[Dict[str, Any]] = []
        for engine in engines:
            try:
                report = batch_drop(
                    sources=[file_path],
                    store_dir=self.store_dir,
                    reliability=reliability,
                    filter_name=filter_name,
                    coref=coref,
                    engine=engine,
                    include_engine_in_hash=True,
                )
                reports[engine] = report
                for item in report.get("items") or []:
                    for fact in item.get("fact_records") or []:
                        fact_copy = dict(fact)
                        fact_copy["model_provenance"] = {
                            "model": engine,
                            "engine": engine,
                            "source": file_path,
                        }
                        all_facts.append(fact_copy)
            except Exception as exc:
                print(
                    f"[Consensus] Engine '{engine}' failed extraction: {exc}",
                    file=sys.stderr,
                )
                reports[engine] = {
                    "errors": 1,
                    "items": [{"status": "error", "error": str(exc)}],
                }

        if not all_facts and os.path.isdir(self.store_dir):
            store = KnowledgeStore(self.store_dir)
            for f in store.iter_facts():
                fact_copy = dict(f)
                if not fact_copy.get("model_provenance"):
                    eng = fact_copy.get("_engine") or fact_copy.get("engine") or "svo"
                    fact_copy["model_provenance"] = {"model": eng, "engine": eng}
                all_facts.append(fact_copy)

        distiller = ModelKnowledgeDistiller(
            all_facts, min_agreement=1, min_reliability="UNVERIFIED"
        )
        facts = distiller.select_facts()

        kstore = get_graph_store(self.graph_db_path)
        try:
            kstore.ingest_facts(facts)
            kstore.auto_link_relations()
            result = kstore.validate_and_reconcile()
            demoted = result.get("demoted", [])
            print(
                f"[Consensus] Completed graph validation: demoted {len(demoted)} contradictory facts."
            )
            return {
                "reports": reports,
                "demoted": demoted,
            }
        finally:
            kstore.close()
