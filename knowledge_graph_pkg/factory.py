"""
Batch automation for the knowledge factory (the "ongoing" layer).

:func:`batch_drop` ingests many sources into a store in one pass --
skipping already-ingested unchanged sources (content-hash idempotency) and
returning a structured run report. :func:`scan_folder` finds supported
documents in a directory. Together they back scheduled or watch-folder
automation: point the factory at a folder / list of sources on a cadence
and it grows the store unattended, reporting what it did.
"""

import os
from typing import Any, Dict, List

# Extensions the ingest layer can handle (PDF requires the [pdf] extra).
_SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}


def scan_folder(folder: str, recursive: bool = False) -> List[str]:
    """Return supported document paths in ``folder`` (sorted)."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    found = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                if os.path.splitext(name)[1].lower() in _SUPPORTED_EXTS:
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in _SUPPORTED_EXTS:
                found.append(path)
    return sorted(found)


def batch_drop(sources: List[str], store_dir: str,
               reliability: str = "likely_true", filter_name: str = "standard",
               coref: bool = False, engine: str = "svo",
               max_object_len: int = 80, dedup: float = 0.9,
               include_engine_in_hash: bool = False) -> Dict[str, Any]:
    """Drop many sources into a store; skip unchanged ones; return a report.

    The report has ``dropped`` / ``skipped`` / ``errors`` counts, a
    ``total_facts`` tally, and a per-source ``items`` list with status.
    """
    # Lazy imports keep module load cheap.
    from .core import KnowledgeGraph, ReliabilityRating
    from .semantic import SemanticKnowledgeGraph
    from .distillation import KnowledgeDistiller
    from .quality import FactQualityFilter
    from .ingest import load_text
    from .store import KnowledgeStore, Drop, content_hash

    rel_map = {
        "unverified": ReliabilityRating.UNVERIFIED,
        "possibly_true": ReliabilityRating.POSSIBLY_TRUE,
        "likely_true": ReliabilityRating.LIKELY_TRUE,
        "verified": ReliabilityRating.VERIFIED,
    }
    reliability_enum = rel_map.get(str(reliability).lower(), ReliabilityRating.LIKELY_TRUE)

    extractor = None
    if engine != "svo":
        from .extractor_base import get_extractor
        extractor = get_extractor(engine)

    if filter_name == "none":
        qf = None
    elif filter_name == "strict":
        qf = FactQualityFilter(max_object_len=min(max_object_len, 60),
                               require_entity_subject=True)
    else:
        qf = FactQualityFilter(max_object_len=max_object_len)

    store = KnowledgeStore(store_dir)
    report = {"dropped": 0, "skipped": 0, "errors": 0, "total_facts": 0, "items": []}

    for src in sources:
        item = {"source": src, "status": None, "facts": 0}
        try:
            if not os.path.isfile(src):
                raise FileNotFoundError(src)
            text = load_text(src)
            hash_input = f"{engine}\x00{text}" if include_engine_in_hash else text
            src_hash = content_hash(hash_input)

            if store.has_source_hash(src_hash):
                item["status"] = "skipped"
                report["skipped"] += 1
                report["items"].append(item)
                continue

            kg = KnowledgeGraph()
            skg = SemanticKnowledgeGraph(kg)
            skg.create_facts_from_text(text, source_id=os.path.basename(src),
                                       reliability=reliability_enum,
                                       resolve_coref=coref, extractor=extractor)
            distiller = KnowledgeDistiller(kg, min_reliability=reliability_enum,
                                           dedup_threshold=dedup, quality_filter=qf)
            facts = distiller.select_facts()
            for fact in facts:
                fact.setdefault("_engine", engine)

            base = os.path.splitext(os.path.basename(src))[0]
            safe_engine = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(engine))
            drop_id = (
                f"{base}-{safe_engine}-{src_hash[:12]}"
                if include_engine_in_hash
                else f"{base}-{src_hash[:12]}"
            )
            drop = Drop(drop_id=drop_id, source=src,
                        source_hash=src_hash, facts=facts, engine=engine,
                        filter_name=filter_name, coref=coref, source_text=text)
            store.write_drop(drop)

            item["status"] = "dropped"
            item["facts"] = len(facts)
            item["fact_records"] = drop.facts
            report["dropped"] += 1
            report["total_facts"] += len(facts)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the batch
            item["status"] = "error"
            item["error"] = str(exc)
            report["errors"] += 1
        report["items"].append(item)

    return report


def format_report(report: Dict[str, Any]) -> str:
    """Render a batch run report as a human-readable block."""
    lines = [
        "Batch drop run",
        f"  dropped:     {report['dropped']}",
        f"  skipped:     {report['skipped']}",
        f"  errors:      {report['errors']}",
        f"  total facts: {report['total_facts']}",
    ]
    for it in report["items"]:
        tag = it["status"]
        if tag == "dropped":
            extra = f" ({it.get('facts', 0)} facts)"
        elif tag == "error":
            extra = f" ({it.get('error', '')})"
        else:
            extra = ""
        lines.append(f"    [{tag}] {it['source']}{extra}")
    return "\n".join(lines)
