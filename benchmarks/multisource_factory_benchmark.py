"""Offline multi-source incremental factory, contradiction & GraphRAG golden benchmark."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks._in_memory_store import InMemoryGraphStore, chat_jsonl_from_facts  # noqa: E402
from knowledge_graph_pkg.catalog import Catalog  # noqa: E402
from knowledge_graph_pkg.factory import batch_drop  # noqa: E402
from knowledge_graph_pkg.graph_compiler import compile_contradiction_instructions  # noqa: E402
from knowledge_graph_pkg.rag import GraphRAGRetriever  # noqa: E402
from knowledge_graph_pkg.store import KnowledgeStore  # noqa: E402
from knowledge_graph_pkg.training_prep import validate_chat_record  # noqa: E402

DEFAULT_MULTISOURCE_FLOORS: dict[str, dict[str, float | int]] = {
    "min": {
        "base_sources": 2,
        "delta_sources": 2,
        "skipped_unchanged": 1,
        "conflict_detection_recall": 1.0,
        "conflict_resolution_accuracy": 1.0,
        "conflicts_detected": 3,
        "retrieval_hit_at_1": 1.0,
        "valid_chat_ratio": 1.0,
    },
    "max": {
        "errors": 0,
        "total_seconds": 5.0,
    },
}


@dataclass(frozen=True)
class MultiSourceFactoryMetrics:
    """Quantitative quality, lineage, conflict, and latency metrics."""

    base_sources: int
    delta_sources: int
    skipped_unchanged: int
    disputed_sources: int
    total_drops: int
    total_facts: int
    catalog_indexed: int
    catalog_reliable_count: int
    conflicts_detected: int
    conflicts_demoted: int
    conflict_detection_recall: float
    conflict_resolution_accuracy: float
    retrieval_hit_at_1: float
    unverified_suppressed: bool
    contradiction_instructions: int
    chat_records: int
    valid_chat_ratio: float
    errors: int
    total_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)


@dataclass
class MultiSourceFactoryResult:
    """Complete artifacts and metrics returned by :func:`evaluate_multisource_factory`."""

    metrics: MultiSourceFactoryMetrics
    fact_statements: list[str]
    chat_jsonl_lines: list[str]
    passed_floors: bool
    floor_failures: list[str]


def _check_floors(metrics: MultiSourceFactoryMetrics) -> tuple[bool, list[str]]:
    payload = metrics.to_dict()
    failures: list[str] = []
    for key, min_val in DEFAULT_MULTISOURCE_FLOORS["min"].items():
        if float(payload[key]) < float(min_val):
            failures.append(f"{key}={payload[key]} < min {min_val}")
    for key, max_val in DEFAULT_MULTISOURCE_FLOORS["max"].items():
        if float(payload[key]) > float(max_val):
            failures.append(f"{key}={payload[key]} > max {max_val}")
    return len(failures) == 0, failures


def evaluate_multisource_factory() -> MultiSourceFactoryResult:
    """Run offline HTML + Markdown incremental drop, conflict reconcile, and RAG benchmark."""
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="kr_multisource_golden_") as tmp:
        root = Path(tmp)
        docs_dir = root / "corpus"
        docs_dir.mkdir(parents=True, exist_ok=True)
        store_dir = str(root / "store")

        overview_html = docs_dir / "overview.html"
        overview_html.write_text(
            "<html><head><title>Civic Honors Overview</title></head><body><main>\n"
            "<h1>Program History</h1>\n"
            "<p>Graduation With Civic Honors was published in 2006.</p>\n"
            "<p>Nels Lindahl developed a civic honors program.</p>\n"
            "</main></body></html>\n",
            encoding="utf-8",
        )

        curriculum_md = docs_dir / "curriculum.md"
        curriculum_md.write_text(
            "# Civil Society Seminar\n\n"
            "George Frederickson taught Concepts of Civil Society.\n"
            "Raymond Davis provided advice and guidance.\n",
            encoding="utf-8",
        )

        # 1. Base drop (2 sources, verified)
        rep_base = batch_drop(
            [str(overview_html), str(curriculum_md)],
            store_dir=store_dir,
            reliability="verified",
            filter_name="standard",
            coref=True,
        )

        # 2. Incremental delta: update curriculum.md, add extensions.html, keep overview.html unchanged
        curriculum_md.write_text(
            "# Civil Society Seminar\n\n"
            "George Frederickson taught Concepts of Civil Society.\n"
            "Charles Carlsen was President of Johnson County Community College.\n",
            encoding="utf-8",
        )
        extensions_html = docs_dir / "extensions.html"
        extensions_html.write_text(
            "<html><head><title>Scholarly Influences</title></head><body><main>\n"
            "<p>Robert Putnam wrote Bowling Alone.</p>\n"
            "<p>Johnson County Community College developed a pilot program.</p>\n"
            "</main></body></html>\n",
            encoding="utf-8",
        )
        rep_delta = batch_drop(
            [str(overview_html), str(curriculum_md), str(extensions_html)],
            store_dir=store_dir,
            reliability="likely_true",
            filter_name="standard",
            coref=True,
        )

        # 3. Disputed overlay (lower reliability)
        disputed_md = docs_dir / "disputed_blog.md"
        disputed_md.write_text(
            "# Unverified Notes\n\nDeborah DeGrate and Chris Engle graduated with civic honors.\n",
            encoding="utf-8",
        )
        rep_disputed = batch_drop(
            [str(disputed_md)],
            store_dir=store_dir,
            reliability="possibly_true",
            filter_name="standard",
            coref=True,
        )

        store = KnowledgeStore(store_dir)
        drops = store.list_drops()
        facts = list(store.iter_facts())

        with Catalog(os.path.join(store_dir, "catalog.sqlite")) as cat:
            catalog_indexed = cat.rebuild(store)
            reliable_rows = cat.query(min_reliability="LIKELY_TRUE")

        # 4. Ingest reliable facts + 3 explicit lower-reliability conflicting claims into graph store
        graph_store = InMemoryGraphStore()
        graph_store.ingest_facts(facts)
        disputed_claims = [
            {
                "subject": "Charles Carlsen",
                "predicate": "not_president_of",
                "object": "Johnson County Community College",
                "fact_statement": "Charles Carlsen was not President of Johnson County Community College.",
                "reliability_rating": "POSSIBLY_TRUE",
                "quality_score": 20,
                "category": "Leadership",
            },
            {
                "subject": "Graduation With Civic Honors",
                "predicate": "not_published",
                "object": "2006",
                "fact_statement": "Graduation With Civic Honors was not published in 2006.",
                "reliability_rating": "POSSIBLY_TRUE",
                "quality_score": 20,
                "category": "History",
            },
            {
                "subject": "Johnson County Community College",
                "predicate": "did_not_develop",
                "object": "pilot program",
                "fact_statement": "Johnson County Community College did not develop a pilot program.",
                "reliability_rating": "POSSIBLY_TRUE",
                "quality_score": 20,
                "category": "Education",
            },
        ]
        graph_store.ingest_facts(disputed_claims)
        graph_store.auto_link_relations()

        conflicts = graph_store.find_contradictions()
        contradiction_insts = compile_contradiction_instructions(graph_store)
        reconcile_report = graph_store.validate_and_reconcile()
        demoted = reconcile_report.get("demoted", [])

        # Confirm disputed claims were demoted to UNVERIFIED
        demoted_stmts = {str(d.get("statement") or "") for d in demoted}
        expected_demoted = {c["fact_statement"] for c in disputed_claims}
        res_acc = 1.0 if expected_demoted.issubset(demoted_stmts) else 0.0
        det_recall = 1.0 if len(conflicts) >= 3 else len(conflicts) / 3.0

        # 5. GraphRAG retrieval after reconciliation
        retriever = GraphRAGRetriever(store=graph_store)
        qa_checks = [
            (
                "Who was President of Johnson County Community College?",
                "charles carlsen",
                "johnson county community college",
                "not president",
            ),
            (
                "When was Graduation With Civic Honors published?",
                "graduation with civic honors",
                "2006",
                "not published",
            ),
            (
                "What did Johnson County Community College develop?",
                "johnson county community college",
                "pilot program",
                "did not develop",
            ),
        ]
        hit_at_1_count = 0
        suppressed = True
        for query_text, req1, req2, forbidden in qa_checks:
            hits = retriever.retrieve(query_text, top_k=3, exclude_unverified=True)
            if hits:
                top_stmt = str(hits[0].get("statement") or "").lower()
                if req1 in top_stmt and req2 in top_stmt and forbidden not in top_stmt:
                    hit_at_1_count += 1
            if any(forbidden in str(h.get("statement") or "").lower() for h in hits):
                suppressed = False
        graph_store.close()

        # 6. Export valid chat JSONL
        chat_lines = chat_jsonl_from_facts(facts)
        valid_chat = sum(1 for ln in chat_lines if validate_chat_record(json.loads(ln))[0])

    errors = (
        int(rep_base.get("errors", 0))
        + int(rep_delta.get("errors", 0))
        + int(rep_disputed.get("errors", 0))
    )
    elapsed = round(time.perf_counter() - t_start, 4)

    metrics = MultiSourceFactoryMetrics(
        base_sources=int(rep_base.get("dropped", 0)),
        delta_sources=int(rep_delta.get("dropped", 0)),
        skipped_unchanged=int(rep_delta.get("skipped", 0)),
        disputed_sources=int(rep_disputed.get("dropped", 0)),
        total_drops=len(drops),
        total_facts=len(facts),
        catalog_indexed=catalog_indexed,
        catalog_reliable_count=len(reliable_rows),
        conflicts_detected=len(conflicts),
        conflicts_demoted=len(demoted),
        conflict_detection_recall=round(det_recall, 4),
        conflict_resolution_accuracy=round(res_acc, 4),
        retrieval_hit_at_1=round(hit_at_1_count / len(qa_checks), 4),
        unverified_suppressed=suppressed,
        contradiction_instructions=len(contradiction_insts),
        chat_records=len(chat_lines),
        valid_chat_ratio=round(valid_chat / max(1, len(chat_lines)), 4),
        errors=errors,
        total_seconds=elapsed,
    )
    passed, failures = _check_floors(metrics)
    return MultiSourceFactoryResult(
        metrics=metrics,
        fact_statements=sorted(str(f["fact_statement"]) for f in facts),
        chat_jsonl_lines=chat_lines,
        passed_floors=passed,
        floor_failures=failures,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Multi-source incremental factory & reconciliation golden benchmark"
    )
    parser.add_argument(
        "--strict-regression",
        action="store_true",
        help="Exit non-zero if any regression floor fails",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    args = parser.parse_args(argv)

    result = evaluate_multisource_factory()
    if args.json:
        print(
            json.dumps(
                {
                    "metrics": result.metrics.to_dict(),
                    "passed_floors": result.passed_floors,
                    "floor_failures": result.floor_failures,
                },
                indent=2,
            )
        )
    else:
        m = result.metrics
        print(
            f"[MultiSource Golden] drops={m.total_drops} skipped={m.skipped_unchanged} "
            f"conflicts={m.conflicts_detected} demoted={m.conflicts_demoted} "
            f"hit@1={m.retrieval_hit_at_1} time={m.total_seconds}s passed={result.passed_floors}"
        )
        if result.floor_failures:
            print(f"[MultiSource Golden] FAILURES: {result.floor_failures}", file=sys.stderr)

    if args.strict_regression and not result.passed_floors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
