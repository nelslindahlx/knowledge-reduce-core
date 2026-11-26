"""Offline real-world golden evaluation benchmark for 'Civic Honors: The Book'."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks._in_memory_store import InMemoryGraphStore, chat_jsonl_from_facts  # noqa: E402
from knowledge_graph_pkg.catalog import Catalog  # noqa: E402
from knowledge_graph_pkg.core import KnowledgeGraph  # noqa: E402
from knowledge_graph_pkg.extraction import SVOExtractor  # noqa: E402
from knowledge_graph_pkg.factory import batch_drop  # noqa: E402
from knowledge_graph_pkg.ontology import OntologyDistiller  # noqa: E402
from knowledge_graph_pkg.rag import GraphRAGRetriever  # noqa: E402
from knowledge_graph_pkg.semantic import SemanticKnowledgeGraph  # noqa: E402
from knowledge_graph_pkg.store import KnowledgeStore  # noqa: E402
from knowledge_graph_pkg.training_prep import validate_chat_record  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "civichonors_book"
SOURCE_DOC_PATH = REPO_ROOT / "data" / "civic_honors_book.txt"
GOLD_ANNOTATIONS_PATH = DATASET_DIR / "gold_annotations.json"
BASELINE_V04_PATH = DATASET_DIR / "baseline_v0_4_0.json"

DEFAULT_REGRESSION_FLOORS: dict[str, dict[str, float | int]] = {
    "min": {
        "distilled_facts": 60,
        "idempotent_skipped": 1,
        "valid_chat_ratio": 1.0,
        "core_entity_name_recall": 0.40,
        "core_fact_retention": 0.95,
        "ontology_categories": 1,
        "retrieval_hit_at_1": 0.90,
        "retrieval_hit_at_5": 0.90,
        "retrieval_mrr_at_5": 0.90,
    },
    "max": {
        "heading_noise_fp_count": 0,
        "question_fact_count": 0,
        "total_seconds": 8.0,
    },
}

_HIGHER_IS_BETTER: tuple[str, ...] = (
    "distilled_facts",
    "idempotent_skipped",
    "valid_chat_ratio",
    "entity_mentions",
    "core_entity_name_recall",
    "core_fact_retention",
    "ontology_categories",
    "retrieval_hit_at_1",
    "retrieval_hit_at_3",
    "retrieval_hit_at_5",
    "retrieval_mrr_at_5",
)

_LOWER_IS_BETTER: tuple[str, ...] = (
    "heading_noise_fp_count",
    "question_fact_count",
)


def _norm(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass(frozen=True)
class BookGoldenMetrics:
    """Comprehensive baseline & regression metrics on the frozen Civic Honors book."""

    corpus_id: str
    source_chars: int
    source_words: int
    content_sha256: str
    raw_facts: int
    distilled_facts: int
    compression_ratio: float
    idempotent_skipped: int
    catalog_indexed: int
    valid_chat_records: int
    valid_chat_ratio: float
    entity_mentions: int
    core_entity_name_recall: float
    heading_noise_fp_count: int
    question_fact_count: int
    core_fact_retention: float
    graph_nodes: int
    graph_edges: int
    ontology_categories: int
    retrieval_hit_at_1: float
    retrieval_hit_at_3: float
    retrieval_hit_at_5: float
    retrieval_mrr_at_5: float
    query_ranks: dict[str, int | None]
    missing_reference_facts: list[str]
    drop_seconds: float
    catalog_seconds: float
    export_seconds: float
    graph_seconds: float
    query_cold_seconds: float
    query_warm_avg_seconds: float
    total_seconds: float
    peak_traced_mib: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)


@dataclass
class BookEvaluationResult:
    """Full evaluation output bundling metrics, facts, chat JSONL, and ontology."""

    metrics: BookGoldenMetrics
    facts: list[dict[str, Any]]
    chat_jsonl_lines: list[str]
    ontology: dict[str, Any]
    warm_query_top1_matches_cold: bool


def load_book_text(path: Path = SOURCE_DOC_PATH) -> str:
    """Load the frozen Civic Honors book corpus and return its UTF-8 text."""
    return path.read_text(encoding="utf-8")


def load_gold_annotations(path: Path = GOLD_ANNOTATIONS_PATH) -> dict[str, Any]:
    """Load curated entities, noise canaries, reference facts, and QA queries."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def load_baseline(path: Path = BASELINE_V04_PATH) -> dict[str, Any]:
    """Load the locked v0.4.0 baseline snapshot and regression floors."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return data


def evaluate_civichonors_book(*, top_k: int = 5) -> BookEvaluationResult:
    """Run batch_drop -> idempotency -> Catalog -> Export -> GraphRAG on the book."""
    raw_text = load_book_text()
    gold = load_gold_annotations()
    expected_sha = str(gold.get("content_sha256") or "")
    actual_sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    if expected_sha and actual_sha != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for {SOURCE_DOC_PATH}: expected {expected_sha}, got {actual_sha}."
        )

    queries: list[dict[str, Any]] = list(gold.get("retrieval_queries", []))
    if not queries:
        raise ValueError("Gold annotations contain no retrieval_queries.")

    started_here = not tracemalloc.is_tracing()
    if started_here:
        tracemalloc.start()

    try:
        raw_count = len(SVOExtractor().extract(raw_text))
        entity_mentions = SemanticKnowledgeGraph(KnowledgeGraph()).extract_entities_from_text(
            raw_text
        )

        with tempfile.TemporaryDirectory(prefix="kr_book_golden_") as tmp_dir:
            store_dir = os.path.join(tmp_dir, "store")

            t0 = time.perf_counter()
            batch_drop(
                [str(SOURCE_DOC_PATH)],
                store_dir=store_dir,
                reliability="likely_true",
                filter_name="standard",
                coref=True,
                engine="svo",
            )
            t_drop = time.perf_counter() - t0

            repeat_report = batch_drop(
                [str(SOURCE_DOC_PATH)],
                store_dir=store_dir,
                reliability="likely_true",
                filter_name="standard",
                coref=True,
                engine="svo",
            )
            idempotent_skipped = int(repeat_report.get("skipped", 0))

            store = KnowledgeStore(store_dir)
            facts = list(store.iter_facts())

            t1 = time.perf_counter()
            with Catalog(os.path.join(store_dir, "catalog.sqlite")) as cat:
                catalog_indexed = cat.rebuild(store)
            t_catalog = time.perf_counter() - t1

        t2 = time.perf_counter()
        chat_lines = chat_jsonl_from_facts(facts)
        valid_chat_records = sum(1 for ln in chat_lines if validate_chat_record(json.loads(ln))[0])
        t_export = time.perf_counter() - t2

        t3 = time.perf_counter()
        graph_store = InMemoryGraphStore()
        graph_store.ingest_facts(facts)
        graph_edges = graph_store.auto_link_relations()
        ontology = OntologyDistiller(graph_store).distill_all()
        retriever = GraphRAGRetriever(store=graph_store)
        t_graph = time.perf_counter() - t3

        first_query = str(queries[0]["query"])
        t4 = time.perf_counter()
        cold_hits = retriever.retrieve(first_query, top_k=top_k)
        t_cold = time.perf_counter() - t4
        cold_top1_id = cold_hits[0].get("block_id") if cold_hits else None

        query_ranks: dict[str, int | None] = {}
        warm_durations: list[float] = []
        warm_query_top1_matches_cold = True

        for idx_q, q_spec in enumerate(queries):
            q_id = str(q_spec["query_id"])
            q_text = str(q_spec["query"])
            req_terms = [str(term).lower() for term in q_spec["required_evidence_substrings"]]

            tq0 = time.perf_counter()
            hits = retriever.retrieve(q_text, top_k=top_k)
            warm_durations.append(time.perf_counter() - tq0)

            if idx_q == 0:
                warm_top1_id = hits[0].get("block_id") if hits else None
                warm_query_top1_matches_cold = warm_top1_id == cold_top1_id

            rank: int | None = None
            for pos, hit in enumerate(hits, start=1):
                stmt_lower = str(hit.get("statement") or "").lower()
                if all(term in stmt_lower for term in req_terms):
                    rank = pos
                    break
            query_ranks[q_id] = rank

        graph_nodes = graph_store.count()
        graph_store.close()
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if started_here and tracemalloc.is_tracing():
            tracemalloc.stop()

    ent_texts = {_norm(str(e.get("text") or "")) for e in entity_mentions}
    expected_entities: list[dict[str, Any]] = list(gold.get("expected_entities", []))
    entity_hits = sum(
        1
        for spec in expected_entities
        if any(_norm(str(spec["canonical_name"])) in t for t in ent_texts)
    )

    stmts_lower = [str(f.get("fact_statement") or "").lower() for f in facts]
    subjs_objs = [
        (_norm(str(f.get("subject") or "")), _norm(str(f.get("object") or ""))) for f in facts
    ]
    noise_candidates: list[str] = list(gold.get("known_heading_noise_entities", []))
    heading_noise_fp_count = sum(
        1
        for candidate in noise_candidates
        if any(
            _norm(candidate) in {subj, obj, _norm(stmt)}
            for (subj, obj), stmt in zip(subjs_objs, stmts_lower)
        )
    )
    question_fact_count = sum(
        1 for f in facts if str(f.get("fact_statement") or "").rstrip().endswith("?")
    )

    reference_facts: list[dict[str, Any]] = list(gold.get("reference_facts", []))
    missing_reference_facts: list[str] = []
    for rf in reference_facts:
        req_terms = [str(t).lower() for t in rf["required_substrings"]]
        if not any(all(term in stmt for term in req_terms) for stmt in stmts_lower):
            missing_reference_facts.append(str(rf["fact_id"]))
    retained = len(reference_facts) - len(missing_reference_facts)

    num_queries = len(queries)
    hit_at_1 = sum(1 for r in query_ranks.values() if r == 1) / num_queries
    hit_at_3 = sum(1 for r in query_ranks.values() if r is not None and r <= 3) / num_queries
    hit_at_5 = sum(1 for r in query_ranks.values() if r is not None and r <= 5) / num_queries
    mrr_at_5 = sum((1.0 / r) for r in query_ranks.values() if r is not None) / num_queries

    t_warm_avg = sum(warm_durations) / len(warm_durations)
    total_seconds = t_drop + t_catalog + t_export + t_graph + t_cold

    metrics = BookGoldenMetrics(
        corpus_id=str(gold.get("corpus_id", "civichonors-book-v1")),
        source_chars=len(raw_text),
        source_words=len(raw_text.split()),
        content_sha256=actual_sha,
        raw_facts=raw_count,
        distilled_facts=len(facts),
        compression_ratio=round(1.0 - len(facts) / max(1, raw_count), 4),
        idempotent_skipped=idempotent_skipped,
        catalog_indexed=catalog_indexed,
        valid_chat_records=valid_chat_records,
        valid_chat_ratio=round(valid_chat_records / max(1, len(chat_lines)), 4),
        entity_mentions=len(entity_mentions),
        core_entity_name_recall=round(entity_hits / max(1, len(expected_entities)), 4),
        heading_noise_fp_count=heading_noise_fp_count,
        question_fact_count=question_fact_count,
        core_fact_retention=round(retained / max(1, len(reference_facts)), 4),
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        ontology_categories=len(ontology.get("semantic_types", {})),
        retrieval_hit_at_1=round(hit_at_1, 4),
        retrieval_hit_at_3=round(hit_at_3, 4),
        retrieval_hit_at_5=round(hit_at_5, 4),
        retrieval_mrr_at_5=round(mrr_at_5, 4),
        query_ranks=query_ranks,
        missing_reference_facts=missing_reference_facts,
        drop_seconds=round(t_drop, 4),
        catalog_seconds=round(t_catalog, 4),
        export_seconds=round(t_export, 4),
        graph_seconds=round(t_graph, 4),
        query_cold_seconds=round(t_cold, 4),
        query_warm_avg_seconds=round(t_warm_avg, 4),
        total_seconds=round(total_seconds, 4),
        peak_traced_mib=round(peak_bytes / (1024 * 1024), 2),
    )
    return BookEvaluationResult(
        metrics=metrics,
        facts=facts,
        chat_jsonl_lines=chat_lines,
        ontology=ontology,
        warm_query_top1_matches_cold=warm_query_top1_matches_cold,
    )


def compare_against_baseline(
    metrics: BookGoldenMetrics, baseline: dict[str, Any]
) -> dict[str, Any]:
    """Compare benchmark metrics against a locked baseline snapshot and regression floors."""
    cur = metrics.to_dict()
    base_metrics: dict[str, Any] = dict(baseline.get("metrics", {}))
    floors: dict[str, dict[str, float | int]] = dict(
        baseline.get("regression_floors", DEFAULT_REGRESSION_FLOORS)
    )

    floor_failures: list[str] = []
    for metric_name, min_val in floors.get("min", {}).items():
        if float(cur[metric_name]) < float(min_val):
            floor_failures.append(f"{metric_name}={cur[metric_name]} < min {min_val}")
    for metric_name, max_val in floors.get("max", {}).items():
        if float(cur[metric_name]) > float(max_val):
            floor_failures.append(f"{metric_name}={cur[metric_name]} > max {max_val}")

    deltas: dict[str, dict[str, Any]] = {}
    for key in (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER):
        if key not in base_metrics:
            continue
        bval = float(base_metrics[key])
        cval = float(cur[key])
        diff = round(cval - bval, 4)
        status = "unchanged"
        if abs(diff) > 1e-6:
            higher = key in _HIGHER_IS_BETTER
            status = "improved" if (diff > 0 if higher else diff < 0) else "regressed"
        deltas[key] = {
            "baseline": base_metrics[key],
            "current": cur[key],
            "delta": diff,
            "status": status,
        }

    return {
        "baseline_version": str(baseline.get("baseline_version", "unknown")),
        "passed_floors": len(floor_failures) == 0,
        "floor_failures": floor_failures,
        "deltas": deltas,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Civic Honors book golden regression benchmark")
    parser.add_argument(
        "--strict-regression",
        action="store_true",
        help="Exit non-zero if any regression floor fails",
    )
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Update baseline_v0_4_0.json from current verified run",
    )
    args = parser.parse_args(argv)

    result = evaluate_civichonors_book()
    if args.write_baseline:
        payload = {
            "baseline_version": "0.4.0",
            "corpus_id": result.metrics.corpus_id,
            "source_path": "data/civic_honors_book.txt",
            "metrics": result.metrics.to_dict(),
            "regression_floors": DEFAULT_REGRESSION_FLOORS,
        }
        BASELINE_V04_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    baseline = load_baseline() if BASELINE_V04_PATH.is_file() else {}
    comparison = compare_against_baseline(result.metrics, baseline)

    if args.json:
        print(
            json.dumps(
                {
                    "metrics": result.metrics.to_dict(),
                    "comparison": comparison,
                    "warm_query_top1_matches_cold": result.warm_query_top1_matches_cold,
                },
                indent=2,
            )
        )
    else:
        m = result.metrics
        print(
            f"[CivicHonors Golden] facts={m.distilled_facts}/{m.raw_facts} "
            f"retention={m.core_fact_retention} hit@1={m.retrieval_hit_at_1} "
            f"mrr@5={m.retrieval_mrr_at_5} noise={m.heading_noise_fp_count} "
            f"time={m.total_seconds}s passed={comparison['passed_floors']}"
        )
        if comparison["floor_failures"]:
            print(f"[CivicHonors Golden] FAILURES: {comparison['floor_failures']}", file=sys.stderr)

    if args.strict_regression and not comparison["passed_floors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
