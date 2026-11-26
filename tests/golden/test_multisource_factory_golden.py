"""Golden regression tests for offline multi-source incremental factory & GraphRAG."""

from __future__ import annotations

import importlib.util
import os
import tempfile

import pytest

from benchmarks.multisource_factory_benchmark import (
    MultiSourceFactoryResult,
    evaluate_multisource_factory,
)

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None


@pytest.fixture(scope="module")
def multisource_eval() -> MultiSourceFactoryResult:
    """Evaluate the deterministic multi-source factory pipeline once per module."""
    return evaluate_multisource_factory()


def test_multisource_incremental_idempotency_and_catalog(
    multisource_eval: MultiSourceFactoryResult,
) -> None:
    m = multisource_eval.metrics
    assert m.errors == 0
    assert m.base_sources == 2
    assert m.delta_sources == 2
    assert m.skipped_unchanged == 1
    assert m.disputed_sources == 1
    assert m.total_drops == 5
    assert m.catalog_indexed == m.total_facts
    assert m.catalog_reliable_count > 0
    assert m.catalog_reliable_count < m.catalog_indexed


def test_multisource_conflicts_reconciliation_and_retrieval(
    multisource_eval: MultiSourceFactoryResult,
) -> None:
    m = multisource_eval.metrics
    assert multisource_eval.passed_floors is True, multisource_eval.floor_failures
    assert m.conflicts_detected >= 3
    assert m.conflicts_demoted >= 3
    assert m.conflict_detection_recall == 1.0
    assert m.conflict_resolution_accuracy == 1.0
    assert m.retrieval_hit_at_1 == 1.0
    assert m.unverified_suppressed is True
    assert m.contradiction_instructions >= 3


def test_multisource_sft_export_and_reproducibility(
    multisource_eval: MultiSourceFactoryResult,
) -> None:
    m = multisource_eval.metrics
    assert m.chat_records == len(multisource_eval.chat_jsonl_lines)
    assert 1 <= m.chat_records <= m.total_facts
    assert m.valid_chat_ratio == 1.0

    reproduced = evaluate_multisource_factory()
    assert reproduced.fact_statements == multisource_eval.fact_statements
    assert reproduced.chat_jsonl_lines == multisource_eval.chat_jsonl_lines
    assert reproduced.metrics.conflicts_detected == m.conflicts_detected
    assert reproduced.metrics.conflicts_demoted == m.conflicts_demoted


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu extra not installed")
def test_multisource_kuzu_store_conflict_parity() -> None:
    from knowledge_graph_pkg.graph_compiler import compile_contradiction_instructions
    from knowledge_graph_pkg.kuzu_store import KuzuStore
    from knowledge_graph_pkg.rag import GraphRAGRetriever

    facts = [
        {
            "subject": "Graduation With Civic Honors",
            "predicate": "published",
            "object": "2006",
            "fact_statement": "Graduation With Civic Honors was published in 2006.",
            "reliability_rating": "VERIFIED",
            "quality_score": 90,
        },
        {
            "subject": "Graduation With Civic Honors",
            "predicate": "not_published",
            "object": "2006",
            "fact_statement": "Graduation With Civic Honors was not published in 2006.",
            "reliability_rating": "POSSIBLY_TRUE",
            "quality_score": 20,
        },
    ]
    with tempfile.TemporaryDirectory(prefix="kr_ms_kuzu_") as tmp_dir:
        kstore = KuzuStore(os.path.join(tmp_dir, "kuzu.db"))
        try:
            kstore.ingest_facts(facts)
            conflicts = kstore.find_contradictions()
            assert len(conflicts) == 1
            insts = compile_contradiction_instructions(kstore)
            assert len(insts) == 1
            reconciled = kstore.validate_and_reconcile()
            assert len(reconciled["demoted"]) == 1

            hits = GraphRAGRetriever(kstore).retrieve(
                "When was Graduation With Civic Honors published?",
                top_k=2,
                exclude_unverified=True,
            )
            assert len(hits) == 1
            assert "not published" not in hits[0]["statement"].lower()
        finally:
            kstore.close()
