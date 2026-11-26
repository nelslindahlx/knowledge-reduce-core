"""Golden regression tests for the offline 'Civic Honors: The Book' benchmark corpus."""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import tempfile

import pytest

from benchmarks.civichonors_book_benchmark import (
    BASELINE_V04_PATH,
    SOURCE_DOC_PATH,
    BookEvaluationResult,
    compare_against_baseline,
    evaluate_civichonors_book,
    load_baseline,
    load_book_text,
    load_gold_annotations,
)
from knowledge_graph_pkg.graph_tool import GraphTools
from knowledge_graph_pkg.ontology import OntologyDistiller
from knowledge_graph_pkg.rag import GraphRAGRetriever

KUZU_AVAILABLE = importlib.util.find_spec("kuzu") is not None


@pytest.fixture(scope="module")
def book_eval() -> BookEvaluationResult:
    """Evaluate the frozen Civic Honors book corpus once per test module."""
    return evaluate_civichonors_book()


def test_golden_corpus_snapshot_integrity() -> None:
    raw_text = SOURCE_DOC_PATH.read_text(encoding="utf-8")
    assert ("/" + "Users/") not in raw_text
    assert ("file://" + "/") not in raw_text

    text = load_book_text()
    gold = load_gold_annotations()
    assert text == raw_text
    assert len(text) == 139_595
    assert len(text.split()) == 21_765
    assert gold["source_path"] == "data/civic_honors_book.txt"
    assert len(gold["reference_facts"]) >= 10
    assert len(gold["retrieval_queries"]) >= 10


def test_civichonors_book_strict_invariants(book_eval: BookEvaluationResult) -> None:
    m = book_eval.metrics
    assert m.idempotent_skipped == 1
    assert m.distilled_facts >= 60
    assert m.catalog_indexed == m.distilled_facts
    assert m.valid_chat_records == m.distilled_facts
    assert m.valid_chat_ratio == 1.0
    assert m.heading_noise_fp_count == 0
    assert m.question_fact_count == 0
    assert m.core_fact_retention == 1.0
    assert m.missing_reference_facts == []
    assert m.ontology_categories > 0
    assert bool(book_eval.ontology.get("semantic_types"))
    assert book_eval.warm_query_top1_matches_cold is True


def test_civichonors_book_meets_v0_4_0_baseline_floors(
    book_eval: BookEvaluationResult,
) -> None:
    assert BASELINE_V04_PATH.is_file()
    baseline = load_baseline(BASELINE_V04_PATH)
    comparison = compare_against_baseline(book_eval.metrics, baseline)
    assert comparison["baseline_version"] == "0.4.0"
    assert comparison["passed_floors"] is True, comparison["floor_failures"]
    assert comparison["floor_failures"] == []
    assert book_eval.metrics.retrieval_hit_at_1 >= 0.90
    assert book_eval.metrics.retrieval_mrr_at_5 >= 0.90


def test_civichonors_book_deterministic_reproducibility(
    book_eval: BookEvaluationResult,
) -> None:
    reproduced = evaluate_civichonors_book()
    assert [
        (f.get("subject"), f.get("predicate"), f.get("object"), f.get("fact_statement"))
        for f in reproduced.facts
    ] == [
        (f.get("subject"), f.get("predicate"), f.get("object"), f.get("fact_statement"))
        for f in book_eval.facts
    ]
    assert reproduced.chat_jsonl_lines == book_eval.chat_jsonl_lines
    assert reproduced.metrics.query_ranks == book_eval.metrics.query_ranks


def test_baseline_comparison_detects_regressions_and_improvements(
    book_eval: BookEvaluationResult,
) -> None:
    baseline = load_baseline()

    improved = dataclasses.replace(
        book_eval.metrics,
        core_entity_name_recall=0.95,
        ontology_categories=book_eval.metrics.ontology_categories + 5,
    )
    imp_cmp = compare_against_baseline(improved, baseline)
    assert imp_cmp["passed_floors"] is True
    assert imp_cmp["deltas"]["core_entity_name_recall"]["status"] == "improved"
    assert imp_cmp["deltas"]["ontology_categories"]["status"] == "improved"

    regressed = dataclasses.replace(
        book_eval.metrics,
        core_fact_retention=0.50,
        heading_noise_fp_count=4,
    )
    reg_cmp = compare_against_baseline(regressed, baseline)
    assert reg_cmp["passed_floors"] is False
    assert reg_cmp["deltas"]["core_fact_retention"]["status"] == "regressed"
    assert reg_cmp["deltas"]["heading_noise_fp_count"]["status"] == "regressed"
    assert any("core_fact_retention" in msg for msg in reg_cmp["floor_failures"])
    assert any("heading_noise_fp_count" in msg for msg in reg_cmp["floor_failures"])


@pytest.mark.skipif(not KUZU_AVAILABLE, reason="kuzu extra not installed")
def test_civichonors_book_kuzu_store_parity(book_eval: BookEvaluationResult) -> None:
    from knowledge_graph_pkg.kuzu_store import KuzuStore

    gold = load_gold_annotations()
    with tempfile.TemporaryDirectory(prefix="kr_book_kuzu_") as tmp_dir:
        kstore = KuzuStore(os.path.join(tmp_dir, "kuzu.db"))
        try:
            ingested = kstore.ingest_facts(book_eval.facts)
            assert ingested == len(book_eval.facts)
            kstore.auto_link_relations()
            ont = OntologyDistiller(kstore).distill_all()
            assert len(ont.get("semantic_types", {})) == book_eval.metrics.ontology_categories

            tools = GraphTools(kstore)
            top_rows = tools.graph_query("MATCH (f:Fact) RETURN f.statement AS statement LIMIT 5")
            assert len(top_rows) == 5

            retriever = GraphRAGRetriever(store=kstore)
            first_q = gold["retrieval_queries"][0]
            hits = retriever.retrieve(first_q["query"], top_k=5)
            assert hits
            req = [s.lower() for s in first_q["required_evidence_substrings"]]
            assert any(
                all(term in str(h.get("statement") or "").lower() for term in req) for h in hits
            )
        finally:
            kstore.close()
