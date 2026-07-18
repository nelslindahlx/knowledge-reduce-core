"""
Tests for Session 5 training-data polish:
- train/validation split
- token-budget truncation
- cross-run dedup via a persistent global fact store
"""
import json
from pathlib import Path

import pytest

from knowledge_graph_pkg import KnowledgeGraph, SemanticKnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.export import split_records, budget_records, estimate_tokens
from knowledge_graph_pkg.factstore import FactStore


# ---------- train/val split ----------

def test_split_ratio_partitions_all_records():
    records = [f"line {i}" for i in range(10)]
    train, val = split_records(records, ratio=0.8, seed=42)
    assert len(train) == 8
    assert len(val) == 2
    # no loss, no duplication
    assert sorted(train + val) == sorted(records)


def test_split_is_deterministic_with_seed():
    records = [f"r{i}" for i in range(20)]
    a = split_records(records, ratio=0.9, seed=7)
    b = split_records(records, ratio=0.9, seed=7)
    assert a == b


def test_split_full_train_when_ratio_one():
    records = ["a", "b", "c"]
    train, val = split_records(records, ratio=1.0, seed=1)
    assert len(train) == 3 and len(val) == 0


# ---------- token budget ----------

def test_estimate_tokens_grows_with_length():
    assert estimate_tokens("a" * 4) < estimate_tokens("a" * 400)


def test_budget_keeps_under_limit():
    records = ["x" * 40 for _ in range(100)]  # ~10 tokens each
    kept = budget_records(records, max_tokens=50)
    assert len(kept) < len(records)
    total = sum(estimate_tokens(r) for r in kept)
    assert total <= 50


def test_budget_returns_all_when_under_limit():
    records = ["short", "lines"]
    kept = budget_records(records, max_tokens=10_000)
    assert kept == records


# ---------- cross-run dedup via FactStore ----------

def test_factstore_dedups_within_run():
    store = FactStore()
    assert store.add("Marie Curie discovered radium.") is True
    assert store.add("Marie Curie discovered radium.") is False  # duplicate
    assert store.add("Water boils at 100 C.") is True
    assert len(store) == 2


def test_factstore_persists_across_runs(tmp_path):
    path = tmp_path / "seen.json"
    s1 = FactStore(path=str(path))
    s1.add("Marie Curie discovered radium.")
    s1.save()

    # New store loading the same file should treat the fact as already seen.
    s2 = FactStore(path=str(path))
    s2.load()
    assert s2.add("Marie Curie discovered radium.") is False
    assert s2.add("A brand new fact.") is True


def test_factstore_normalizes_whitespace_and_case():
    store = FactStore()
    store.add("Marie Curie discovered radium.")
    # case + spacing variation should still count as a duplicate
    assert store.add("  marie   curie discovered RADIUM. ") is False
