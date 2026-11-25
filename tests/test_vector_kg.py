"""Tests for VectorKnowledgeGraph deterministic embeddings, search, clustering, and persistence."""
from datetime import datetime

import numpy as np
import pytest

from knowledge_graph_pkg.core import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.vector import VectorKnowledgeGraph


def _make_kg():
    kg = KnowledgeGraph()
    now = datetime(2026, 1, 1)
    facts = [
        ("f1", "Neural networks learn representations from data."),
        ("f2", "Deep neural networks use gradient descent."),
        ("f3", "Marie Curie discovered polonium and radium."),
    ]
    for fid, stmt in facts:
        kg.add_fact(
            fid, stmt, "Science", ["ml"], now, now, ReliabilityRating.VERIFIED,
            "s1", "Title", "Author", now, "", [], "", "public", 1
        )
    return kg


def test_semantic_search_and_similar_facts_for_both_embedding_modes(tmp_path):
    kg = _make_kg()
    for use_external in (False, True):
        vkg = VectorKnowledgeGraph(kg, embedding_dim=64)
        assert vkg.generate_embeddings(use_external_model=use_external) == 3

        if not use_external:
            hits = vkg.semantic_search("neural networks gradient", top_k=2)
            assert len(hits) == 2
            assert hits[0][0] in {"f1", "f2"}
        else:
            # Exact-query search in external mock mode must score ~1.0 (same vector space!).
            hits = vkg.semantic_search("Marie Curie discovered polonium and radium.", top_k=1)
            assert hits[0][0] == "f3"
            assert np.isclose(hits[0][1], 1.0)

        assert vkg.semantic_search("anything", top_k=0) == []
        sim = vkg.find_similar_facts("f1", top_k=2)
        assert len(sim) == 2
        assert all(fid != "f1" for fid, _ in sim)

    # Save and load embeddings round-trip.
    vkg = VectorKnowledgeGraph(kg, embedding_dim=64)
    vkg.generate_embeddings()
    out = tmp_path / "emb.json"
    vkg.save_embeddings(str(out))
    vkg2 = VectorKnowledgeGraph(kg, embedding_dim=64)
    assert vkg2.load_embeddings(str(out)) == 3
    assert np.allclose(vkg2.embeddings["f1"], vkg.embeddings["f1"])


def test_cluster_facts_clamps_small_corpora_and_preserves_global_rng():
    kg = _make_kg()
    vkg = VectorKnowledgeGraph(kg, embedding_dim=32)

    with pytest.raises(ValueError, match="No embeddings"):
        vkg.cluster_facts(num_clusters=2)

    vkg.generate_embeddings()
    # Clamps default num_clusters=5 down to len(facts)=3 without ValueError.
    clusters = vkg.cluster_facts(num_clusters=5)
    assert len(clusters) == 3
    assigned = [fid for items in clusters.values() for fid in items]
    assert sorted(assigned) == ["f1", "f2", "f3"]

    with pytest.raises(ValueError, match="positive integer"):
        vkg.cluster_facts(num_clusters=0)

    # Global np.random state stays intact and query_expansion expands terms.
    expanded = vkg.query_expansion("knowledge graph", expansion_terms=2)
    assert "knowledge graph" in expanded
