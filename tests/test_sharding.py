"""Tests for ShardedKnowledgeGraph persistence, queries, and shard optimization."""
from datetime import datetime

import pytest

from knowledge_graph_pkg.core import ReliabilityRating
from knowledge_graph_pkg.sharding import ShardedKnowledgeGraph


def _add(skg: ShardedKnowledgeGraph, fact_id: str, stmt: str, category="General", rating=ReliabilityRating.VERIFIED):
    now = datetime(2026, 1, 1)
    return skg.add_fact(
        fact_id=fact_id,
        fact_statement=stmt,
        category=category,
        tags=["tag"],
        date_recorded=now,
        last_updated=now,
        reliability_rating=rating,
        source_id="src",
        source_title="Title",
        author_creator="Author",
        publication_date=now,
        url_reference="https://example.com",
        related_facts=[],
        contextual_notes="",
        access_level="public",
        usage_count=1,
    )


def test_invalid_shard_size_raises(tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        ShardedKnowledgeGraph(str(tmp_path / "shards"), shard_size=0)


def test_crud_query_reload_and_optimize_deletes_stale_shards(tmp_path):
    shard_dir = tmp_path / "shards"
    skg = ShardedKnowledgeGraph(str(shard_dir), shard_size=2)

    for i in range(5):
        _add(skg, f"f{i}", f"Statement {i}", category="Science" if i % 2 == 0 else "History")

    assert skg.get_fact_count() == 5
    assert skg.get_shard_count() == 3

    skg.update_fact("f0", fact_statement="Updated Statement 0")
    assert skg.get_fact("f0")["fact_statement"] == "Updated Statement 0"

    assert len(skg.get_facts_by_category("Science")) == 3
    assert len(skg.get_facts_by_reliability(ReliabilityRating.VERIFIED)) == 5

    # Reload from disk cleanly.
    reloaded = ShardedKnowledgeGraph(str(shard_dir), shard_size=2)
    assert reloaded.get_fact("f0")["fact_statement"] == "Updated Statement 0"

    # Delete 3 facts so remaining 2 fit into 1 shard, then optimize.
    reloaded.delete_fact("f1")
    reloaded.delete_fact("f2")
    reloaded.delete_fact("f3")
    stats = reloaded.optimize_shards()

    assert stats["shards_before"] == 3
    assert stats["shards_after"] == 1
    assert reloaded.get_fact_count() == 2
    assert len(list(shard_dir.glob("shard_*.json"))) == 1
    assert sorted(reloaded.search_facts({"category": "Science"})) == ["f0", "f4"]
