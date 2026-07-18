"""
Tests for the KnowledgeReduce distillation layer.

Distillation = the "reduce" step: take a populated knowledge graph and
compress it into a compact, high-quality, model-absorbable artifact
(distilled text, instruction-tuning JSONL, chat JSONL).

These tests are written first (TDD). They define the desired behavior of
the not-yet-implemented knowledge_graph_pkg.distillation module.
"""
import json
from datetime import datetime

import pytest

from knowledge_graph_pkg import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.distillation import KnowledgeDistiller


def _add(kg, fid, statement, rating=ReliabilityRating.VERIFIED, usage=10,
         category="General", tags=None):
    kg.add_fact(
        fact_id=fid,
        fact_statement=statement,
        category=category,
        tags=tags or ["t"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=rating,
        source_id="src",
        source_title="Source Title",
        author_creator="Author",
        publication_date=datetime.now(),
        url_reference="http://example.com",
        related_facts=[],
        contextual_notes="",
        access_level="public",
        usage_count=usage,
    )


@pytest.fixture
def populated_kg():
    kg = KnowledgeGraph()
    _add(kg, "f_verified", "The Earth orbits the Sun.", ReliabilityRating.VERIFIED, 20, "Astronomy")
    _add(kg, "f_likely", "Mars may have hosted microbial life.", ReliabilityRating.LIKELY_TRUE, 5, "Astronomy")
    _add(kg, "f_unverified", "Aliens built the pyramids.", ReliabilityRating.UNVERIFIED, 0, "Myth")
    return kg


# ---------- selection / filtering ----------

def test_min_reliability_filters_out_low_confidence(populated_kg):
    d = KnowledgeDistiller(populated_kg, min_reliability=ReliabilityRating.LIKELY_TRUE)
    statements = [f["fact_statement"] for f in d.select_facts()]
    assert "The Earth orbits the Sun." in statements
    assert "Mars may have hosted microbial life." in statements
    assert "Aliens built the pyramids." not in statements


def test_select_facts_ranked_by_quality_desc(populated_kg):
    d = KnowledgeDistiller(populated_kg)
    facts = d.select_facts()
    scores = [f["quality_score"] for f in facts]
    assert scores == sorted(scores, reverse=True)
    # highest reliability+usage fact should be first
    assert facts[0]["fact_statement"] == "The Earth orbits the Sun."


def test_top_k_limits_output(populated_kg):
    d = KnowledgeDistiller(populated_kg, top_k=1)
    facts = d.select_facts()
    assert len(facts) == 1
    assert facts[0]["fact_statement"] == "The Earth orbits the Sun."


# ---------- deduplication ----------

def test_dedup_removes_near_duplicate_facts():
    kg = KnowledgeGraph()
    _add(kg, "a", "The Earth orbits the Sun.", ReliabilityRating.VERIFIED, 20)
    _add(kg, "b", "The Earth orbits the Sun.", ReliabilityRating.LIKELY_TRUE, 1)  # dup, lower quality
    _add(kg, "c", "Water boils at 100 degrees Celsius.", ReliabilityRating.VERIFIED, 10)
    d = KnowledgeDistiller(kg, dedup_threshold=0.9)
    facts = d.select_facts()
    statements = [f["fact_statement"] for f in facts]
    # only one Earth fact survives, and it's the higher-quality one
    assert statements.count("The Earth orbits the Sun.") == 1
    assert "Water boils at 100 degrees Celsius." in statements


# ---------- text output ----------

def test_to_text_contains_statements_and_is_nonempty(populated_kg):
    d = KnowledgeDistiller(populated_kg, min_reliability=ReliabilityRating.LIKELY_TRUE)
    text = d.to_text()
    assert isinstance(text, str)
    assert "The Earth orbits the Sun." in text
    assert "Aliens built the pyramids." not in text


# ---------- instruction-tuning JSONL ----------

def test_to_instruction_jsonl_valid_lines(populated_kg):
    d = KnowledgeDistiller(populated_kg, min_reliability=ReliabilityRating.LIKELY_TRUE)
    out = d.to_instruction_jsonl()
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 2  # two facts pass the filter
    for line in lines:
        rec = json.loads(line)  # must be valid JSON
        assert set(rec) >= {"instruction", "output"}
        assert rec["output"]


# ---------- chat JSONL ----------

def test_to_chat_jsonl_valid_messages(populated_kg):
    d = KnowledgeDistiller(populated_kg, min_reliability=ReliabilityRating.LIKELY_TRUE)
    out = d.to_chat_jsonl()
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 2
    for line in lines:
        rec = json.loads(line)
        assert "messages" in rec
        roles = [m["role"] for m in rec["messages"]]
        assert "user" in roles and "assistant" in roles


# ---------- stats ----------

def test_stats_reports_reduction(populated_kg):
    d = KnowledgeDistiller(populated_kg, min_reliability=ReliabilityRating.LIKELY_TRUE)
    s = d.stats()
    assert s["total_facts"] == 3
    assert s["selected_facts"] == 2
    assert 0.0 <= s["reduction_ratio"] <= 1.0
