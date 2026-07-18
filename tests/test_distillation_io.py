"""
Tests for distillation file I/O and real Q&A-based chat output.
"""
import json
from datetime import datetime

import pytest

from knowledge_graph_pkg import (
    KnowledgeGraph, SemanticKnowledgeGraph, KnowledgeDistiller, ReliabilityRating,
)


@pytest.fixture
def kg_from_text():
    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    skg.create_facts_from_text(
        "Marie Curie discovered radium. Marie Curie was born in Warsaw.",
        source_id="demo",
        reliability=ReliabilityRating.LIKELY_TRUE,
    )
    return kg


# ---------- real Q&A in chat output ----------

def test_chat_jsonl_uses_real_questions(kg_from_text):
    d = KnowledgeDistiller(kg_from_text, min_reliability=ReliabilityRating.LIKELY_TRUE)
    lines = [l for l in d.to_chat_jsonl().splitlines() if l.strip()]
    user_msgs = []
    for line in lines:
        rec = json.loads(line)
        user_msgs += [m["content"] for m in rec["messages"] if m["role"] == "user"]
    # at least one real, specific question (not the generic 'Tell me a fact')
    assert any(q.startswith(("Where", "What", "Who")) for q in user_msgs)


def test_chat_jsonl_answer_matches_object(kg_from_text):
    d = KnowledgeDistiller(kg_from_text, min_reliability=ReliabilityRating.LIKELY_TRUE)
    found = False
    for line in d.to_chat_jsonl().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        q = next(m["content"] for m in rec["messages"] if m["role"] == "user")
        a = next(m["content"] for m in rec["messages"] if m["role"] == "assistant")
        if q == "Where was Marie Curie born?":
            assert a == "Warsaw"
            found = True
    assert found, "expected the born_in Q&A pair in chat output"


# ---------- file ingestion ----------

def test_create_facts_from_file(tmp_path):
    src = tmp_path / "source.txt"
    src.write_text("Marie Curie discovered radium.")
    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    ids = skg.create_facts_from_file(
        str(src), source_id="filedemo", reliability=ReliabilityRating.LIKELY_TRUE
    )
    assert ids
    assert kg.get_fact(ids[0])["answer"] == "radium"


# ---------- distill to file ----------

def test_distill_to_file_chat(tmp_path, kg_from_text):
    out = tmp_path / "train.jsonl"
    d = KnowledgeDistiller(kg_from_text, min_reliability=ReliabilityRating.LIKELY_TRUE)
    n = d.distill_to_file(str(out), fmt="chat")
    assert out.exists()
    lines = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == n
    for line in lines:
        json.loads(line)  # valid JSON per line


def test_distill_to_file_text(tmp_path, kg_from_text):
    out = tmp_path / "digest.txt"
    d = KnowledgeDistiller(kg_from_text, min_reliability=ReliabilityRating.LIKELY_TRUE)
    d.distill_to_file(str(out), fmt="text")
    assert out.exists()
    assert "Marie Curie" in out.read_text()


def test_distill_to_file_rejects_unknown_format(tmp_path, kg_from_text):
    out = tmp_path / "x.dat"
    d = KnowledgeDistiller(kg_from_text)
    with pytest.raises(ValueError):
        d.distill_to_file(str(out), fmt="bogus")
