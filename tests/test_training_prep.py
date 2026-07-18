"""
Tests for training-data preparation (ModelReduce Session 7).

prepare_sft_dataset turns a compiled chat-JSONL shard into a validated,
training-ready dataset: it checks record shape, drops malformed/empty
records, can enforce a minimum reliability, and reports stats. This is the
testable core of the training pipeline; the actual LoRA run is a separate,
hardware-dependent script (scripts/train_sft.py) that is NOT executed here.
"""
import json
import pytest

from knowledge_graph_pkg.training_prep import (
    validate_chat_record, prepare_sft_dataset, dataset_stats,
)


def _chat(user, assistant, **meta):
    rec = {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}
    if meta:
        rec["metadata"] = meta
    return rec


def test_validate_good_record():
    ok, _ = validate_chat_record(_chat("What is ATP?", "energy currency"))
    assert ok is True


def test_validate_rejects_missing_messages():
    ok, reason = validate_chat_record({"foo": "bar"})
    assert ok is False and "messages" in reason.lower()


def test_validate_rejects_empty_content():
    ok, _ = validate_chat_record(_chat("", "answer"))
    assert ok is False
    ok2, _ = validate_chat_record(_chat("q", "   "))
    assert ok2 is False


def test_validate_rejects_wrong_roles():
    bad = {"messages": [{"role": "system", "content": "x"},
                        {"role": "assistant", "content": "y"}]}
    ok, _ = validate_chat_record(bad)
    assert ok is False


def test_prepare_filters_malformed(tmp_path):
    src = tmp_path / "shard.jsonl"
    lines = [
        json.dumps(_chat("What is ATP?", "energy currency")),
        json.dumps({"messages": []}),                       # empty
        json.dumps(_chat("good q", "good a")),
        "not json at all",                                   # junk line
        json.dumps(_chat("", "")),                           # empty content
    ]
    src.write_text("\n".join(lines))
    out = tmp_path / "train.jsonl"
    report = prepare_sft_dataset(str(src), str(out))
    kept = [l for l in out.read_text().splitlines() if l.strip()]
    assert len(kept) == 2
    assert report["kept"] == 2
    assert report["dropped"] == 3


def test_prepare_min_reliability(tmp_path):
    src = tmp_path / "shard.jsonl"
    src.write_text("\n".join([
        json.dumps(_chat("q1", "a1", reliability="VERIFIED")),
        json.dumps(_chat("q2", "a2", reliability="POSSIBLY_TRUE")),
        json.dumps(_chat("q3", "a3", reliability="LIKELY_TRUE")),
    ]))
    out = tmp_path / "train.jsonl"
    report = prepare_sft_dataset(str(src), str(out), min_reliability="LIKELY_TRUE")
    # POSSIBLY_TRUE dropped; VERIFIED + LIKELY_TRUE kept
    assert report["kept"] == 2


def test_dataset_stats(tmp_path):
    src = tmp_path / "train.jsonl"
    src.write_text("\n".join([
        json.dumps(_chat("q1", "a1", reliability="VERIFIED")),
        json.dumps(_chat("q2", "a2", reliability="LIKELY_TRUE")),
        json.dumps(_chat("q3", "a3", reliability="LIKELY_TRUE")),
    ]))
    stats = dataset_stats(str(src))
    assert stats["records"] == 3
    assert stats["by_reliability"]["LIKELY_TRUE"] == 2
    assert stats["avg_user_chars"] > 0
