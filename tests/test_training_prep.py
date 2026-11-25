"""
Tests for training-data preparation (ModelReduce Session 7).
"""

import json
import pytest

from knowledge_graph_pkg.training_prep import (
    dataset_stats,
    prepare_mlx_dataset,
    prepare_sft_dataset,
    validate_chat_record,
)


def _chat(user, assistant, **meta):
    rec = {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    if meta:
        rec["metadata"] = meta
    return rec


def test_validate_good_record_and_system_role():
    ok, _ = validate_chat_record(_chat("What is ATP?", "energy currency"))
    assert ok is True

    with_sys = {
        "messages": [
            {"role": "system", "content": "You are a tutor."},
            {"role": "user", "content": "What is ATP?"},
            {"role": "assistant", "content": "Energy currency."},
        ]
    }
    ok_sys, _ = validate_chat_record(with_sys)
    assert ok_sys is True


def test_validate_rejects_missing_messages():
    ok, reason = validate_chat_record({"foo": "bar"})
    assert ok is False and "messages" in reason.lower()


def test_validate_rejects_empty_and_non_string_content():
    ok, _ = validate_chat_record(_chat("", "answer"))
    assert ok is False
    ok2, _ = validate_chat_record(_chat("q", "   "))
    assert ok2 is False
    ok3, _ = validate_chat_record(_chat(None, "answer"))
    assert ok3 is False
    ok4, _ = validate_chat_record(_chat("q", 123))
    assert ok4 is False


def test_validate_rejects_wrong_roles():
    bad = {
        "messages": [
            {"role": "system", "content": "x"},
            {"role": "assistant", "content": "y"},
        ]
    }
    ok, _ = validate_chat_record(bad)
    assert ok is False


def test_prepare_filters_malformed(tmp_path):
    src = tmp_path / "shard.jsonl"
    lines = [
        json.dumps(_chat("What is ATP?", "energy currency")),
        json.dumps({"messages": []}),
        json.dumps(_chat("good q", "good a")),
        "not json at all",
        json.dumps(_chat("", "")),
    ]
    src.write_text("\n".join(lines))
    out = tmp_path / "train.jsonl"
    report = prepare_sft_dataset(str(src), str(out))
    kept = [line for line in out.read_text().splitlines() if line.strip()]
    assert len(kept) == 2
    assert report["kept"] == 2
    assert report["dropped"] == 3


def test_prepare_in_place_does_not_truncate(tmp_path):
    src = tmp_path / "inplace.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps(_chat("q1", "a1", reliability="VERIFIED")),
                json.dumps({"bad": "row"}),
            ]
        )
    )
    rep = prepare_sft_dataset(str(src), str(src))
    assert rep["kept"] == 1
    assert rep["dropped"] == 1
    kept = [line for line in src.read_text().splitlines() if line.strip()]
    assert len(kept) == 1


def test_prepare_min_reliability_case_insensitive(tmp_path):
    src = tmp_path / "shard.jsonl"
    src.write_text(
        "\n".join(
            [
                json.dumps(_chat("q1", "a1", reliability="VERIFIED")),
                json.dumps(_chat("q2", "a2", reliability="POSSIBLY_TRUE")),
                json.dumps(_chat("q3", "a3", reliability="LIKELY_TRUE")),
            ]
        )
    )
    out = tmp_path / "train.jsonl"
    report = prepare_sft_dataset(str(src), str(out), min_reliability="likely_true")
    assert report["kept"] == 2

    with pytest.raises(ValueError):
        prepare_sft_dataset(str(src), str(out), min_reliability="INVALID")


def test_prepare_mlx_dataset_creates_splits(tmp_path):
    src = tmp_path / "shard.jsonl"
    src.write_text(
        "\n".join(json.dumps(_chat(f"q{i}", f"a{i}", reliability="VERIFIED")) for i in range(5))
    )
    mlx_dir = tmp_path / "mlx_data"
    rep = prepare_mlx_dataset(str(src), str(mlx_dir), valid_ratio=0.2, seed=42)
    assert rep["kept"] == 5
    assert rep["train"] == 4 and rep["valid"] == 1
    assert (mlx_dir / "train.jsonl").is_file()
    assert (mlx_dir / "valid.jsonl").is_file()


def test_dataset_stats_handles_null_metadata_and_non_dicts(tmp_path):
    src = tmp_path / "train.jsonl"
    rec_null_meta = _chat("q0", "a0")
    rec_null_meta["metadata"] = None
    src.write_text(
        "\n".join(
            [
                json.dumps(_chat("q1", "a1", reliability="VERIFIED")),
                json.dumps(_chat("q2", "a2", reliability="LIKELY_TRUE")),
                json.dumps(rec_null_meta),
                '"just-a-string"',
            ]
        )
    )
    stats = dataset_stats(str(src))
    assert stats["records"] == 3
    assert stats["by_reliability"]["VERIFIED"] == 1
    assert stats["by_reliability"]["UNVERIFIED"] == 1
    assert stats["avg_user_chars"] > 0
