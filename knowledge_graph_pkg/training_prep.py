"""Training-data preparation for ModelReduce (Session 7)."""

import json
import os
import random
import tempfile
from typing import Any, Dict, List, Optional, Tuple

_RELIABILITY_ORDER = ["UNVERIFIED", "POSSIBLY_TRUE", "LIKELY_TRUE", "VERIFIED"]
_VALID_ROLES = {"system", "user", "assistant"}


def validate_chat_record(rec: Any) -> Tuple[bool, str]:
    """Validate one chat-format SFT record. Returns (ok, reason)."""
    if not isinstance(rec, dict) or "messages" not in rec:
        return False, "record has no 'messages' field"
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return False, "messages must be a list of >= 2 turns"
    for m in msgs:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            return False, "each message needs role + content"
        if m["role"] not in _VALID_ROLES:
            return False, f"invalid role: {m.get('role')}"
        content = m.get("content")
        if not isinstance(content, str) or not content.strip():
            return False, "empty message content"
    roles = [m["role"] for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return False, "needs at least one user and one assistant turn"
    return True, ""


def _reliability_of(rec: Any) -> str:
    if not isinstance(rec, dict):
        return "UNVERIFIED"
    meta = rec.get("metadata")
    if not isinstance(meta, dict):
        return "UNVERIFIED"
    rel = str(meta.get("reliability") or "UNVERIFIED").strip().upper()
    return rel or "UNVERIFIED"


def _resolve_min_reliability_idx(min_reliability: Optional[str]) -> int:
    if min_reliability is None:
        return -1
    norm = str(min_reliability).strip().upper()
    if norm not in _RELIABILITY_ORDER:
        raise ValueError(
            f"Unknown min_reliability {min_reliability!r}. Expected one of {_RELIABILITY_ORDER}."
        )
    return _RELIABILITY_ORDER.index(norm)


def _filter_records(
    input_path: str, min_reliability: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int, int]:
    min_idx = _resolve_min_reliability_idx(min_reliability)
    kept_records: List[Dict[str, Any]] = []
    dropped = total = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            ok, _reason = validate_chat_record(rec)
            if not ok:
                dropped += 1
                continue
            if min_idx >= 0:
                rel = _reliability_of(rec)
                rel_idx = _RELIABILITY_ORDER.index(rel) if rel in _RELIABILITY_ORDER else 0
                if rel_idx < min_idx:
                    dropped += 1
                    continue
            kept_records.append(rec)

    return kept_records, dropped, total


def _write_jsonl_atomic(records: List[Dict[str, Any]], output_path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path)) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=out_dir,
            delete=False,
            encoding="utf-8",
            prefix=".prep_",
            suffix=".tmp",
        ) as fout:
            tmp_path = fout.name
            for rec in records:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def prepare_sft_dataset(
    input_path: str, output_path: str, min_reliability: Optional[str] = None
) -> Dict[str, Any]:
    """Validate + filter a chat-JSONL shard into a training-ready dataset atomically."""
    kept_records, dropped, total = _filter_records(
        input_path, min_reliability=min_reliability
    )
    _write_jsonl_atomic(kept_records, output_path)
    return {"kept": len(kept_records), "dropped": dropped, "total": total}


def prepare_mlx_dataset(
    input_path: str,
    output_dir: str,
    min_reliability: Optional[str] = None,
    valid_ratio: float = 0.1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Prepare split ``train.jsonl`` and ``valid.jsonl`` files for :class:`MLXTrainer`."""
    kept_records, dropped, total = _filter_records(
        input_path, min_reliability=min_reliability
    )
    records = list(kept_records)
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    if n == 0:
        train_records, valid_records = [], []
    elif n == 1:
        train_records, valid_records = records, records
    else:
        ratio = max(0.0, min(0.5, float(valid_ratio)))
        n_valid = max(1, round(n * ratio)) if ratio > 0 else 1
        n_valid = min(n_valid, n - 1)
        valid_records = records[:n_valid]
        train_records = records[n_valid:]

    os.makedirs(output_dir, exist_ok=True)
    _write_jsonl_atomic(train_records, os.path.join(output_dir, "train.jsonl"))
    _write_jsonl_atomic(valid_records, os.path.join(output_dir, "valid.jsonl"))
    return {
        "train": len(train_records),
        "valid": len(valid_records),
        "kept": n,
        "dropped": dropped,
        "total": total,
    }


def dataset_stats(path: str) -> Dict[str, Any]:
    """Summarize a prepared SFT dataset: counts, reliability mix, avg lengths."""
    records = 0
    by_reliability: Dict[str, int] = {}
    user_chars = assistant_chars = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            records += 1
            rel = _reliability_of(rec)
            by_reliability[rel] = by_reliability.get(rel, 0) + 1
            msgs = rec.get("messages")
            if not isinstance(msgs, list):
                continue
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                if m.get("role") == "user":
                    user_chars += len(str(m.get("content") or ""))
                elif m.get("role") == "assistant":
                    assistant_chars += len(str(m.get("content") or ""))
    return {
        "records": records,
        "by_reliability": by_reliability,
        "avg_user_chars": round(user_chars / records, 1) if records else 0,
        "avg_assistant_chars": round(assistant_chars / records, 1) if records else 0,
    }
