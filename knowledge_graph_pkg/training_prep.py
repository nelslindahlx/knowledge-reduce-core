"""
Training-data preparation for ModelReduce (Session 7).

The bridge between a distilled shard and a fine-tuning run: validate the
chat-JSONL records, drop malformed ones, optionally enforce a minimum
reliability tier, and report dataset stats. Pure stdlib -- the heavy
training libraries (torch/transformers/peft) are NOT imported here; the
actual LoRA run lives in ``scripts/train_sft.py`` and is executed on
appropriate hardware, not in this package's test path.

See ``scripts/train_sft.py`` and ``docs/model_reduce_training.md`` for the
end-to-end pipeline and the (honest) hardware requirements.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

_RELIABILITY_ORDER = ["UNVERIFIED", "POSSIBLY_TRUE", "LIKELY_TRUE", "VERIFIED"]
_VALID_ROLES = {"user", "assistant"}


def validate_chat_record(rec: Dict[str, Any]) -> Tuple[bool, str]:
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
        if not str(m.get("content", "")).strip():
            return False, "empty message content"
    roles = [m["role"] for m in msgs]
    if "user" not in roles or "assistant" not in roles:
        return False, "needs at least one user and one assistant turn"
    return True, ""


def _reliability_of(rec: Dict[str, Any]) -> str:
    return str(rec.get("metadata", {}).get("reliability", "UNVERIFIED"))


def prepare_sft_dataset(input_path: str, output_path: str,
                        min_reliability: Optional[str] = None) -> Dict[str, Any]:
    """Validate + filter a chat-JSONL shard into a training-ready dataset.

    Drops malformed/unparseable/empty records and (optionally) any record
    below ``min_reliability``. Writes the survivors to ``output_path`` and
    returns a report ``{kept, dropped, total}``.
    """
    min_idx = _RELIABILITY_ORDER.index(min_reliability) if min_reliability else -1
    kept = dropped = total = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
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
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    return {"kept": kept, "dropped": dropped, "total": total}


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
            records += 1
            rel = _reliability_of(rec)
            by_reliability[rel] = by_reliability.get(rel, 0) + 1
            for m in rec.get("messages", []):
                if m.get("role") == "user":
                    user_chars += len(str(m.get("content", "")))
                elif m.get("role") == "assistant":
                    assistant_chars += len(str(m.get("content", "")))
    return {
        "records": records,
        "by_reliability": by_reliability,
        "avg_user_chars": round(user_chars / records, 1) if records else 0,
        "avg_assistant_chars": round(assistant_chars / records, 1) if records else 0,
    }
