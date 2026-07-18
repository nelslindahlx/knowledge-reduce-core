"""
Export helpers for training-data polish.

Utilities applied to serialized record lines (JSONL or text) before they
are written:

* :func:`split_records` -- deterministic train/validation partition.
* :func:`estimate_tokens` / :func:`budget_records` -- cap output to a
  rough token budget (useful for RAG context windows).

These operate on plain string records so they compose with any of the
distiller's serializers.
"""

import random
from typing import List, Tuple


def split_records(records: List[str], ratio: float = 0.9,
                  seed: int = 42) -> Tuple[List[str], List[str]]:
    """Partition records into (train, val) by ``ratio``, deterministically.

    Args:
        records: The serialized record lines.
        ratio: Fraction assigned to the training set (0..1).
        seed: RNG seed for a reproducible shuffle.

    Returns:
        ``(train, val)`` lists; together they contain every input record
        exactly once.
    """
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    n_train = round(len(shuffled) * ratio)
    return shuffled[:n_train], shuffled[n_train:]


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token).

    A cheap, model-agnostic heuristic — good enough for budgeting without
    pulling in a tokenizer dependency.
    """
    return max(1, len(text) // 4)


def budget_records(records: List[str], max_tokens: int) -> List[str]:
    """Return the longest prefix of ``records`` that fits ``max_tokens``.

    Records are assumed to be pre-ranked (best first), so truncation keeps
    the highest-value items. A record that alone exceeds the budget is
    skipped rather than included.
    """
    kept: List[str] = []
    used = 0
    for rec in records:
        cost = estimate_tokens(rec)
        if used + cost > max_tokens:
            continue
        kept.append(rec)
        used += cost
    return kept
