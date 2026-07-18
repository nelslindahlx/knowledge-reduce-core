"""
Knowledge distillation: the "reduce" step of KnowledgeReduce.

This module turns a populated :class:`KnowledgeGraph` into a compact,
high-quality, *model-absorbable* artifact. The idea is to take sprawling,
noisy knowledge and distill it down to a ranked, de-duplicated set of
reliable facts, then serialize that set into formats a language model can
ingest:

* :meth:`KnowledgeDistiller.to_text` -- a ranked plain-text digest suitable
  for retrieval-augmented context or re-prompting.
* :meth:`KnowledgeDistiller.to_instruction_jsonl` -- instruction-tuning
  records (``{"instruction", "input", "output"}``).
* :meth:`KnowledgeDistiller.to_chat_jsonl` -- chat fine-tuning records
  (``{"messages": [...]}``), the format used by most SFT pipelines.

The distillation pipeline is:

    select -> filter by reliability -> deduplicate -> rank by quality -> top_k

so the output is both smaller than the input and biased toward the most
trustworthy facts.
"""

import json
from typing import Any, Dict, List, Optional

from .core import KnowledgeGraph, ReliabilityRating
from .quality import FactQualityFilter


class KnowledgeDistiller:
    """Distill a knowledge graph into model-absorbable training data.

    Args:
        knowledge_graph: The populated :class:`KnowledgeGraph` to distill.
        min_reliability: Drop facts whose reliability rating is below this
            level. Defaults to ``UNVERIFIED`` (keep everything).
        dedup_threshold: If > 0, near-duplicate facts whose semantic
            similarity meets or exceeds this threshold are collapsed,
            keeping the highest-quality representative. Set to ``0`` to
            disable deduplication.
        top_k: If set, keep at most this many facts after ranking.
    """

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        min_reliability: ReliabilityRating = ReliabilityRating.UNVERIFIED,
        dedup_threshold: float = 0.0,
        top_k: Optional[int] = None,
        quality_filter: Optional["FactQualityFilter"] = None,
    ):
        self.kg = knowledge_graph
        self.min_reliability = min_reliability
        self.dedup_threshold = dedup_threshold
        self.top_k = top_k
        self.quality_filter = quality_filter

    # ------------------------------------------------------------------ #
    # Selection pipeline
    # ------------------------------------------------------------------ #
    def select_facts(self) -> List[Dict[str, Any]]:
        """Return the distilled facts as a ranked list of node dicts.

        Each returned dict is a copy of the graph node's attributes with an
        added ``fact_id`` key. The list is filtered by reliability, passed
        through the optional quality filter, de-duplicated (if enabled),
        ranked by quality score (descending), and truncated to ``top_k``.
        """
        facts: List[Dict[str, Any]] = []
        for fact_id, data in self.kg.graph.nodes(data=True):
            if "fact_statement" not in data:
                continue
            rating = data.get("reliability_rating")
            if isinstance(rating, ReliabilityRating) and rating.value < self.min_reliability.value:
                continue
            record = dict(data)
            record["fact_id"] = fact_id
            facts.append(record)

        # Drop low-quality facts (junk subjects, run-on objects) if a
        # quality filter was supplied.
        if self.quality_filter is not None:
            facts = [f for f in facts if self.quality_filter.is_acceptable(f)]

        # Rank by quality score (descending), tie-break by fact_id for
        # determinism.
        facts.sort(key=lambda f: (-f.get("quality_score", 0), f["fact_id"]))

        if self.dedup_threshold and self.dedup_threshold > 0:
            facts = self._deduplicate(facts)

        if self.top_k is not None:
            facts = facts[: self.top_k]

        return facts

    def _deduplicate(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse near-duplicate facts, keeping the highest-quality one.

        Facts are assumed to arrive pre-sorted by quality (descending), so
        the first time we see a statement it is the best representative and
        any later near-duplicate is dropped.
        """
        kept: List[Dict[str, Any]] = []
        for fact in facts:
            statement = fact["fact_statement"]
            is_dup = any(
                self._similarity(statement, k["fact_statement"]) >= self.dedup_threshold
                for k in kept
            )
            if not is_dup:
                kept.append(fact)
        return kept

    @staticmethod
    def _similarity(text1: str, text2: str) -> float:
        """Lightweight Jaccard word-overlap similarity in ``[0, 1]``.

        Kept dependency-free on purpose so distillation works without numpy
        or any NLP model. Good enough to catch obvious duplicates.
        """
        if text1 == text2:
            return 1.0
        w1 = set(text1.lower().split())
        w2 = set(text2.lower().split())
        if not w1 or not w2:
            return 0.0
        return len(w1 & w2) / len(w1 | w2)

    # ------------------------------------------------------------------ #
    # Serializers
    # ------------------------------------------------------------------ #
    def to_text(self, include_metadata: bool = True) -> str:
        """Render the distilled facts as a ranked plain-text digest."""
        lines: List[str] = []
        for i, fact in enumerate(self.select_facts(), start=1):
            statement = fact["fact_statement"].strip()
            if include_metadata:
                rating = fact.get("reliability_rating")
                rating_name = rating.name if isinstance(rating, ReliabilityRating) else str(rating)
                category = fact.get("category", "General")
                lines.append(f"{i}. {statement} [{category}; {rating_name}]")
            else:
                lines.append(f"{i}. {statement}")
        return "\n".join(lines)

    def to_instruction_jsonl(self) -> str:
        """Render distilled facts as instruction-tuning JSONL.

        Each line is ``{"instruction", "input", "output"}``. When a fact
        carries a generated question, that question becomes the instruction
        and its answer the output; otherwise a generic category-based
        instruction is used with the full statement as output.
        """
        records = []
        for fact in self.select_facts():
            statement = fact["fact_statement"].strip()
            category = fact.get("category", "General")
            question = fact.get("question")
            answer = fact.get("answer")
            if question and answer:
                instruction = question
                output = answer
            else:
                instruction = f"State a verified fact about {category}."
                output = statement
            records.append(json.dumps({
                "instruction": instruction,
                "input": "",
                "output": output,
            }, ensure_ascii=False))
        return "\n".join(records)

    def to_chat_jsonl(self) -> str:
        """Render distilled facts as chat fine-tuning JSONL.

        Each line is ``{"messages": [user, assistant]}`` -- the format
        consumed by most supervised fine-tuning pipelines. When a fact
        carries a generated Q&A pair (``question``/``answer`` attributes),
        the real question and answer are used; otherwise a generic prompt
        about the fact's category is emitted as a fallback.
        """
        records = []
        for fact in self.select_facts():
            statement = fact["fact_statement"].strip()
            category = fact.get("category", "General")
            question = fact.get("question")
            answer = fact.get("answer")
            if question and answer:
                user_content = question
                assistant_content = answer
            else:
                user_content = f"Tell me a fact about {category}."
                assistant_content = statement
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            records.append(json.dumps({"messages": messages}, ensure_ascii=False))
        return "\n".join(records)

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        """Return distillation statistics, including the reduction ratio."""
        total = sum(
            1 for _, d in self.kg.graph.nodes(data=True) if "fact_statement" in d
        )
        selected = len(self.select_facts())
        reduction = 0.0 if total == 0 else 1.0 - (selected / total)
        return {
            "total_facts": total,
            "selected_facts": selected,
            "removed_facts": total - selected,
            "reduction_ratio": reduction,
        }

    # ------------------------------------------------------------------ #
    # File output
    # ------------------------------------------------------------------ #
    def distill_to_file(self, path: str, fmt: str = "chat", encoding: str = "utf-8") -> int:
        """Distill the graph and write the result to a file.

        Args:
            path: Destination file path.
            fmt: One of ``"chat"`` (chat JSONL), ``"instruction"``
                (instruction JSONL), or ``"text"`` (ranked digest).
            encoding: Text encoding for the output file.

        Returns:
            The number of facts written (for JSONL formats, the number of
            non-empty lines; for text, the number of distilled facts).

        Raises:
            ValueError: If ``fmt`` is not a recognized format.
        """
        serializers = {
            "chat": self.to_chat_jsonl,
            "instruction": self.to_instruction_jsonl,
            "text": self.to_text,
        }
        if fmt not in serializers:
            raise ValueError(
                f"Unknown format '{fmt}'. Expected one of: {', '.join(sorted(serializers))}."
            )

        content = serializers[fmt]()
        with open(path, "w", encoding=encoding) as fh:
            fh.write(content)
            if content and not content.endswith("\n"):
                fh.write("\n")

        return len([line for line in content.splitlines() if line.strip()])
