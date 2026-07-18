#!/usr/bin/env python3
"""
End-to-end KnowledgeReduce distillation pipeline.

Demonstrates the full flow:
    raw text  ->  semantic extraction  ->  knowledge graph
              ->  distillation (filter + dedup + rank)
              ->  model-absorbable output (text digest + chat JSONL)

Run:  python examples/distillation_pipeline.py
"""
from datetime import datetime

from knowledge_graph_pkg import (
    KnowledgeGraph,
    SemanticKnowledgeGraph,
    KnowledgeDistiller,
    ReliabilityRating,
)


SOURCE_TEXT = (
    "Marie Curie was born in Warsaw. She discovered radium. "
    "Albert Einstein developed relativity. Einstein was born in Germany. "
    "Aliens built the pyramids."  # noise / unverified
)


def main():
    # 1. Build a graph and extract facts from raw text.
    #    resolve_coref=True rewrites leading pronouns (e.g. "She discovered
    #    radium" -> "Marie Curie discovered radium") so facts and the
    #    generated Q&A pairs are attributed to the right entity.
    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    auto_ids = skg.create_facts_from_text(
        SOURCE_TEXT, source_id="demo",
        reliability=ReliabilityRating.LIKELY_TRUE,
        resolve_coref=True,
    )

    # 2. Add a couple of curated facts at different reliability levels.
    kg.add_fact(
        "curated_verified", "Water is composed of hydrogen and oxygen.",
        "Chemistry", ["water"], datetime.now(), datetime.now(),
        ReliabilityRating.VERIFIED, "textbook", "Chem 101", "Author",
        datetime.now(), "", [], "", "public", 30,
    )
    kg.add_fact(
        "curated_noise", "The moon is made of cheese.",
        "Myth", ["moon"], datetime.now(), datetime.now(),
        ReliabilityRating.UNVERIFIED, "folklore", "Tales", "Unknown",
        datetime.now(), "", [], "", "public", 0,
    )

    print(f"Auto-extracted {len(auto_ids)} facts from text.")
    print(f"Graph now holds {kg.graph.number_of_nodes()} total facts.\n")

    # 3. Distill: keep only LIKELY_TRUE+, dedup, rank by quality.
    distiller = KnowledgeDistiller(
        kg,
        min_reliability=ReliabilityRating.LIKELY_TRUE,
        dedup_threshold=0.85,
    )

    stats = distiller.stats()
    print("=== DISTILLATION STATS ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n=== DISTILLED TEXT DIGEST (model-absorbable context) ===")
    print(distiller.to_text())

    print("\n=== CHAT FINE-TUNING JSONL (first 3 lines) ===")
    for line in distiller.to_chat_jsonl().splitlines()[:3]:
        print(line)

    print("\n=== INSTRUCTION-TUNING JSONL (first 3 lines) ===")
    for line in distiller.to_instruction_jsonl().splitlines()[:3]:
        print(line)


if __name__ == "__main__":
    main()
