"""
Advanced knowledge graph example demonstrating enhanced features.

This example shows how to use the enhanced features of the KnowledgeReduce package,
including performance optimization, semantic capabilities, and sharding.
"""
import argparse
from contextlib import nullcontext
import os
import tempfile
import time
from datetime import datetime
from knowledge_graph_pkg.core import ReliabilityRating
from knowledge_graph_pkg.enhanced import EnhancedKnowledgeGraph
from knowledge_graph_pkg.semantic import SemanticKnowledgeGraph
from knowledge_graph_pkg.sharding import ShardedKnowledgeGraph


def main(output_dir: str = None):
    print("KnowledgeReduce Enhanced Example")
    print("================================\n")

    ctx = (
        tempfile.TemporaryDirectory(prefix="kr_enhanced_example_")
        if output_dir is None
        else nullcontext(output_dir)
    )
    with ctx as resolved_out:
        os.makedirs(resolved_out, exist_ok=True)

        # Part 1: Enhanced Knowledge Graph with Performance Optimization
        print("\n1. Enhanced Knowledge Graph with Performance Optimization")
        print("-------------------------------------------------------")

        enhanced_kg = EnhancedKnowledgeGraph(cache_enabled=True, auto_save_interval=0)

        print("Adding sample facts...")
        start_time = time.perf_counter()

        facts = []
        for i in range(100):
            facts.append({
                'fact_id': f"fact_{i:03d}",
                'fact_statement': f"This is sample fact {i}",
                'category': "Sample",
                'tags': ["sample", "test", f"tag_{i % 5}"],
                'date_recorded': datetime.now(),
                'last_updated': datetime.now(),
                'reliability_rating': ReliabilityRating.VERIFIED if i % 3 == 0 else
                                     (ReliabilityRating.LIKELY_TRUE if i % 3 == 1 else ReliabilityRating.POSSIBLY_TRUE),
                'source_id': f"source_{i % 10}",
                'source_title': f"Test Source {i % 10}",
                'author_creator': f"Author {i % 5}",
                'publication_date': datetime.now(),
                'url_reference': f"https://example.com/fact/{i}",
                'related_facts': [f"fact_{(i+j) % 100:03d}" for j in range(1, 4)],
                'contextual_notes': f"Notes for fact {i}",
                'access_level': "public",
                'usage_count': i % 20
            })

        enhanced_kg.batch_add_facts(facts)

        add_time = time.perf_counter() - start_time
        print(f"Added {len(facts)} facts in {add_time:.2f} seconds")

        print("\nDemonstrating caching benefits:")

        start_time = time.perf_counter()
        for i in range(20):
            _ = enhanced_kg.get_fact(f"fact_{i:03d}")
        uncached_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        for i in range(20):
            _ = enhanced_kg.get_fact(f"fact_{i:03d}")
        cached_time = time.perf_counter() - start_time

        print(f"Uncached retrieval time: {uncached_time:.6f} seconds")
        print(f"Cached retrieval time: {cached_time:.6f} seconds")
        print(f"Speedup factor: {uncached_time / max(cached_time, 1e-9):.2f}x")

        print("\nSearching for facts:")
        results = enhanced_kg.search_facts("sample fact 5")
        print(f"Found {len(results)} results for 'sample fact 5'")
        for fact_id, score in results[:3]:
            print(f"  - {fact_id}: {enhanced_kg.get_fact(fact_id)['fact_statement']} (score: {score:.2f})")

        # Part 2: Semantic Knowledge Graph
        print("\n2. Semantic Knowledge Graph")
        print("-------------------------")

        semantic_kg = SemanticKnowledgeGraph(enhanced_kg)

        sample_text = """
        John Smith works for Microsoft Corporation in Seattle. 
        The company was founded by Bill Gates in 1975. 
        Apple Inc. is headquartered in Cupertino and was founded by Steve Jobs.
        """

        print("\nExtracting entities from text:")
        entities = semantic_kg.extract_entities_from_text(sample_text)
        print(f"Found {len(entities)} entities:")
        for entity in entities:
            print(f"  - {entity['text']} ({entity['type']})")

        print("\nExtracting relations from text:")
        relations = semantic_kg.extract_relations_from_text(sample_text)
        print(f"Found {len(relations)} relations:")
        for relation in relations:
            print(f"  - {relation['subject']} {relation['predicate']} {relation['object']}")

        print("\nCreating facts from text:")
        fact_ids = semantic_kg.create_facts_from_text(sample_text, "text_sample_1", ReliabilityRating.POSSIBLY_TRUE)
        print(f"Created {len(fact_ids)} facts from text")

        print("\nFinding semantically similar facts:")
        if fact_ids:
            similar_facts = semantic_kg.find_semantically_similar_facts(fact_ids[0], threshold=0.3)
            print(f"Found {len(similar_facts)} facts similar to {fact_ids[0]}:")
            for fact_id, score in similar_facts[:3]:
                print(f"  - {fact_id}: {enhanced_kg.get_fact(fact_id)['fact_statement']} (similarity: {score:.2f})")

        # Part 3: Sharded Knowledge Graph
        print("\n3. Sharded Knowledge Graph")
        print("------------------------")

        shards_dir = os.path.join(resolved_out, "shards")
        sharded_kg = ShardedKnowledgeGraph(shards_dir, shard_size=25)

        print("Adding facts to sharded knowledge graph...")
        start_time = time.perf_counter()

        for i in range(100):
            sharded_kg.add_fact(
                fact_id=f"sharded_fact_{i:03d}",
                fact_statement=f"This is sharded fact {i}",
                category="Sharded" if i % 2 == 0 else "Distributed",
                tags=["sharded", "test", f"tag_{i % 5}"],
                date_recorded=datetime.now(),
                last_updated=datetime.now(),
                reliability_rating=ReliabilityRating.VERIFIED if i % 3 == 0 else
                                  (ReliabilityRating.LIKELY_TRUE if i % 3 == 1 else ReliabilityRating.POSSIBLY_TRUE),
                source_id=f"source_{i % 10}",
                source_title=f"Test Source {i % 10}",
                author_creator=f"Author {i % 5}",
                publication_date=datetime.now(),
                url_reference=f"https://example.com/fact/{i}",
                related_facts=[],
                contextual_notes=f"Notes for sharded fact {i}",
                access_level="public",
                usage_count=i % 20
            )

        add_time = time.perf_counter() - start_time
        print(f"Added 100 facts to sharded knowledge graph in {add_time:.2f} seconds")

        shard_count = sharded_kg.get_shard_count()
        fact_count = sharded_kg.get_fact_count()
        print(f"Knowledge graph is distributed across {shard_count} shards with {fact_count} total facts")

        print("\nRetrieving facts from different shards:")
        for i in [5, 30, 75]:
            fact_id = f"sharded_fact_{i:03d}"
            fact = sharded_kg.get_fact(fact_id)
            print(f"  - {fact_id}: {fact['fact_statement']} (from shard {sharded_kg.fact_to_shard[fact_id]})")

        print("\nSearching across shards:")
        category_facts = sharded_kg.get_facts_by_category("Sharded")
        print(f"Found {len(category_facts)} facts in category 'Sharded'")

        print("\nOptimizing shards:")
        stats = sharded_kg.optimize_shards()
        print(f"Optimization stats: {stats}")

    print("\nEnhanced KnowledgeReduce example completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KnowledgeGraph enhanced features example.")
    parser.add_argument("--output-dir", default=None, help="Directory for shard output (default: temporary).")
    cli_args = parser.parse_args()
    main(output_dir=cli_args.output_dir)
