"""
Ultimate features demonstration for KnowledgeReduce.

This example demonstrates all the advanced capabilities of KnowledgeReduce Ultimate,
including vector embeddings, real-time streaming, and blockchain verification.
"""
import argparse
from contextlib import nullcontext
import os
import tempfile
from datetime import datetime
from knowledge_graph_pkg.core import ReliabilityRating
from knowledge_graph_pkg.enhanced import EnhancedKnowledgeGraph
from knowledge_graph_pkg.vector import VectorKnowledgeGraph
from knowledge_graph_pkg.streaming import StreamingKnowledgeGraph
from knowledge_graph_pkg.blockchain import BlockchainKnowledgeGraph
from knowledge_graph_pkg.visualization import plot_knowledge_graph


def main(output_dir: str = None):
    print("KnowledgeReduce Ultimate Example")
    print("================================\n")

    ctx = (
        tempfile.TemporaryDirectory(prefix="kr_ultimate_example_")
        if output_dir is None
        else nullcontext(output_dir)
    )
    with ctx as resolved_out:
        os.makedirs(resolved_out, exist_ok=True)

        # Create a base knowledge graph with enhanced capabilities
        print("\n1. Creating Enhanced Knowledge Graph")
        print("----------------------------------")
        kg = EnhancedKnowledgeGraph(cache_enabled=True, auto_save_interval=0)

        print("Adding initial facts...")

        astronomy_facts = [
            {
                'fact_id': 'fact_001',
                'fact_statement': 'The Earth orbits the Sun',
                'category': 'Astronomy',
                'tags': ['earth', 'sun', 'orbit', 'solar system'],
                'reliability_rating': ReliabilityRating.VERIFIED
            },
            {
                'fact_id': 'fact_002',
                'fact_statement': 'Jupiter is the largest planet in our solar system',
                'category': 'Astronomy',
                'tags': ['jupiter', 'planet', 'solar system', 'size'],
                'reliability_rating': ReliabilityRating.VERIFIED
            },
            {
                'fact_id': 'fact_003',
                'fact_statement': 'The Moon orbits the Earth',
                'category': 'Astronomy',
                'tags': ['moon', 'earth', 'orbit', 'satellite'],
                'reliability_rating': ReliabilityRating.VERIFIED
            },
            {
                'fact_id': 'fact_004',
                'fact_statement': 'Mars has two moons: Phobos and Deimos',
                'category': 'Astronomy',
                'tags': ['mars', 'moons', 'phobos', 'deimos'],
                'reliability_rating': ReliabilityRating.VERIFIED
            },
            {
                'fact_id': 'fact_005',
                'fact_statement': 'Venus is the hottest planet in our solar system',
                'category': 'Astronomy',
                'tags': ['venus', 'temperature', 'solar system'],
                'reliability_rating': ReliabilityRating.VERIFIED
            }
        ]

        now = datetime.now()
        for fact in astronomy_facts:
            fact.update({
                'date_recorded': now,
                'last_updated': now,
                'source_id': 'astronomy_textbook',
                'source_title': 'Principles of Astronomy',
                'author_creator': 'Dr. Neil Stargazer',
                'publication_date': now,
                'url_reference': 'https://example.com/astronomy',
                'related_facts': [],
                'contextual_notes': f"Basic fact about {fact['tags'][0]}",
                'access_level': 'public',
                'usage_count': 0
            })

        kg.batch_add_facts(astronomy_facts)
        print(f"Added {len(astronomy_facts)} initial facts")

        # Part 2: Vector-based Semantic Search
        print("\n2. Vector-based Semantic Search")
        print("-----------------------------")

        vector_kg = VectorKnowledgeGraph(kg)

        print("Generating vector embeddings...")
        num_embeddings = vector_kg.generate_embeddings()
        print(f"Generated {num_embeddings} embeddings")

        print("\nPerforming semantic search:")
        search_query = "planets in the solar system"
        search_results = vector_kg.semantic_search(search_query, top_k=3)

        print(f"Top results for query '{search_query}':")
        for fact_id, score in search_results:
            fact = kg.get_fact(fact_id)
            print(f"  - {fact_id}: {fact['fact_statement']} (score: {score:.2f})")

        print("\nFinding facts similar to 'fact_001':")
        similar_facts = vector_kg.find_similar_facts('fact_001', top_k=2)

        for fact_id, score in similar_facts:
            fact = kg.get_fact(fact_id)
            print(f"  - {fact_id}: {fact['fact_statement']} (score: {score:.2f})")

        print("\nDemonstrating query expansion:")
        original_query = "earth orbit"
        expanded_query = vector_kg.query_expansion(original_query)
        print(f"Original query: '{original_query}'")
        print(f"Expanded query: '{expanded_query}'")

        print("\nClustering facts:")
        clusters = vector_kg.cluster_facts(num_clusters=2)

        for cluster_id, fact_ids in clusters.items():
            print(f"Cluster {cluster_id}:")
            for fact_id in fact_ids:
                fact = kg.get_fact(fact_id)
                print(f"  - {fact_id}: {fact['fact_statement']}")

        emb_path = os.path.join(resolved_out, "embeddings.json")
        vector_kg.save_embeddings(emb_path)
        print(f"\nSaved embeddings to {emb_path}")

        # Part 3: Real-time Streaming
        print("\n3. Real-time Streaming")
        print("-------------------")

        streaming_kg = StreamingKnowledgeGraph(kg, auto_start=True)
        try:
            print("Adding facts from stream...")

            streaming_facts = [
                {
                    'fact_id': 'stream_001',
                    'fact_statement': 'Saturn has at least 82 moons',
                    'category': 'Astronomy',
                    'tags': ['saturn', 'moons', 'solar system'],
                },
                {
                    'fact_id': 'stream_002',
                    'fact_statement': 'Neptune is the windiest planet in our solar system',
                    'category': 'Astronomy',
                    'tags': ['neptune', 'wind', 'solar system'],
                },
                {
                    'fact_id': 'stream_003',
                    'fact_statement': 'Pluto was reclassified as a dwarf planet in 2006',
                    'category': 'Astronomy',
                    'tags': ['pluto', 'dwarf planet', 'classification'],
                }
            ]

            for fact in streaming_facts:
                event_id = streaming_kg.add_fact_from_stream(
                    fact_data=fact,
                    source_id='live_astronomy_feed',
                    reliability=ReliabilityRating.LIKELY_TRUE
                )
                print(f"  - Added streaming fact: {fact['fact_statement']} (event: {event_id})")

            streaming_kg.event_queue.join()

            update_event_id = streaming_kg.update_fact_from_stream(
                'stream_001',
                {'fact_statement': 'Saturn has 83 moons as of 2023', 'reliability_rating': ReliabilityRating.VERIFIED}
            )
            print(f"Updated fact stream_001 (event: {update_event_id})")
            streaming_kg.event_queue.join()

            print("\nEvent history:")
            events = streaming_kg.get_event_history(limit=5)
            for i, event in enumerate(events):
                print(f"  {i+1}. {event['type']} - {event['timestamp']} - {event['id']}")
        finally:
            streaming_kg.stop()
            print("Stopped streaming")

        # Part 4: Blockchain Verification
        print("\n4. Blockchain Verification")
        print("------------------------")

        blockchain_kg = BlockchainKnowledgeGraph(kg)

        print("Adding facts with blockchain verification...")

        blockchain_facts = [
            {
                'fact_id': 'blockchain_001',
                'fact_statement': 'Mercury is the smallest planet in our solar system',
                'category': 'Astronomy',
                'tags': ['mercury', 'planet', 'size', 'solar system'],
                'reliability_rating': ReliabilityRating.VERIFIED,
                'source_id': 'astronomy_blockchain',
            },
            {
                'fact_id': 'blockchain_002',
                'fact_statement': 'Uranus rotates on its side with an axial tilt of about 98 degrees',
                'category': 'Astronomy',
                'tags': ['uranus', 'rotation', 'axial tilt', 'solar system'],
                'reliability_rating': ReliabilityRating.VERIFIED,
                'source_id': 'astronomy_blockchain',
            }
        ]

        for fact in blockchain_facts:
            tx_hash = blockchain_kg.add_fact(**fact)
            print(f"  - Added blockchain fact: {fact['fact_statement']} (tx: {tx_hash})")

        print("\nMining a block...")
        block = blockchain_kg.mine_block()
        print(f"Mined block #{block['index']} with {len(block['transactions'])} transactions")

        update_tx = blockchain_kg.update_fact(
            'blockchain_001',
            fact_statement='Mercury is the smallest and innermost planet in our solar system'
        )
        print(f"Updated fact blockchain_001 (tx: {update_tx})")

        blockchain_kg.mine_block()

        print("\nVerifying facts:")
        for fact_id in ['blockchain_001', 'blockchain_002']:
            verification = blockchain_kg.verify_fact(fact_id)
            print(f"  - {fact_id}: {'Verified' if verification['verified'] else 'Not verified'}")
            print(f"    First recorded: {verification['first_recorded']}")
            print(f"    Last updated: {verification['last_updated']}")
            print(f"    Transactions: {verification['transactions']}")

        chain_verification = blockchain_kg.verify_chain()
        print(f"\nBlockchain integrity: {'Valid' if chain_verification['valid'] else 'Invalid'}")
        print(f"Blocks: {chain_verification['blocks']}")
        print(f"Transactions: {chain_verification['transactions']}")

        chain_path = os.path.join(resolved_out, "blockchain.json")
        blockchain_kg.export_chain(chain_path)
        print(f"\nExported blockchain to {chain_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            print("\nVisualizing knowledge graph...")
            fig = plot_knowledge_graph(kg.graph, title="Ultimate Knowledge Graph")
            fig_path = os.path.join(resolved_out, "knowledge_graph.png")
            fig.savefig(fig_path)
            plt.close(fig)
            print(f"Saved visualization to {fig_path}")
        except ImportError:
            print("\nmatplotlib not installed; skipping image plot.")

    print("\nKnowledgeReduce Ultimate example completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KnowledgeGraph ultimate features example.")
    parser.add_argument("--output-dir", default=None, help="Directory for example artifacts (default: temporary).")
    cli_args = parser.parse_args()
    main(output_dir=cli_args.output_dir)
