"""
Example script demonstrating advanced usage of the KnowledgeGraph package.

This script shows how to create a knowledge graph, perform analysis,
visualize the graph, and use utility functions for import/export.
"""
from datetime import datetime
import os
import matplotlib.pyplot as plt
from knowledge_graph_pkg import KnowledgeGraph
from knowledge_graph_pkg.core import ReliabilityRating
from knowledge_graph_pkg.analysis import KnowledgeGraphAnalyzer
from knowledge_graph_pkg.visualization import (
    plot_knowledge_graph,
    export_to_gexf,
    plot_reliability_distribution,
    get_graph_statistics
)
from knowledge_graph_pkg.utils import export_to_json, filter_facts_by_reliability

def main():
    # Create a new knowledge graph
    print("Creating a new knowledge graph with sample data...")
    kg = KnowledgeGraph()
    
    # Add facts about the solar system
    kg.add_fact(
        fact_id="earth_sun",
        fact_statement="The Earth orbits the Sun",
        category="Astronomy",
        tags=["earth", "sun", "orbit"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.VERIFIED,
        source_id="astronomy_textbook",
        source_title="Principles of Astronomy",
        author_creator="Dr. Neil Stargazer",
        publication_date=datetime.now(),
        url_reference="https://example.com/astronomy",
        related_facts=["earth_third", "earth_moon"],
        contextual_notes="Fundamental astronomical fact",
        access_level="public",
        usage_count=100
    )
    
    kg.add_fact(
        fact_id="earth_third",
        fact_statement="The Earth is the third planet from the Sun",
        category="Astronomy",
        tags=["earth", "sun", "planet", "solar system"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.VERIFIED,
        source_id="astronomy_textbook",
        source_title="Principles of Astronomy",
        author_creator="Dr. Neil Stargazer",
        publication_date=datetime.now(),
        url_reference="https://example.com/astronomy",
        related_facts=["earth_sun", "mars_fourth"],
        contextual_notes="Position in the solar system",
        access_level="public",
        usage_count=80
    )
    
    kg.add_fact(
        fact_id="earth_moon",
        fact_statement="The Moon orbits the Earth",
        category="Astronomy",
        tags=["earth", "moon", "orbit"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.VERIFIED,
        source_id="astronomy_textbook",
        source_title="Principles of Astronomy",
        author_creator="Dr. Neil Stargazer",
        publication_date=datetime.now(),
        url_reference="https://example.com/astronomy",
        related_facts=["earth_sun"],
        contextual_notes="Earth's natural satellite",
        access_level="public",
        usage_count=75
    )
    
    kg.add_fact(
        fact_id="mars_fourth",
        fact_statement="Mars is the fourth planet from the Sun",
        category="Astronomy",
        tags=["mars", "sun", "planet", "solar system"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.VERIFIED,
        source_id="astronomy_textbook",
        source_title="Principles of Astronomy",
        author_creator="Dr. Neil Stargazer",
        publication_date=datetime.now(),
        url_reference="https://example.com/astronomy",
        related_facts=["earth_third", "mars_life"],
        contextual_notes="Position in the solar system",
        access_level="public",
        usage_count=60
    )
    
    kg.add_fact(
        fact_id="mars_life",
        fact_statement="Mars may have once supported microbial life",
        category="Astronomy",
        tags=["mars", "life", "microbial"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.LIKELY_TRUE,
        source_id="mars_research",
        source_title="Mars Research Journal",
        author_creator="Mars Research Team",
        publication_date=datetime.now(),
        url_reference="https://example.com/mars-research",
        related_facts=["mars_fourth"],
        contextual_notes="Based on evidence of past water and organic compounds",
        access_level="public",
        usage_count=40
    )
    
    kg.add_fact(
        fact_id="earth_flat",
        fact_statement="The Earth is flat",
        category="Pseudoscience",
        tags=["earth", "flat earth"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.UNVERIFIED,
        source_id="conspiracy_blog",
        source_title="Truth Seekers Blog",
        author_creator="Anonymous",
        publication_date=datetime.now(),
        url_reference="https://example.com/conspiracy",
        related_facts=[],
        contextual_notes="Widely debunked claim",
        access_level="public",
        usage_count=5
    )
    
    # Add relationships between facts
    kg.graph.add_edge("earth_sun", "earth_third")
    kg.graph.add_edge("earth_third", "earth_sun")
    kg.graph.add_edge("earth_sun", "earth_moon")
    kg.graph.add_edge("earth_moon", "earth_sun")
    kg.graph.add_edge("earth_third", "mars_fourth")
    kg.graph.add_edge("mars_fourth", "earth_third")
    kg.graph.add_edge("mars_fourth", "mars_life")
    kg.graph.add_edge("earth_sun", "earth_flat")  # Contradiction
    
    print(f"Created knowledge graph with {len(kg.graph.nodes)} facts and {len(kg.graph.edges)} relationships")
    
    # Analyze the knowledge graph
    print("\nAnalyzing knowledge graph...")
    analyzer = KnowledgeGraphAnalyzer(kg)
    
    # Get central facts
    central_facts = analyzer.get_central_facts()
    print("\nMost central facts:")
    for fact_id, score in central_facts:
        print(f"  - {fact_id}: {kg.get_fact(fact_id)['fact_statement']} (centrality: {score:.4f})")
    
    # Get reliability summary
    reliability_summary = analyzer.get_reliability_summary()
    print("\nReliability summary:")
    for rating, count in reliability_summary.items():
        if count > 0:
            print(f"  - {rating}: {count} facts")
    
    # Find contradictions
    contradictions = analyzer.find_contradictions()
    print("\nPotential contradictions:")
    if contradictions:
        for fact1, fact2 in contradictions:
            print(f"  - {fact1} vs {fact2}:")
            print(f"    * {kg.get_fact(fact1)['fact_statement']} ({kg.get_fact(fact1)['reliability_rating'].name})")
            print(f"    * {kg.get_fact(fact2)['fact_statement']} ({kg.get_fact(fact2)['reliability_rating'].name})")
    else:
        print("  No contradictions found")
    
    # Suggest missing links
    suggestions = analyzer.suggest_missing_links(threshold=0.5)
    print("\nSuggested missing links:")
    for fact1, fact2, score in suggestions[:3]:  # Show top 3
        print(f"  - {fact1} and {fact2} (similarity: {score:.4f})")
        print(f"    * {kg.get_fact(fact1)['fact_statement']}")
        print(f"    * {kg.get_fact(fact2)['fact_statement']}")
    
    # Visualize the knowledge graph
    print("\nVisualizing knowledge graph...")
    fig = plot_knowledge_graph(kg.graph, title="Solar System Knowledge Graph")
    
    # Save the visualization
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "knowledge_graph.png"))
    plt.close(fig)
    
    # Plot reliability distribution
    fig2 = plot_reliability_distribution(kg.graph)
    fig2.savefig(os.path.join(output_dir, "reliability_distribution.png"))
    plt.close(fig2)
    
    # Get graph statistics
    stats = get_graph_statistics(kg.graph)
    print("\nGraph statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    # Filter facts by reliability
    verified_kg = filter_facts_by_reliability(kg, ReliabilityRating.VERIFIED)
    print(f"\nFiltered to only VERIFIED facts: {len(verified_kg.graph.nodes)} facts remaining")
    
    # Export to JSON
    json_file = os.path.join(output_dir, "knowledge_graph.json")
    export_to_json(kg, json_file)
    print(f"\nExported knowledge graph to {json_file}")
    
    # Export to GEXF for visualization in tools like Gephi
    gexf_file = os.path.join(output_dir, "knowledge_graph.gexf")
    export_to_gexf(kg.graph, gexf_file)
    print(f"Exported knowledge graph to {gexf_file}")
    
    print("\nAdvanced knowledge graph operations completed successfully.")

if __name__ == "__main__":
    main()
