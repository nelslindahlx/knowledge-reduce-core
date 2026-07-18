"""
Example script demonstrating basic usage of the KnowledgeGraph package.

This script shows how to create a knowledge graph, add facts with reliability ratings,
retrieve facts, and update them.
"""
from datetime import datetime
from knowledge_graph_pkg import KnowledgeGraph
from knowledge_graph_pkg.core import ReliabilityRating

def main():
    # Create a new knowledge graph
    print("Creating a new knowledge graph...")
    kg = KnowledgeGraph()
    
    # Add facts with different reliability ratings
    print("\nAdding facts with different reliability ratings...")
    
    # Fact 1: Verified fact
    kg.add_fact(
        fact_id="earth_round",
        fact_statement="The Earth is approximately spherical in shape.",
        category="Astronomy",
        tags=["earth", "planet", "shape"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.VERIFIED,
        source_id="nasa_gov",
        source_title="NASA Earth Facts",
        author_creator="NASA",
        publication_date=datetime.now(),
        url_reference="https://nasa.gov/earth-facts",
        related_facts=[],
        contextual_notes="Observed through multiple space missions",
        access_level="public",
        usage_count=100
    )
    
    # Fact 2: Likely true fact
    kg.add_fact(
        fact_id="mars_life",
        fact_statement="Mars may have once supported microbial life.",
        category="Astronomy",
        tags=["mars", "life", "microbial"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.LIKELY_TRUE,
        source_id="science_journal",
        source_title="Science Journal Mars Studies",
        author_creator="Dr. Mars Researcher",
        publication_date=datetime.now(),
        url_reference="https://example.com/mars-studies",
        related_facts=[],
        contextual_notes="Based on evidence of past water and organic compounds",
        access_level="public",
        usage_count=50
    )
    
    # Fact 3: Possibly true fact
    kg.add_fact(
        fact_id="jupiter_core",
        fact_statement="Jupiter's core may be partially dissolved.",
        category="Astronomy",
        tags=["jupiter", "planet", "core"],
        date_recorded=datetime.now(),
        last_updated=datetime.now(),
        reliability_rating=ReliabilityRating.POSSIBLY_TRUE,
        source_id="astronomy_today",
        source_title="Astronomy Today Journal",
        author_creator="Planetary Core Research Team",
        publication_date=datetime.now(),
        url_reference="https://example.com/jupiter-core",
        related_facts=[],
        contextual_notes="Based on gravitational measurements and models",
        access_level="public",
        usage_count=20
    )
    
    # Retrieve and display facts
    print("\nRetrieving facts:")
    for fact_id in ["earth_round", "mars_life", "jupiter_core"]:
        fact = kg.get_fact(fact_id)
        print(f"\nFact ID: {fact_id}")
        print(f"Statement: {fact['fact_statement']}")
        print(f"Reliability: {fact['reliability_rating'].name}")
        print(f"Quality Score: {fact['quality_score']}")
    
    # Update a fact
    print("\nUpdating the Mars fact with new reliability rating...")
    kg.update_fact("mars_life", 
                  reliability_rating=ReliabilityRating.VERIFIED,
                  usage_count=75,
                  contextual_notes="New evidence from Mars rover strengthens this hypothesis")
    
    # Display the updated fact
    updated_fact = kg.get_fact("mars_life")
    print(f"\nUpdated Fact ID: mars_life")
    print(f"Statement: {updated_fact['fact_statement']}")
    print(f"New Reliability: {updated_fact['reliability_rating'].name}")
    print(f"New Quality Score: {updated_fact['quality_score']}")
    print(f"New Notes: {updated_fact['contextual_notes']}")
    
    print("\nKnowledge graph operations completed successfully.")

if __name__ == "__main__":
    main()
