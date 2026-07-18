"""
Graph-RAG Pipeline: Multi-Hop Retrieval & Context Formatting.

This script demonstrates how to load a graph store, ingest structured facts,
perform a multi-hop traversal with PageRank weights, and generate a formatted
context prompt suitable for passing into a language model.
"""

import os
import shutil
import tempfile
from knowledge_graph_pkg.graph_store_factory import get_graph_store
from knowledge_graph_pkg.rag import GraphRAGRetriever

def main():
    # 1. Initialize a temporary local Kuzu graph database
    temp_dir = tempfile.mkdtemp(prefix="graph_rag_demo_")
    db_path = os.path.join(temp_dir, "kdb")
    print(f"Initializing temporary KuzuStore at: {db_path}\n")
    
    store = None
    try:
        store = get_graph_store(db_path)
        
        # 2. Ingest sample facts representing a mini semantic network
        # We will create nodes representing concepts: Mitochondria, ATP, Cells, and Aerobic Respiration.
        sample_facts = [
            {
                "subject": "Mitochondria",
                "predicate": "produce",
                "object": "ATP",
                "fact_statement": "Mitochondria produce ATP.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": ["gpt-4", "claude-3", "gemini-pro"]
            },
            {
                "subject": "ATP",
                "predicate": "fuels",
                "object": "cellular activities",
                "fact_statement": "ATP fuels cellular activities.",
                "domain": "biochemistry",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": ["gpt-4", "claude-3", "gemini-pro"]
            },
            {
                "subject": "Aerobic respiration",
                "predicate": "occurs in",
                "object": "Mitochondria",
                "fact_statement": "Aerobic respiration occurs in Mitochondria.",
                "domain": "biochemistry",
                "reliability_rating": "LIKELY_TRUE",
                "cross_model_agreement": 2,
                "quality_score": 1,
                "source_models": ["llama3-8b", "gpt-4"]
            },
            {
                "subject": "Mitochondria",
                "predicate": "contain",
                "object": "own DNA",
                "fact_statement": "Mitochondria contain their own DNA.",
                "domain": "genetics",
                "reliability_rating": "VERIFIED",
                "cross_model_agreement": 3,
                "quality_score": 1,
                "source_models": ["gpt-4", "claude-3", "gemini-pro"]
            }
        ]
        
        print("Ingesting sample biochemistry facts into the graph...")
        inserted = store.ingest_facts(sample_facts)
        print(f"Successfully ingested {inserted} facts.\n")
        
        # 3. Setup the GraphRAGRetriever
        # (We use the default TF-IDF/offline lenient retriever since no embedder is provided)
        retriever = GraphRAGRetriever(store)
        
        # 4. Perform a Multi-Hop query
        query = "Mitochondria ATP activities"
        print(f"Executing Graph-RAG Retrieval for query: '{query}'")
        
        # Retrieve direct + multi-hop facts (hops=2)
        retrieved_facts = retriever.retrieve(query, top_k=3, hops=2, pagerank_weight=0.4)
        
        print("\n--- Retrieved Facts ---")
        for i, fact in enumerate(retrieved_facts, 1):
            print(f"{i}. [{fact.get('reliability')}] {fact.get('statement')}")
            
        # 5. Format context for an LLM Prompt
        context_str = retriever.format_context(query, top_k=3, hops=2)
        
        print("\n--- Formatted LLM Context Prompt ---")
        prompt = (
            "You are a factual assistant. Answer the user's question using the provided "
            "knowledge graph context below. Pay attention to the reliability of each fact.\n\n"
            "=== KNOWLEDGE CONTEXT ===\n"
            f"{context_str}\n"
            "=========================\n\n"
            f"Question: Explain the relationship between Mitochondria and cellular energy.\n"
            "Answer:"
        )
        print(prompt)
        
    finally:
        # Cleanup temporary database files
        if store is not None:
            store.close()
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary store at: {temp_dir}")

if __name__ == "__main__":
    main()
