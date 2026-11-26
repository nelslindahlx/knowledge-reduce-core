import os
import sys
import tempfile

# Ensure package directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from knowledge_graph_pkg.consensus import ConsensusEngine
from knowledge_graph_pkg.kuzu_store import KuzuStore
from knowledge_graph_pkg.rag import GraphRAGRetriever


def main():
    print("=== STARTING KNOWLEDGEREDUCE TEST DRIVE ===")

    with tempfile.TemporaryDirectory(prefix="kr_test_drive_") as tmpdir:
        store_dir = os.path.join(tmpdir, "store")
        graph_db_path = os.path.join(tmpdir, "graph")
        sample_file = os.path.join(tmpdir, "test_sample.txt")

        os.makedirs(store_dir, exist_ok=True)

        text = (
            "Mitochondria are double-membrane-bound organelles. Mitochondria produce ATP. "
            "ATP is the energy currency of the cell. Mitochondria contain DNA. "
            "Mitochondrial DNA is inherited from the mother."
        )
        print("\n[1/4] Writing sample text to file:")
        print(f'  "{text}"')
        with open(sample_file, "w", encoding="utf-8") as fh:
            fh.write(text)

        print("\n[2/4] Running Consensus Engine extraction and graph ingestion...")
        engine = ConsensusEngine(store_dir=store_dir, graph_db_path=graph_db_path)
        report = engine.process_with_consensus(
            file_path=sample_file,
            engines=["svo"],
        )
        print(f"  Consensus complete. Report: {report}")

        print('\n[3/4] Running Graph-RAG hybrid retrieval for query: "mitochondria"')
        kuzu_store = KuzuStore(graph_db_path)
        try:
            retriever = GraphRAGRetriever(store=kuzu_store)

            pr_scores = retriever.calculate_pagerank()
            print("\nComputed Node Page-Rank Scores:")
            for node, score in pr_scores.items():
                print(f"  - Fact {node}: Score {score:.4f}")

            context = retriever.format_context("mitochondria", top_k=2, hops=2)
            print("\nRetrieved Markdown Prompt Context:")
            print(context)
        finally:
            kuzu_store.close()

    print("\n=== TEST DRIVE COMPLETE ===")


if __name__ == "__main__":
    main()
