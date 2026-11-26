"""
Agent Integration Examples: LangChain and LlamaIndex.

This module provides copy-pasteable examples for integrating the
knowledge-reduce-core graph-RAG retrievers and HTTP MCP server into
standard LLM agent frameworks.
"""

import json

# =====================================================================
# 1. LangChain Custom Tool Example
# =====================================================================

try:
    from langchain.tools import tool

    @tool
    def search_knowledge_graph(query: str, workspace_id: str = "default") -> str:
        """Search the Knowledge Graph store using the hosted MCP endpoint.
        
        Args:
            query: The search query (e.g. concept or SVO keywords).
            workspace_id: The tenant/workspace identifier.
            
        Returns:
            A JSON-formatted string of matching fact statements and ratings.
        """
        import urllib.request
        import urllib.error

        # Call the serve-mcp HTTP endpoint (/api/rag_retrieve).
        url = "http://127.0.0.1:8080/api/rag_retrieve"
        headers = {
            "Authorization": "Bearer sample_token",
            "Content-Type": "application/json",
            "X-Workspace-Id": workspace_id,
        }
        payload = json.dumps({"query": query}).encode("utf-8")

        try:
            req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
                facts = data.get("facts", [])
                return json.dumps(facts, indent=2)
        except Exception as exc:
            return f"Error connecting to MCP server: {exc}"

except ImportError:
    # Safe fallback if langchain is not installed
    def search_knowledge_graph(query: str, workspace_id: str = "default") -> str:
        return "LangChain is not installed in the environment."


# =====================================================================
# 2. LlamaIndex Custom QueryEngine Example
# =====================================================================

try:
    from llama_index.core.query_engine import CustomQueryEngine
    from llama_index.core.llms import LLM
    from llama_index.core.prompts import PromptTemplate
    from knowledge_graph_pkg.rag import GraphRAGRetriever

    class GraphRAGQueryEngine(CustomQueryEngine):
        """Custom LlamaIndex query engine wrapping GraphRAGRetriever."""

        retriever: GraphRAGRetriever
        llm: LLM
        qa_prompt: PromptTemplate

        def custom_query(self, query_str: str) -> str:
            # 1. Retrieve relevant facts using page-rank weighted similarity
            seeds = self.retriever.retrieve_seeds(query_str, limit=5)
            context = "\n".join([
                f"- [{f.get('reliability', 'UNVERIFIED')}] {f.get('statement')}" 
                for f in seeds
            ])
            
            # 2. Query LLM with context
            prompt = self.qa_prompt.format(context=context, query=query_str)
            response = self.llm.complete(prompt)
            return str(response)

except ImportError:
    # Safe fallback if LlamaIndex is not installed
    class GraphRAGQueryEngine:
        pass


if __name__ == "__main__":
    print("=== Agent Integration Examples ===")
    print("This file contains blueprints for LangChain and LlamaIndex agent integration.")
    print("Verify the MCP server is running on port 8080 before calling `search_knowledge_graph`.")
