"""
Agent Integration Examples: LangChain and LlamaIndex.

This module provides copy-pasteable examples for integrating the
knowledge-reduce-core graph-RAG retrievers and HTTP MCP server into
standard LLM agent frameworks.
"""

import os
import json
import requests
from typing import List, Dict, Any

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
        # Call the serve-mcp FastAPI endpoint
        url = "http://127.0.0.1:8080/query"
        headers = {
            "Authorization": "Bearer sample_token",
            "X-Workspace-Id": workspace_id
        }
        payload = {"query": query}
        
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                facts = data.get("facts", [])
                return json.dumps(facts, indent=2)
            else:
                return f"Error: MCP server returned status code {response.status_code}"
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
