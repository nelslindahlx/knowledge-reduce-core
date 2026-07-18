# Master Plan: Phase II System Hardening & Performance Optimizations

This master plan details the remediation steps to address the performance, scalability, security, and domain-generalization gaps identified during the chief critic audit.

---

## 📋 Remediation Trackers

### 🚀 Track 1: SQLite WAL Mode & Concurrency Resilience
* **Objective**: Prevent SQLite `database is locked` errors in the Watcher daemon under high concurrent document ingestion.
* **Changes**:
  * Enable WAL (Write-Ahead Logging) journal mode: `PRAGMA journal_mode=WAL;`.
  * Set a busy timeout of 5000ms: `PRAGMA busy_timeout=5000;`.
* **Files**: [watcher.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/watcher.py)
* **Status**: **COMPLETED**

---

### 🔒 Track 2: JWT Signature & Expiration Hardening
* **Objective**: Transition token verification from a basic static secret comparison to real cryptographic signature and expiration validation using `PyJWT`.
* **Changes**:
  * Validate token structure, algorithm, signature, and expiration (`exp`) claims.
  * Fall back safely to standard verification if keys are not configured.
* **Files**: [mcp_server.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/mcp_server.py)
* **Status**: **COMPLETED**

---

### ⚡ Track 3: Incremental Entity Resolution & Blocking
* **Objective**: Optimize Jaccard similarity entity resolution from a full $O(N^2)$ global scan to an incremental, partition-blocked scan.
* **Changes**:
  * Implement prefix/first-character blocking to narrow comparison sets.
  * Restrict real-time daemon calls to newly ingested concept nodes and their 2-hop neighborhoods.
* **Files**: [entity_resolution.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/entity_resolution.py), [watcher.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/watcher.py)
* **Status**: **COMPLETED**

---

### 🎛️ Track 4: Configurable Ontology Heuristics
* **Objective**: Remove domain-specific (biology/chemistry) hardcoding from `OntologyDistiller` and support YAML-based rule mapping injections.
* **Changes**:
  * Accept an optional rules dictionary/configuration file path in `OntologyDistiller.__init__`.
  * Define fallbacks to standard rules.
* **Files**: [ontology.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/ontology.py)
* **Status**: **COMPLETED**

---

### 🛡️ Track 5: Conflict-Aware Graph-RAG walking
* **Objective**: Prevent the RAG path walker from traversing edges connected to active unresolved contradictions.
* **Changes**:
  * Add a conflict-pruning pass during Graph-RAG neighbor extraction.
* **Files**: [rag.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/rag.py)
* **Status**: **COMPLETED**

---

### 🔌 Track 6: Exposing Distill Ontology via MCP Server
* **Objective**: Allow connected LLMs and agents to query high-level schema architectures programmatically over the MCP protocol.
* **Changes**:
  * Register the `graph_distill_ontology` schema to `TOOL_SCHEMAS`.
  * Implement the dispatch method in `GraphTools`.
* **Files**: [graph_tool.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/graph_tool.py)
* **Status**: **COMPLETED**

---

### 🚂 Track 7: Cross-Platform SFT CLI Command (`train-sft`)
* **Objective**: Enable platform parity for supervised fine-tuning on CUDA-enabled GPUs and Linux server setups directly from the CLI.
* **Changes**:
  * Add `train-sft` subcommand parsing options, dispatch routing, and PyTorch/PEFT training script mapping.
* **Files**: [cli.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/cli.py), `tests/test_train_wrapper.py`
* **Status**: **COMPLETED**

---

### 🚗 Track 8: Quick Integration System Smoke Test (`test-drive`)
* **Objective**: Allow developers to quickly run a full end-to-end extraction, consensus ingestion, PageRank calculation, and Graph-RAG walk test.
* **Changes**:
  * Expose `test-drive` subcommand and map it to `scripts/test_drive.py`.
* **Files**: [cli.py](file:///Users/nelslindahl/.gemini/antigravity/scratch/knowledgereduce/knowledge_graph_pkg/cli.py), `tests/test_cli.py`
* **Status**: **COMPLETED**

---

## 🧪 Verification Plan

### Automated Tests
* Run `pytest` on all 333 tests to ensure complete system compliance.
