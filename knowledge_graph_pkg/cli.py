"""
Command-line interface for KnowledgeReduce.

Run the full pipeline -- ingest text, extract facts, optionally resolve
coreferences, filter for quality, distill, and write model-absorbable
output -- from a single command:

    python -m knowledge_graph_pkg distill input.txt -o train.jsonl \\
        --format chat --filter standard --coref

Subcommands:
    distill   Extract + distill a document into training data.
"""

import argparse
import sys
from typing import List, Optional

from .core import KnowledgeGraph, ReliabilityRating
from .semantic import SemanticKnowledgeGraph
from .distillation import KnowledgeDistiller
from .quality import FactQualityFilter


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="knowledge_graph_pkg",
        description="KnowledgeReduce: distill documents into model-absorbable facts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("distill", help="Extract and distill a document.")
    d.add_argument("input", help="Path to the input text document.")
    d.add_argument("-o", "--output", required=True, help="Output file path.")
    d.add_argument("--format", choices=["chat", "instruction", "text"],
                   default="chat", help="Output format (default: chat).")
    d.add_argument("--filter", choices=["none", "standard", "strict"],
                   default="standard", help="Quality filter (default: standard).")
    d.add_argument("--coref", action="store_true",
                   help="Resolve leading pronouns to named-entity antecedents.")
    d.add_argument("--engine", choices=["svo", "spacy"], default="svo",
                   help="Extraction engine (default: svo; spacy needs [nlp] extra).")
    d.add_argument("--max-object-len", type=int, default=80,
                   help="Max object length for the quality filter (default: 80).")
    d.add_argument("--dedup", type=float, default=0.9,
                   help="Dedup similarity threshold 0..1 (default: 0.9; 0 disables).")
    d.add_argument("--min-reliability",
                   choices=["unverified", "possibly_true", "likely_true", "verified"],
                   default="likely_true",
                   help="Minimum reliability to keep (default: likely_true).")
    d.add_argument("--split", type=float, default=None,
                   help="Train/val split ratio (e.g. 0.9). Writes <out> and "
                        "<out>.val. Default: no split.")
    d.add_argument("--max-tokens", type=int, default=None,
                   help="Cap output to ~N tokens (keeps highest-ranked facts).")
    d.add_argument("--dedup-store", default=None,
                   help="Path to a persistent fact store JSON for cross-run "
                        "dedup (skips facts seen in previous runs).")
    d.add_argument("--seed", type=int, default=42,
                   help="Random seed for the train/val split (default: 42).")

    e = sub.add_parser("eval", help="Score the extractor against a gold set.")
    e.add_argument("--gold", default="data/gold_set.json",
                   help="Path to the gold-set JSON (default: data/gold_set.json).")

    p = sub.add_parser("drop", help="Ingest a source into the knowledge store (one drop per effort).")
    p.add_argument("input", help="Path to the input document.")
    p.add_argument("--store", default="store",
                   help="Knowledge store directory (default: store).")
    p.add_argument("--filter", choices=["none", "standard", "strict"],
                   default="standard", help="Quality filter (default: standard).")
    p.add_argument("--coref", action="store_true",
                   help="Resolve leading pronouns before extraction.")
    p.add_argument("--engine", choices=["svo", "spacy"], default="svo",
                   help="Extraction engine (default: svo).")
    p.add_argument("--max-object-len", type=int, default=80,
                   help="Max object length for the quality filter (default: 80).")
    p.add_argument("--dedup", type=float, default=0.9,
                   help="Dedup similarity threshold 0..1 (default: 0.9).")
    p.add_argument("--min-reliability",
                   choices=["unverified", "possibly_true", "likely_true", "verified"],
                   default="likely_true",
                   help="Minimum reliability to keep (default: likely_true).")
    p.add_argument("--force", action="store_true",
                   help="Write a drop even if this exact source was already ingested.")

    c = sub.add_parser("catalog", help="Index the store and show stats / query facts.")
    c.add_argument("--store", default="store", help="Knowledge store directory.")
    c.add_argument("--source", help="Filter: facts from this source.")
    c.add_argument("--reliability",
                   choices=["UNVERIFIED", "POSSIBLY_TRUE", "LIKELY_TRUE", "VERIFIED"],
                   help="Filter: minimum reliability label.")
    c.add_argument("--category", help="Filter: facts in this category.")
    c.add_argument("--min-quality", type=int, help="Filter: minimum quality score.")
    c.add_argument("--limit", type=int, default=20, help="Max rows to print (default 20).")

    cp = sub.add_parser("compile", help="Compile a training set from the store (a reproducible view).")
    cp.add_argument("-o", "--output", required=True, help="Output file path.")
    cp.add_argument("--store", default="store", help="Knowledge store directory.")
    cp.add_argument("--format", choices=["chat", "instruction", "text"],
                    default="chat", help="Output format (default: chat).")
    cp.add_argument("--source", help="Filter: only facts from this source.")
    cp.add_argument("--reliability",
                    choices=["UNVERIFIED", "POSSIBLY_TRUE", "LIKELY_TRUE", "VERIFIED"],
                    help="Filter: reliability label.")
    cp.add_argument("--category", help="Filter: facts in this category.")
    cp.add_argument("--min-quality", type=int, help="Filter: minimum quality score.")
    cp.add_argument("--split", type=float, default=None,
                    help="Train/val split ratio (writes <out> + <out>.val).")
    cp.add_argument("--max-tokens", type=int, default=None,
                    help="Cap output to ~N tokens (highest quality first).")
    cp.add_argument("--seed", type=int, default=42, help="Split seed (default 42).")

    b = sub.add_parser("batch", help="Drop many sources (or a folder) into the store.")
    b.add_argument("inputs", nargs="*", help="Source files to ingest.")
    b.add_argument("--folder", help="Ingest all supported files in this folder.")
    b.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    b.add_argument("--store", default="store", help="Knowledge store directory.")
    b.add_argument("--filter", choices=["none", "standard", "strict"], default="standard")
    b.add_argument("--coref", action="store_true")
    b.add_argument("--engine", choices=["svo", "spacy"], default="svo")

    lc = sub.add_parser("lifecycle", help="Corpus lifecycle: promote / contradictions / reextract.")
    lc.add_argument("op", choices=["promote", "contradictions", "reextract"],
                    help="Operation to run over the store.")
    lc.add_argument("--store", default="store", help="Knowledge store directory.")
    lc.add_argument("--min-sources", type=int, default=2,
                    help="promote: min distinct sources to corroborate (default 2).")
    lc.add_argument("--engine", choices=["svo", "spacy"], default="svo",
                    help="reextract: extraction engine (default svo).")

    mp = sub.add_parser("model-probe",
                        help="Probe local models for facts and store them as model drops.")
    mp.add_argument("--models", required=True,
                    help="Comma-separated model names (e.g. qwen2.5:14b,phi4:latest).")
    mp.add_argument("--domains", required=True,
                    help="Comma-separated domains to probe (e.g. biochem,physics).")
    mp.add_argument("--store", default="store", help="Knowledge store directory.")
    mp.add_argument("--n-prompts", type=int, default=10,
                    help="Prompts per (model, domain) (default 10).")
    mp.add_argument("--backend", choices=["ollama", "llama-cpp", "openai", "gemini"], default="ollama",
                    help="Probe backend (default: ollama).")
    mp.add_argument("--model-path", help="Path to local GGUF model file (llama-cpp backend).")
    mp.add_argument("--api-key", help="API key (openai backend).")
    mp.add_argument("--base-url", help="Base URL for remote endpoint or vLLM (openai backend).")
    mp.add_argument("--host", default="http://localhost:11434",
                    help="Ollama host (default http://localhost:11434).")
    mp.add_argument("--seed", type=int, default=42, help="Probe seed (default 42).")
    mp.add_argument("--force", action="store_true",
                    help="Write a drop even if this model+domain was already probed.")

    md = sub.add_parser("model-distill",
                        help="Distill cross-model-corroborated facts from the store into shards.")
    md.add_argument("-o", "--output", required=True, help="Output file path.")
    md.add_argument("--store", default="store", help="Knowledge store directory.")
    md.add_argument("--format", choices=["chat", "instruction", "text"],
                    default="chat", help="Output format (default chat).")
    md.add_argument("--min-agreement", type=int, default=2,
                    help="Min distinct models that must agree on a fact (default 2).")
    md.add_argument("--min-reliability",
                    choices=["possibly_true", "likely_true", "verified"],
                    default="likely_true",
                    help="Min promoted reliability to keep (default likely_true).")
    md.add_argument("--similarity", type=float, default=0.8,
                    help="Jaccard threshold for clustering facts across models (default 0.8).")
    md.add_argument("--dedup", type=float, default=0.9,
                    help="Dedup similarity threshold 0..1 (default 0.9; 0 disables).")
    md.add_argument("--filter", choices=["none", "standard", "strict"],
                    default="standard", help="Quality filter (default standard).")
    md.add_argument("--max-object-len", type=int, default=80,
                    help="Max object length for the quality filter (default 80).")
    md.add_argument("--max-tokens", type=int, default=None,
                    help="Cap output to ~N tokens (highest-ranked first).")
    md.add_argument("--split", type=float, default=None,
                    help="Train/val split ratio (writes <out> + <out>.val).")
    md.add_argument("--seed", type=int, default=42, help="Split seed (default 42).")
    md.add_argument("--manifest", default=None,
                    help="Optional path to write a provenance manifest JSON.")

    gv = sub.add_parser("graveyard",
                        help="Batch-probe a fleet of models across domains into the store.")
    gv.add_argument("--models", default=None,
                    help="Comma-separated models; omit to auto-discover local Ollama models.")
    gv.add_argument("--domains", required=True,
                    help="Comma-separated domains to probe.")
    gv.add_argument("--store", default="store", help="Knowledge store directory.")
    gv.add_argument("--n-prompts", type=int, default=10,
                    help="Prompts per (model, domain) (default 10).")
    gv.add_argument("--host", default="http://localhost:11434",
                    help="Ollama host (default http://localhost:11434).")
    gv.add_argument("--seed", type=int, default=42, help="Probe seed (default 42).")
    gv.add_argument("--no-resume", action="store_true",
                    help="Re-probe all pairs even if checkpointed as done.")

    me = sub.add_parser("model-eval",
                        help="Evaluate corroborated store facts against a domain gold set.")
    me.add_argument("--store", default="store", help="Knowledge store directory.")
    me.add_argument("--gold", required=True, help="Path to the gold set JSON.")
    me.add_argument("--min-agreement", type=int, default=1,
                    help="Min distinct-model agreement to include a fact (default 1).")
    me.add_argument("--similarity", type=float, default=0.8,
                    help="Clustering Jaccard threshold for corroboration (default 0.8).")
    me.add_argument("--embed", action="store_true",
                    help="Use embeddings for gold matching (else lenient strings).")
    me.add_argument("--host", default="http://localhost:11434", help="Ollama host.")
    me.add_argument("--embedder", choices=["ollama", "sentence-transformers"], default="ollama",
                    help="Embedder backend (default: ollama).")
    me.add_argument("--embedder-model", default=None,
                    help="Custom embedder model name.")
    me.add_argument("--output", default=None, help="Optional path to write the report JSON.")
    me.add_argument("--ci", action="store_true",
                    help="Exit non-zero if quality gates fail.")

    gi = sub.add_parser("graph-ingest",
                        help="Load corroborated store facts into a KùzuDB graph.")
    gi.add_argument("--store", default="store", help="Knowledge store directory.")
    gi.add_argument("--graph-db", default="graph_db", help="KùzuDB path (default graph_db).")
    gi.add_argument("--min-agreement", type=int, default=1,
                    help="Min distinct-model agreement to include (default 1).")
    gi.add_argument("--similarity", type=float, default=0.8,
                    help="Clustering Jaccard threshold (default 0.8).")
    gi.add_argument("--embed", action="store_true",
                    help="Use embeddings for cross-model clustering.")
    gi.add_argument("--host", default="http://localhost:11434", help="Ollama host.")
    gi.add_argument("--embedder", choices=["ollama", "sentence-transformers"], default="ollama",
                    help="Embedder backend (default: ollama).")
    gi.add_argument("--embedder-model", default=None,
                    help="Custom embedder model name.")

    gr = sub.add_parser("graph-reason",
                        help="Execute graph reasoning algorithms on the KùzuDB graph.")
    gr.add_argument("--graph-db", default="graph_db", help="KùzuDB path (default graph_db).")
    gr.add_argument("--op", choices=["link", "contradictions", "transitive", "validate"],
                    required=True, help="Graph reasoning operation to perform.")

    wd = sub.add_parser("watch-daemon",
                        help="Start a directory watcher daemon to automatically distill and ingest documents.")
    wd.add_argument("--dir", default="data/ingest_watch", help="Directory to watch (default: data/ingest_watch).")
    wd.add_argument("--store", default="store", help="Knowledge store directory (default: store).")
    wd.add_argument("--db-log", default="watcher_state.db", help="Path to SQLite watcher log database.")
    wd.add_argument("--engine", choices=["svo", "spacy"], default="svo",
                    help="Extraction engine (default: svo).")
    wd.add_argument("--coref", action="store_true", help="Enable pronoun coreference resolution.")
    wd.add_argument("--min-reliability", choices=["verified", "likely_true", "possibly_true", "unverified"],
                    default="likely_true", help="Min reliability threshold (default: likely_true).")
    wd.add_argument("--filter", default="standard", help="Fact quality filter name (default: standard).")
    wd.add_argument("--graph-db", default=None, help="Optional path to a Kùzu graph database to update in real-time.")
    wd.add_argument("--distill-dir", default=None, help="Optional output directory to save distilled ontology reports.")

    co = sub.add_parser("consensus",
                        help="Ingest a document using multiple engines, load into graph, and reconcile contradictions.")
    co.add_argument("input", help="Input file path (txt, md, html, pdf).")
    co.add_argument("--engines", nargs="+", default=["svo"], choices=["svo", "spacy"],
                    help="List of extraction engines to run (choices: svo, spacy).")
    co.add_argument("--store", default="store", help="Knowledge store directory (default: store).")
    co.add_argument("--graph-db", default="graph_db", help="KùzuDB path (default: graph_db).")
    co.add_argument("--coref", action="store_true", help="Enable pronoun coreference resolution.")
    co.add_argument("--min-reliability", choices=["verified", "likely_true", "possibly_true", "unverified"],
                    default="likely_true", help="Min reliability threshold (default: likely_true).")
    co.add_argument("--filter", default="standard", help="Fact quality filter name (default: standard).")

    cr = sub.add_parser("crawl",
                        help="Recursively crawl a model backend's weights starting from a seed topic.")
    cr.add_argument("--seed", required=True, help="Seed topic to start the crawl.")
    cr.add_argument("--backend", choices=["ollama", "llama-cpp", "openai", "gemini"], default="ollama",
                    help="Probing backend (default: ollama).")
    cr.add_argument("--model", required=True, help="Model name or GGUF model path.")
    cr.add_argument("--max-depth", type=int, default=2, help="Max recursion depth (default: 2).")
    cr.add_argument("--concepts-per-level", type=int, default=3,
                    help="Max new concepts to queue per level (default: 3).")
    cr.add_argument("--logprob-threshold", type=float, default=-1.5,
                    help="Min token log probability threshold (default: -1.5).")
    cr.add_argument("--store", default="store", help="Knowledge store directory (default: store).")
    cr.add_argument("--engine", choices=["svo", "spacy"], default="svo",
                    help="Extraction engine (default: svo).")

    ct = sub.add_parser("critique",
                        help="Audit and critique facts using a high-capability model backend to flag hallucinations.")
    ct.add_argument("--graph-db", default="graph_db", help="Path to the Kùzu graph database.")
    ct.add_argument("--backend", choices=["ollama", "llama-cpp", "openai", "gemini", "none"], default="gemini",
                    help="Critique backend (default: gemini, use 'none' for offline rule-based heuristics).")
    ct.add_argument("--model", default="gemini-1.5-flash", help="Critic model name.")
    ct.add_argument("--store", default=None, help="Knowledge store directory (instead of graph-db).")

    cs = sub.add_parser("compile-sft",
                        help="Compile facts from KùzuDB or store into standard SFT instruction training formats.")
    cs.add_argument("-o", "--output", required=True, help="Output JSONL/JSON file path.")
    cs.add_argument("--store", default=None, help="Knowledge store directory to load facts from.")
    cs.add_argument("--graph-db", default="graph_db", help="Path to the Kùzu graph database to load facts from.")
    cs.add_argument("--format", choices=["alpaca", "sharegpt", "plain"], default="alpaca",
                    help="SFT training format (default: alpaca).")


    tr = sub.add_parser("train",
                        help="Perform local LoRA fine-tuning on Apple Silicon via mlx-lm.")
    tr.add_argument("--model", required=True, help="Path to local MLX model or Hugging Face repo.")
    tr.add_argument("--data", default="data/train", help="Directory with train.jsonl and valid.jsonl (default: data/train).")
    tr.add_argument("--adapter-path", default="adapters", help="Output directory for adapters (default: adapters).")
    tr.add_argument("--iters", type=int, default=100, help="Number of training iterations (default: 100).")
    tr.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4).")
    tr.add_argument("--lr", type=float, default=1e-5, help="Learning rate (default: 1e-5).")
    tr.add_argument("--num-layers", type=int, default=16, help="Number of layers to fine-tune (default: 16).")

    sm = sub.add_parser("serve-mcp",
                        help="Serve the graph as LLM-callable tools over HTTP+JSON.")
    sm.add_argument("--graph-db", default="graph_db", help="KùzuDB path (default graph_db).")
    sm.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    sm.add_argument("--port", type=int, default=8080, help="Bind port (default 8080).")

    pr = sub.add_parser("model-prep",
                        help="Validate/filter a chat-JSONL shard into a training-ready SFT dataset.")
    pr.add_argument("input", help="Input chat-JSONL shard (e.g. from model-distill).")
    pr.add_argument("-o", "--output", required=True, help="Output SFT dataset path.")
    pr.add_argument("--min-reliability",
                    choices=["unverified", "possibly_true", "likely_true", "verified"],
                    default=None, help="Drop records below this reliability tier.")
    pr.add_argument("--stats", action="store_true", help="Print dataset stats after prep.")

    tm = sub.add_parser("train-mlx",
                        help="Fine-tune a model locally on Apple Silicon using MLX-LM.")
    tm.add_argument("--dataset", required=True, help="Prepared chat-JSONL dataset.")
    tm.add_argument("--base-model", default="mlx-community/Qwen2.5-3B-Instruct-4bit",
                    help="Hugging Face ID or local path to base model / mlx-quantized model.")
    tm.add_argument("--adapter-path", default="./adapters",
                    help="Directory to save the resulting LoRA adapters (default: ./adapters).")
    tm.add_argument("--iters", type=int, default=600,
                    help="Number of training iterations (default: 600).")
    tm.add_argument("--batch-size", type=int, default=4,
                    help="Training batch size (default: 4).")
    tm.add_argument("--lr", type=float, default=1e-5,
                    help="Learning rate (default: 1e-5).")
    tm.add_argument("--lora-layers", type=int, default=16,
                    help="Number of layers to apply LoRA to (default: 16).")
    tm.add_argument("--dry-run", action="store_true",
                    help="Validate dataset + print the command, then exit.")

    ts = sub.add_parser("train-sft",
                        help="Fine-tune a model using PyTorch, Hugging Face PEFT, and TRL (for CUDA/Linux/GPUs).")
    ts.add_argument("--dataset", required=True, help="Prepared chat-JSONL dataset.")
    ts.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct",
                    help="Hugging Face repo or local path to base model.")
    ts.add_argument("--output-dir", default="./lora-out",
                    help="Output directory for adapters (default: ./lora-out).")
    ts.add_argument("--epochs", type=int, default=1, help="Number of training epochs (default: 1).")
    ts.add_argument("--batch-size", type=int, default=1, help="Batch size per device (default: 1).")
    ts.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps (default: 8).")
    ts.add_argument("--lr", type=float, default=2e-4, help="Learning rate (default: 2e-4).")
    ts.add_argument("--lora-r", type=int, default=16, help="LoRA rank parameter (default: 16).")
    ts.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter (default: 32).")
    ts.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length (default: 1024).")
    ts.add_argument("--dry-run", action="store_true",
                    help="Validate dataset + print training configuration, then exit.")

    au = sub.add_parser("audit-store",
                        help="Perform diagnostic quality audits on the knowledge store.")
    au.add_argument("--store", default="store", help="Path to knowledge store (default: store).")

    td = sub.add_parser("test-drive",
                        help="Run a quick end-to-end smoke test verifying extraction, consensus, graph ingestion, and Graph-RAG.")
    td.add_argument("--crawl", action="store_true",
                    help="Also run local GGUF model download and weight crawling verification.")

    qg = sub.add_parser("query-graph",
                        help="Run interactive or one-shot Graph-RAG queries over the graph store.")
    qg.add_argument("query", help="The natural-language question or key terms to retrieve paths for.")
    qg.add_argument("--graph-db", default="graph_db", help="Path to Kùzu graph database (default: graph_db).")
    qg.add_argument("--top-k", type=int, default=5, help="Number of seed facts to retrieve (default: 5).")
    qg.add_argument("--hops", type=int, default=2, help="Max hops to traverse (default: 2).")
    qg.add_argument("--format", choices=["markdown", "json"], default="markdown",
                    help="Output format to print (default: markdown).")

    res = sub.add_parser("resolve-entities",
                         help="Perform entity resolution and merge synonym nodes in the Kùzu graph.")
    res.add_argument("--graph-db", default="graph_db", help="Path to the Kùzu graph database.")
    res.add_argument("--threshold", type=float, default=0.85,
                     help="Jaccard word-similarity threshold for merging entities (default: 0.85).")

    cg = sub.add_parser("compile-graph-instructions",
                         help="Compile multi-hop relationship chains from the graph into fine-tuning instructions.")
    cg.add_argument("--graph-db", default="graph_db", help="Path to the Kùzu graph database.")
    cg.add_argument("-o", "--output", required=True, help="Output JSONL instruction file path.")
    cg.add_argument("--max-chains", type=int, default=100,
                     help="Maximum number of relationship chains to compile (default: 100).")
    cg.add_argument("--include-contradictions", action="store_true",
                     help="Compile contradiction resolution instruction pairs alongside path walks.")

    do = sub.add_parser("distill-ontology",
                         help="Distill a high-level concept taxonomy and relationship schema from the graph.")
    do.add_argument("--graph-db", default="graph_db", help="Path to the Kùzu graph database.")
    do.add_argument("-o", "--output", required=True, help="Output JSON path to save the distilled ontology summary.")

    return parser


_RELIABILITY = {
    "unverified": ReliabilityRating.UNVERIFIED,
    "possibly_true": ReliabilityRating.POSSIBLY_TRUE,
    "likely_true": ReliabilityRating.LIKELY_TRUE,
    "verified": ReliabilityRating.VERIFIED,
}


def _make_filter(name: str, max_object_len: int) -> Optional[FactQualityFilter]:
    if name == "none":
        return None
    if name == "strict":
        return FactQualityFilter(max_object_len=min(max_object_len, 60),
                                 require_entity_subject=True)
    return FactQualityFilter(max_object_len=max_object_len)


def _cmd_distill(args) -> int:
    import os
    if not os.path.isfile(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    reliability = _RELIABILITY[args.min_reliability]

    # Resolve the extraction engine (svo default; spacy needs the [nlp] extra).
    extractor = None
    if getattr(args, "engine", "svo") != "svo":
        from .extractor_base import get_extractor
        try:
            extractor = get_extractor(args.engine)
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3

    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    ids = skg.create_facts_from_file(
        args.input, reliability=reliability, resolve_coref=args.coref,
        extractor=extractor,
    )

    quality_filter = _make_filter(args.filter, args.max_object_len)
    distiller = KnowledgeDistiller(
        kg,
        min_reliability=reliability,
        dedup_threshold=args.dedup,
        quality_filter=quality_filter,
    )

    # Build serialized records (one per selected fact).
    serializer = {
        "chat": distiller.to_chat_jsonl,
        "instruction": distiller.to_instruction_jsonl,
        "text": distiller.to_text,
    }[args.format]
    selected = distiller.select_facts()
    records = [line for line in serializer().splitlines() if line.strip()]

    # Cross-run dedup via a persistent fact store (by fact statement).
    if args.dedup_store:
        from .factstore import FactStore
        store = FactStore(path=args.dedup_store).load()
        kept_records = []
        for fact, rec in zip(selected, records):
            if store.add(fact.get("fact_statement", rec)):
                kept_records.append(rec)
        records = kept_records
        store.save()

    # Token budget: keep the highest-ranked records that fit.
    if args.max_tokens is not None:
        from .export import budget_records
        records = budget_records(records, args.max_tokens)

    def _write(path, lines):
        with open(path, "w", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")

    # Optional train/val split.
    if args.split is not None:
        from .export import split_records
        train, val = split_records(records, ratio=args.split, seed=args.seed)
        val_path = args.output + ".val"
        _write(args.output, train)
        _write(val_path, val)
        print(
            f"Extracted {len(ids)} raw facts; wrote {len(train)} train -> "
            f"{args.output} and {len(val)} val -> {val_path}."
        )
    else:
        _write(args.output, records)
        stats = distiller.stats()
        print(
            f"Extracted {len(ids)} raw facts; "
            f"wrote {len(records)} {args.format} pairs to {args.output} "
            f"(reduction ratio {stats['reduction_ratio']:.2f})."
        )
    return 0


def _cmd_eval(args) -> int:
    import os
    from .evaluation import load_gold_set, evaluate, format_report
    from .extraction import SVOExtractor
    if not os.path.isfile(args.gold):
        print(f"error: gold set not found: {args.gold}", file=sys.stderr)
        return 2
    report = evaluate(SVOExtractor(), load_gold_set(args.gold))
    print(format_report(report))
    return 0


def _cmd_drop(args) -> int:
    import os
    from .ingest import load_text
    from .store import KnowledgeStore, Drop, content_hash

    if not os.path.isfile(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    text = load_text(args.input)
    src_hash = content_hash(text)
    store = KnowledgeStore(args.store)

    # Idempotency: skip if this exact source content was already dropped.
    if not args.force and store.has_source_hash(src_hash):
        print(f"skip: source already ingested (hash {src_hash[:12]}); use --force to re-drop.")
        return 0

    reliability = _RELIABILITY[args.min_reliability]

    extractor = None
    if args.engine != "svo":
        from .extractor_base import get_extractor
        try:
            extractor = get_extractor(args.engine)
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 3

    kg = KnowledgeGraph()
    skg = SemanticKnowledgeGraph(kg)
    skg.create_facts_from_text(text, source_id=os.path.basename(args.input),
                               reliability=reliability, resolve_coref=args.coref,
                               extractor=extractor)

    quality_filter = _make_filter(args.filter, args.max_object_len)
    distiller = KnowledgeDistiller(kg, min_reliability=reliability,
                                   dedup_threshold=args.dedup,
                                   quality_filter=quality_filter)
    facts = distiller.select_facts()

    # Build a deterministic drop id from source basename + hash prefix.
    base = os.path.splitext(os.path.basename(args.input))[0]
    drop_id = f"{base}-{src_hash[:12]}"

    drop = Drop(
        drop_id=drop_id,
        source=args.input,
        source_hash=src_hash,
        facts=facts,
        engine=args.engine,
        filter_name=args.filter,
        coref=args.coref,
        source_text=text,
    )
    shard = store.write_drop(drop)
    print(
        f"Dropped {len(facts)} facts from {args.input} -> {shard} "
        f"(store now has {store.stats()['total_drops']} drops, "
        f"{store.stats()['total_facts']} facts)."
    )
    return 0


def _cmd_catalog(args) -> int:
    import os
    from .store import KnowledgeStore
    from .catalog import Catalog
    if not os.path.isdir(args.store):
        print(f"error: store not found: {args.store}", file=sys.stderr)
        return 2
    store = KnowledgeStore(args.store)
    cat = Catalog(os.path.join(args.store, "catalog.db"))
    cat.rebuild(store)

    any_filter = any([args.source, args.reliability, args.category, args.min_quality])
    if any_filter:
        rows = cat.query(source=args.source, reliability=args.reliability,
                         category=args.category, min_quality=args.min_quality,
                         limit=args.limit)
        print(f"{len(rows)} matching facts (showing up to {args.limit}):")
        for r in rows:
            print(f"  [{r['reliability']}/{r['quality']}] {r['statement']}  <- {r['source']}")
    else:
        s = cat.stats()
        print("Knowledge store catalog")
        print(f"  total facts: {s['total_facts']}")
        print(f"  total drops: {s['total_drops']}")
        print(f"  sources:     {s['sources']}")
        print("  by reliability:")
        for rel, n in sorted(s["by_reliability"].items()):
            print(f"    {rel}: {n}")
    cat.close()
    return 0


def _row_to_record(row, fmt):
    """Turn a catalog row into a serialized record line for the given format."""
    import json as _json
    q = row.get("question") or f"Tell me a fact about {row.get('category') or 'General'}."
    a = row.get("answer") or row.get("statement") or ""
    if fmt == "chat":
        return _json.dumps({"messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]}, ensure_ascii=False)
    if fmt == "instruction":
        return _json.dumps({"instruction": q, "input": "", "output": a}, ensure_ascii=False)
    # text
    return f"- {row.get('statement') or a}"


def _cmd_compile(args) -> int:
    import os
    from .store import KnowledgeStore
    from .catalog import Catalog
    if not os.path.isdir(args.store):
        print(f"error: store not found: {args.store}", file=sys.stderr)
        return 2

    store = KnowledgeStore(args.store)
    cat = Catalog(os.path.join(args.store, "catalog.db"))
    cat.rebuild(store)

    rows = cat.query(source=args.source, reliability=args.reliability,
                     category=args.category, min_quality=args.min_quality)
    records = [_row_to_record(r, args.format) for r in rows]
    drops_used = sorted({r["drop_id"] for r in rows})
    cat.close()

    if args.max_tokens is not None:
        from .export import budget_records
        records = budget_records(records, args.max_tokens)

    def _write(path, lines):
        with open(path, "w", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")

    if args.split is not None:
        from .export import split_records
        train, val = split_records(records, ratio=args.split, seed=args.seed)
        _write(args.output, train)
        _write(args.output + ".val", val)
        print(
            f"Compiled {len(records)} facts from {len(drops_used)} drop(s) -> "
            f"{len(train)} train ({args.output}) + {len(val)} val ({args.output}.val)."
        )
    else:
        _write(args.output, records)
        print(
            f"Compiled {len(records)} facts from {len(drops_used)} drop(s) -> "
            f"{args.output} (format {args.format})."
        )
    return 0


def _cmd_batch(args) -> int:
    import os
    from .factory import batch_drop, scan_folder, format_report
    sources = list(args.inputs or [])
    if args.folder:
        if not os.path.isdir(args.folder):
            print(f"error: folder not found: {args.folder}", file=sys.stderr)
            return 2
        sources += scan_folder(args.folder, recursive=args.recursive)
    if not sources:
        print("error: no inputs (pass files or --folder).", file=sys.stderr)
        return 2
    report = batch_drop(sources, args.store, filter_name=args.filter,
                        coref=args.coref, engine=args.engine)
    print(format_report(report))
    return 0


def _cmd_lifecycle(args) -> int:
    import os
    from .store import KnowledgeStore
    from .lifecycle import promote_reliability, find_contradictions, reextract_store
    if not os.path.isdir(args.store):
        print(f"error: store not found: {args.store}", file=sys.stderr)
        return 2
    store = KnowledgeStore(args.store)

    if args.op == "promote":
        proms = promote_reliability(store, min_sources=args.min_sources)
        print(f"{len(proms)} reliability promotion(s) suggested:")
        for p in proms:
            print(f"  {p['old_reliability']} -> {p['new_reliability']} "
                  f"({p['sources']} sources): {p['statement']}")
    elif args.op == "contradictions":
        conflicts = find_contradictions(store)
        print(f"{len(conflicts)} contradiction(s) found:")
        for c in conflicts:
            print(f"  {c['subject']} {c['predicate']} -> {', '.join(c['objects'])}")
    elif args.op == "reextract":
        result = reextract_store(store, engine=args.engine)
        print(f"Re-extraction complete: {result['reextracted']} reextracted, "
              f"{result['skipped']} skipped (no source text).")
    return 0


def _cmd_model_probe(args) -> int:
    """Probe local models across domains; store each (model, domain) as a ModelDrop."""
    from .store import KnowledgeStore
    from .model_drop import ModelDrop
    from .model_probe import ModelProbe, get_backend

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not models or not domains:
        print("error: --models and --domains must each name at least one item.",
              file=sys.stderr)
        return 2

    store = KnowledgeStore(args.store)
    total_drops = total_facts = skipped = 0
    for model in models:
        backend = get_backend(
            args.backend,
            model=model,
            host=args.host,
            model_path=args.model_path,
            api_key=args.api_key,
            base_url=args.base_url
        )
        probe = ModelProbe(backend=backend, model=model)
        for domain in domains:
            outputs = probe.probe_domain(domain, n_prompts=args.n_prompts, seed=args.seed)
            drop = ModelDrop.from_probe_outputs(model, domain, outputs,
                                                backend=args.backend)
            if not args.force and store.has_source_hash(drop.source_hash):
                print(f"skip: {model}/{domain} already probed "
                      f"(hash {drop.source_hash[:12]}); use --force to re-probe.")
                skipped += 1
                continue
            store.write_drop(drop)
            total_drops += 1
            total_facts += len(drop.facts)
            print(f"probed {model}/{domain}: {len(drop.facts)} facts "
                  f"from {args.n_prompts} prompts.")

    print(f"Done: {total_drops} drop(s), {total_facts} facts added "
          f"({skipped} skipped). Store now has "
          f"{store.stats()['total_drops']} drops, {store.stats()['total_facts']} facts.")
    return 0


def _cmd_model_distill(args) -> int:
    import os
    from .store import KnowledgeStore
    from .model_distill import ModelKnowledgeDistiller

    if not os.path.isdir(args.store):
        print(f"error: store not found: {args.store}", file=sys.stderr)
        return 2

    min_rel = args.min_reliability.upper()
    quality_filter = _make_filter(args.filter, args.max_object_len)
    store = KnowledgeStore(args.store)
    distiller = ModelKnowledgeDistiller.from_store(
        store, min_agreement=args.min_agreement, min_reliability=min_rel,
        similarity_threshold=args.similarity, dedup_threshold=args.dedup,
        quality_filter=quality_filter)

    serializer = {
        "chat": distiller.to_chat_jsonl,
        "instruction": distiller.to_instruction_jsonl,
        "text": distiller.to_text,
    }[args.format]
    records = [ln for ln in serializer().splitlines() if ln.strip()]

    if args.max_tokens is not None:
        from .export import budget_records
        records = budget_records(records, args.max_tokens)

    def _write(path, lines):
        with open(path, "w", encoding="utf-8") as fh:
            for ln in lines:
                fh.write(ln + "\n")

    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as fh:
            import json as _json
            _json.dump(distiller.manifest(os.path.basename(args.output)), fh,
                       ensure_ascii=False, indent=2)

    if args.split is not None:
        from .export import split_records
        train, val = split_records(records, ratio=args.split, seed=args.seed)
        _write(args.output, train)
        _write(args.output + ".val", val)
        print(f"Distilled {len(records)} corroborated facts -> {len(train)} train "
              f"({args.output}) + {len(val)} val ({args.output}.val).")
    else:
        _write(args.output, records)
        m = distiller.manifest()
        print(f"Distilled {len(records)} corroborated facts -> {args.output} "
              f"(VERIFIED={m['verified']}, LIKELY_TRUE={m['likely_true']}, "
              f"models={m['models']}).")
    return 0


def _cmd_graveyard(args) -> int:
    """Batch-probe models x domains into the store, with resume + report."""
    from .store import KnowledgeStore
    from .model_drop import ModelDrop
    from .model_probe import ModelProbe, OllamaBackend
    from .graveyard import run_graveyard, discover_ollama_models

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if not domains:
        print("error: --domains must name at least one domain.", file=sys.stderr)
        return 2

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        try:
            models = discover_ollama_models(host=args.host)
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not auto-discover models: {exc}", file=sys.stderr)
            return 3
        if not models:
            print("error: no local models discovered; pass --models explicitly.",
                  file=sys.stderr)
            return 2
        print(f"Discovered {len(models)} model(s): {', '.join(models)}")

    # Ollama-backed prober: probe one (model, domain), write a ModelDrop,
    # return the fact count. Idempotency is handled by the orchestrator's
    # checkpoint; we also guard on the store's content hash.
    def _prober(model, domain, store, n_prompts=10, seed=42, **kw):
        backend = OllamaBackend(model=model, host=args.host)
        probe = ModelProbe(backend=backend, model=model)
        outputs = probe.probe_domain(domain, n_prompts=n_prompts, seed=seed)
        drop = ModelDrop.from_probe_outputs(model, domain, outputs, backend="ollama")
        if store.has_source_hash(drop.source_hash):
            return 0
        store.write_drop(drop)
        return len(drop.facts)

    report = run_graveyard(
        models=models, domains=domains, store_dir=args.store,
        prober=_prober, resume=not args.no_resume,
        n_prompts=args.n_prompts, seed=args.seed, progress=True,
    )
    print()
    print(report.render())
    return 0


def _cmd_model_eval(args) -> int:
    """Evaluate corroborated store facts against a domain gold set + gates."""
    import os
    from .store import KnowledgeStore
    from .model_distill import ModelKnowledgeDistiller
    from .model_eval import ModelShardEvaluator, check_gates, format_report

    if not os.path.isdir(args.store):
        print(f"error: store not found: {args.store}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.gold):
        print(f"error: gold set not found: {args.gold}", file=sys.stderr)
        return 2

    store = KnowledgeStore(args.store)

    embedder = None
    if args.embed:
        try:
            from .embeddings import get_embedder
            embedder = get_embedder(args.embedder, model=args.embedder_model, host=args.host)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: embeddings unavailable ({exc}); using string matching.",
                  file=sys.stderr)

    distiller = ModelKnowledgeDistiller.from_store(
        store, min_agreement=args.min_agreement, min_reliability="POSSIBLY_TRUE",
        similarity_threshold=args.similarity, embedder=embedder)
    shard_facts = distiller.select_facts()

    evaluator = ModelShardEvaluator(embedder=embedder)
    report = evaluator.evaluate(shard_facts, args.gold)
    print(format_report(report))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            import json as _json
            _json.dump(report, fh, ensure_ascii=False, indent=2)

    passed, failures = check_gates(report)
    if failures:
        print("\nQuality gates FAILED:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("\nQuality gates passed.")

    if args.ci and not passed:
        return 1
    return 0


def _cmd_graph_ingest(args) -> int:
    """Distill corroborated facts from the store and load them into KùzuDB."""
    import os
    from .store import KnowledgeStore
    from .model_distill import ModelKnowledgeDistiller
    if not os.path.isdir(args.store):
        print(f"error: store not found: {args.store}", file=sys.stderr)
        return 2
    try:
        from .graph_store_factory import get_graph_store
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    embedder = None
    if args.embed:
        try:
            from .embeddings import get_embedder
            embedder = get_embedder(args.embedder, model=args.embedder_model, host=args.host)
        except Exception as exc:  # noqa: BLE001
            print(f"warning: embeddings unavailable ({exc}); using Jaccard.",
                  file=sys.stderr)

    store = KnowledgeStore(args.store)
    distiller = ModelKnowledgeDistiller.from_store(
        store, min_agreement=args.min_agreement, min_reliability="POSSIBLY_TRUE",
        similarity_threshold=args.similarity, embedder=embedder)
    facts = distiller.select_facts()

    kstore = get_graph_store(args.graph_db)
    kstore.ingest_facts(facts)
    print(f"Ingested {len(facts)} facts into Graph DB at {args.graph_db} "
          f"(graph now has {kstore.count()} facts).")
    return 0


def _cmd_serve_mcp(args) -> int:
    """Start the HTTP graph-tool server."""
    from .mcp_server import serve
    serve(args.graph_db, host=args.host, port=args.port)
    return 0


def _cmd_model_prep(args) -> int:
    """Validate/filter a chat-JSONL shard into a training-ready SFT dataset."""
    import os
    from .training_prep import prepare_sft_dataset, dataset_stats
    if not os.path.isfile(args.input):
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2
    min_rel = args.min_reliability.upper() if args.min_reliability else None
    report = prepare_sft_dataset(args.input, args.output, min_reliability=min_rel)
    print(f"Prepared {report['kept']} SFT records -> {args.output} "
          f"({report['dropped']} dropped of {report['total']}).")
    if args.stats:
        stats = dataset_stats(args.output)
        print(f"  records:          {stats['records']}")
        print(f"  by reliability:   {stats['by_reliability']}")
        print(f"  avg user chars:   {stats['avg_user_chars']}")
        print(f"  avg asst chars:   {stats['avg_assistant_chars']}")
    return 0


def _cmd_train_mlx(args) -> int:
    """Fine-tune a model locally on Apple Silicon using MLX-LM."""
    from scripts import train_mlx
    argv = [
        "--dataset", args.dataset,
        "--base-model", args.base_model,
        "--adapter-path", args.adapter_path,
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--lora-layers", str(args.lora_layers)
    ]
    if args.dry_run:
        argv.append("--dry-run")
    return train_mlx.main(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "distill":
        return _cmd_distill(args)
    if args.command == "eval":
        return _cmd_eval(args)
    if args.command == "drop":
        return _cmd_drop(args)
    if args.command == "catalog":
        return _cmd_catalog(args)
    if args.command == "compile":
        return _cmd_compile(args)
    if args.command == "batch":
        return _cmd_batch(args)
    if args.command == "lifecycle":
        return _cmd_lifecycle(args)
    if args.command == "model-probe":
        return _cmd_model_probe(args)
    if args.command == "model-distill":
        return _cmd_model_distill(args)
    if args.command == "graveyard":
        return _cmd_graveyard(args)
    if args.command == "model-eval":
        return _cmd_model_eval(args)
    if args.command == "graph-ingest":
        return _cmd_graph_ingest(args)
    if args.command == "serve-mcp":
        return _cmd_serve_mcp(args)
    if args.command == "model-prep":
        return _cmd_model_prep(args)
    if args.command == "train-mlx":
        return _cmd_train_mlx(args)
    if args.command == "graph-reason":
        return _cmd_graph_reason(args)
    if args.command == "watch-daemon":
        return _cmd_watch_daemon(args)
    if args.command == "consensus":
        return _cmd_consensus(args)
    if args.command == "train":
        return _cmd_train(args)
    if args.command == "crawl":
        return _cmd_crawl(args)
    if args.command == "critique":
        return _cmd_critique(args)
    if args.command == "compile-sft":
        return _cmd_compile_sft(args)
    if args.command == "audit-store":
        return _cmd_audit_store(args)
    if args.command == "resolve-entities":
        return _cmd_resolve_entities(args)
    if args.command == "compile-graph-instructions":
        return _cmd_compile_graph_instructions(args)
    if args.command == "distill-ontology":
        return _cmd_distill_ontology(args)
    if args.command == "query-graph":
        return _cmd_query_graph(args)
    if args.command == "train-sft":
        return _cmd_train_sft(args)
    if args.command == "test-drive":
        return _cmd_test_drive(args)
    parser.print_help()
    return 1


def _cmd_graph_reason(args) -> int:
    """Execute graph reasoning algorithms on the KùzuDB graph."""
    import os
    import sys
    import json
    try:
        from .graph_store_factory import get_graph_store
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    kstore = get_graph_store(args.graph_db)
    try:
        if args.op == "link":
            n_links = kstore.auto_link_relations()
            print(f"Graph auto-linking complete: created {n_links} RELATED edges.")
        elif args.op == "contradictions":
            contras = kstore.find_contradictions()
            print(f"Found {len(contras)} contradiction(s):")
            for c in contras:
                print(f"  * Conflict between:")
                print(f"    - Fact A: {c['a_stmt']} [quality: {c['a_qual']}, rel: {c['a_rel']}]")
                print(f"    - Fact B: {c['b_stmt']} [quality: {c['b_qual']}, rel: {c['b_rel']}]")
        elif args.op == "transitive":
            infers = kstore.find_transitive_inferences()
            print(f"Found {len(infers)} transitive inference suggestion(s):")
            for row in infers:
                print(f"  * Chain: {row['step1']} -> {row['step2']}")
                print(f"    Propose direct relation: {row['subject']} -> {row['predicate']} -> {row['object']}")
        elif args.op == "validate":
            result = kstore.validate_and_reconcile()
            demoted = result["demoted"]
            print(f"Path validation complete: demoted {len(demoted)} contradictory facts to UNVERIFIED:")
            for d in demoted:
                print(f"  - [{d['block_id']}] {d['statement']}")
    finally:
        kstore.close()
    return 0


def _cmd_watch_daemon(args) -> int:
    """Start the directory watcher daemon."""
    try:
        from .watcher import WatcherDaemon
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    watcher = WatcherDaemon(
        watch_dir=args.dir,
        store_dir=args.store,
        db_log_path=args.db_log,
        reliability=args.min_reliability,
        filter_name=args.filter,
        coref=args.coref,
        engine=args.engine,
        graph_db=args.graph_db,
        distill_dir=args.distill_dir
    )
    try:
        watcher.run()
    except KeyboardInterrupt:
        print("\nWatcher daemon stopped.")
    return 0


def _cmd_consensus(args) -> int:
    """Run extraction with multiple engines and validate consistency."""
    import os
    if not os.path.isfile(args.input):
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    try:
        from .consensus import ConsensusEngine
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    engine = ConsensusEngine(store_dir=args.store, graph_db_path=args.graph_db)
    try:
        result = engine.process_with_consensus(
            file_path=args.input,
            engines=args.engines,
            reliability=args.min_reliability,
            filter_name=args.filter,
            coref=args.coref
        )
        demoted = result.get("demoted", [])
        if demoted:
            print(f"Consensus reconciliation complete: demoted {len(demoted)} conflicting facts to UNVERIFIED.")
        else:
            print("Consensus reconciliation complete: no conflicts detected.")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 4
    return 0


def _cmd_train(args) -> int:
    """Run local LoRA training via mlx-lm."""
    try:
        from .train import MLXTrainer
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    trainer = MLXTrainer(
        model_path=args.model,
        data_dir=args.data,
        adapter_path=args.adapter_path
    )
    try:
        trainer.train_lora(
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            num_layers=args.num_layers
        )
    except Exception as exc:
        print(f"error during training: {exc}", file=sys.stderr)
        return 4
    return 0


def _cmd_crawl(args) -> int:
    """Run model weight crawler."""
    try:
        from .model_probe import get_backend
        from .crawler import ModelCrawler
        from .store import KnowledgeStore, Drop, content_hash
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    try:
        backend = get_backend(args.backend, model=args.model, model_path=args.model)
    except Exception as exc:
        print(f"error instantiating backend: {exc}", file=sys.stderr)
        return 4

    crawler = ModelCrawler(backend=backend)
    try:
        facts = crawler.crawl(
            seed_topic=args.seed,
            max_depth=args.max_depth,
            concepts_per_level=args.concepts_per_level,
            logprob_threshold=args.logprob_threshold,
            engine=args.engine
        )
        if not facts:
            print("Crawl complete: no facts extracted.")
            return 0

        store = KnowledgeStore(args.store)
        text_summary = "\n".join(f.get("fact_statement", "") for f in facts)
        src_hash = content_hash(text_summary)
        
        drop_id = f"crawl-{args.seed.replace(' ', '_').lower()}-{src_hash[:12]}"
        drop = Drop(
            drop_id=drop_id,
            source=f"model-crawl:{args.seed}",
            source_hash=src_hash,
            facts=facts,
            engine=args.engine,
            filter_name="none",
            coref=False,
            source_text=text_summary
        )
        shard = store.write_drop(drop)
        print(f"Saved {len(facts)} crawled facts into store drop shard: {shard}")
    except Exception as exc:
        print(f"error during crawl: {exc}", file=sys.stderr)
        return 5
    return 0


def _cmd_critique(args) -> int:
    """Audit and critique facts using a high-capability model backend."""
    import os
    import sys
    try:
        from .critique import FactCritic
        from .graph_store_factory import get_graph_store
        from .store import KnowledgeStore
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    # Load facts
    facts = []
    kstore = None
    if args.store:
        store = KnowledgeStore(args.store)
        facts = list(store.iter_facts())
        print(f"Loaded {len(facts)} facts from KnowledgeStore '{args.store}'.")
    else:
        kstore = get_graph_store(args.graph_db)
        facts = kstore.query(
            "MATCH (f:Fact) "
            "RETURN f.block_id AS block_id, f.statement AS fact_statement, "
            "f.subject AS subject, f.predicate AS predicate, f.object AS object"
        )
        print(f"Loaded {len(facts)} facts from Graph Database '{args.graph_db}'.")

    if not facts:
        print("No facts found to critique.")
        if kstore:
            kstore.close()
        return 0

    print(f"Auditing facts using backend '{args.backend}' and model '{args.model}'...")
    try:
        critic = FactCritic(args.backend, model_name=args.model)
        reports = critic.critique_facts(facts)
    except Exception as exc:
        print(f"error instantiating or running critic: {exc}", file=sys.stderr)
        if kstore:
            kstore.close()
        return 4

    flagged = [r for r in reports if not r["is_factual"]]
    print(f"\nAudit Complete: Flagged {len(flagged)} / {len(reports)} facts as non-factual.")
    for f in flagged:
        print(f"  - Flagged: \"{f['statement']}\"")
        print(f"    Reasoning: {f['reasoning']}")

    if kstore and flagged:
        print("\nUpdating Kùzu Graph Database with critic feedback...")
        demote_count = 0
        for f in flagged:
            try:
                kstore.query(
                    "MATCH (fac:Fact) "
                    "WHERE fac.block_id = $bid "
                    "SET fac.reliability = 'UNVERIFIED', fac.quality = 0",
                    {"bid": f["block_id"]}
                )
                demote_count += 1
            except Exception as exc:
                print(f"Failed to update fact {f['block_id']}: {exc}", file=sys.stderr)
        print(f"Successfully demoted {demote_count} facts in the graph database.")

    if kstore:
        kstore.close()

    return 0


def _cmd_compile_sft(args) -> int:
    """Compile facts into standard SFT instruction training formats."""
    import os
    import sys
    try:
        from .graph_store_factory import get_graph_store
        from .store import KnowledgeStore
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    facts = []
    if args.store:
        store = KnowledgeStore(args.store)
        facts = list(store.iter_facts())
    else:
        kstore = get_graph_store(args.graph_db)
        facts = kstore.query(
            "MATCH (f:Fact) "
            "RETURN f.statement AS fact_statement, f.subject AS subject"
        )
        kstore.close()

    if not facts:
        print("No facts found to compile.")
        return 0

    compiled = []
    for f in facts:
        statement = f.get("fact_statement") or f.get("statement") or ""
        subj = f.get("subject") or "General"
        if not statement:
            continue
            
        if args.format == "alpaca":
            compiled.append({
                "instruction": f"State a verified fact about {subj}.",
                "input": "",
                "output": statement
            })
        elif args.format == "sharegpt":
            compiled.append({
                "conversations": [
                    {"from": "human", "value": f"State a verified fact about {subj}."},
                    {"from": "gpt", "value": statement}
                ]
            })
        elif args.format == "plain":
            compiled.append({
                "text": statement
            })

    try:
        import json
        with open(args.output, "w", encoding="utf-8") as fh:
            if args.output.endswith(".jsonl"):
                for item in compiled:
                    fh.write(json.dumps(item) + "\n")
            else:
                json.dump(compiled, fh, indent=2)
        print(f"Successfully compiled {len(compiled)} facts into SFT format '{args.format}' at '{args.output}'.")
    except Exception as exc:
        print(f"error writing SFT dataset: {exc}", file=sys.stderr)
        return 4

    return 0


def _cmd_audit_store(args) -> int:
    """Perform a diagnostic quality audit on the knowledge store."""
    import os
    import sys
    from .store import KnowledgeStore
    
    if not os.path.isdir(args.store):
        print(f"error: store not found: {args.store}", file=sys.stderr)
        return 2
        
    store = KnowledgeStore(args.store)
    report = store.audit_summary()
    
    print("=== Knowledge Store Audit ===")
    print(f"Total Drops: {report['total_drops']}")
    print(f"Total Facts: {report['total_facts']}")
    print("\nReliability Tier Distribution:")
    for tier, count in sorted(report['reliability_tier_distribution'].items()):
        print(f"  * {tier}: {count}")
    print(f"\nMissing SVO Fields: {report['missing_fields_count']}")
    print(f"Duplicate SVO Triplets: {report['duplicate_svo_count']}")
    return 0


def _cmd_resolve_entities(args) -> int:
    """Perform entity resolution and merge synonym nodes in the Kùzu graph."""
    import sys
    try:
        from .graph_store_factory import get_graph_store
        from .entity_resolution import resolve_and_merge_entities
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    kstore = get_graph_store(args.graph_db)
    try:
        res = resolve_and_merge_entities(kstore, threshold=args.threshold)
        print("Entity Resolution Complete:")
        print(f"  - Resolved clusters: {res['resolved_clusters']}")
        print(f"  - Merged redundant nodes: {res['merged_nodes']}")
    finally:
        kstore.close()
    return 0


def _cmd_compile_graph_instructions(args) -> int:
    """Compile multi-hop relationship chains from the graph into SFT instructions."""
    import sys
    try:
        from .graph_store_factory import get_graph_store
        from .graph_compiler import compile_subgraph_instructions, save_compiled_instructions
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    kstore = get_graph_store(args.graph_db)
    try:
        from .graph_compiler import compile_contradiction_instructions
        
        instructions = compile_subgraph_instructions(kstore, max_chains=args.max_chains)
        if args.include_contradictions:
            contra_instructions = compile_contradiction_instructions(kstore)
            instructions.extend(contra_instructions)
            print(f"Added {len(contra_instructions)} contradiction resolution instruction(s).")
            
        save_compiled_instructions(instructions, args.output)
        print(f"Successfully compiled {len(instructions)} total instructions to {args.output}")
    finally:
        kstore.close()
    return 0


def _cmd_distill_ontology(args) -> int:
    """Distill the high-level taxonomy and relation schema from the graph."""
    import sys
    import json
    try:
        from .graph_store_factory import get_graph_store
        from .ontology import OntologyDistiller
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    kstore = get_graph_store(args.graph_db)
    try:
        distiller = OntologyDistiller(kstore)
        taxonomy = distiller.distill_taxonomy()
        semantic_types = distiller.infer_semantic_types()
        schema = distiller.infer_relation_schema()
        
        summary = {
            "taxonomy": taxonomy,
            "semantic_types": semantic_types,
            "relation_schema": schema
        }
        
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
            
        print("Ontology Distillation Complete:")
        print(f"  - Extracted {len(taxonomy)} taxonomic hierarchies.")
        print(f"  - Categorized {len(semantic_types)} concepts.")
        print(f"  - Inferred {len(schema)} schema relationships.")
        print(f"  - Summary saved to: {args.output}")
    finally:
        kstore.close()
    return 0


def _cmd_query_graph(args) -> int:
    """Run Graph-RAG retrieval over the store and print a premium styled output."""
    import sys
    import json
    try:
        from .graph_store_factory import get_graph_store
        from .rag import GraphRAGRetriever
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3

    kstore = get_graph_store(args.graph_db)
    try:
        retriever = GraphRAGRetriever(store=kstore)
        results = retriever.retrieve(args.query, top_k=args.top_k, hops=args.hops)
        
        if args.format == "json":
            print(json.dumps(results, indent=2))
        else:
            # Color-coded Markdown rendering in terminal
            print("\033[1;33m" + "=" * 60 + "\033[0m")
            print(f"\033[1;36m🔍 Graph-RAG Multi-Hop Path Retrieval\033[0m")
            print(f"\033[1;30mQuery:\033[0m \"{args.query}\"")
            print("\033[1;33m" + "=" * 60 + "\033[0m")
            
            if not results:
                print("\033[1;31mNo matching facts found in the knowledge graph.\033[0m")
                return 0
                
            print(f"\033[1;32mFound {len(results)} relevant facts:\033[0m")
            for idx, r in enumerate(results, 1):
                reliability = r.get("reliability", "UNVERIFIED")
                color = "\033[1;32m" if reliability == "VERIFIED" else "\033[1;34m"
                print(f"  \033[1;35m{idx}.\033[0m \033[1m{r['subject']}\033[0m {r['predicate']} \033[1m{r['object']}\033[0m")
                print(f"     \033[3m\"{r['statement']}\"\033[0m (Reliability: {color}{reliability}\033[0m)")
                
            print("\033[1;33m" + "-" * 60 + "\033[0m")
            print("\033[1;36mCompiled LLM Prompt Context:\033[0m")
            print(retriever.format_context(args.query, top_k=args.top_k, hops=args.hops))
            print("\033[1;33m" + "=" * 60 + "\033[0m")
    finally:
        kstore.close()
    return 0


def _cmd_train_sft(args) -> int:
    """Run supervised fine-tuning using PyTorch, PEFT, and TRL."""
    from scripts import train_sft
    argv = [
        "--dataset", args.dataset,
        "--base-model", args.base_model,
        "--output-dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--lr", str(args.lr),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--max-seq-len", str(args.max_seq_len)
    ]
    if args.dry_run:
        argv.append("--dry-run")
    return train_sft.main(argv)


def _cmd_test_drive(args) -> int:
    """Run a quick end-to-end smoke test of the entire system."""
    from scripts import test_drive
    try:
        test_drive.main()
        if args.crawl:
            from scripts import download_and_crawl
            download_and_crawl.main()
        return 0
    except Exception as exc:
        import sys
        print(f"Error running test drive: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
