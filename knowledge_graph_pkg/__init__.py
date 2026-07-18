"""
KnowledgeReduce Ultimate: A comprehensive knowledge graph framework.

This package provides advanced tools for creating, managing, and analyzing knowledge graphs
with reliability ratings, semantic capabilities, performance optimizations, real-time
streaming, vector embeddings, and blockchain verification.
"""

from .core import KnowledgeGraph, ReliabilityRating
from .enhanced import EnhancedKnowledgeGraph
from .semantic import SemanticKnowledgeGraph
from .sharding import ShardedKnowledgeGraph
from .vector import VectorKnowledgeGraph
from .streaming import StreamingKnowledgeGraph
from .blockchain import BlockchainKnowledgeGraph
from .distillation import KnowledgeDistiller
from .qa import QAGenerator
from .coref import resolve_coreferences
from .extraction import SVOExtractor
from .extractor_base import Extractor, get_extractor
from .quality import FactQualityFilter
from .ingest import load_text
from .export import split_records, budget_records, estimate_tokens
from .factstore import FactStore
from .store import KnowledgeStore, Drop, content_hash, SCHEMA_VERSION
from .catalog import Catalog
from .lifecycle import promote_reliability, find_contradictions, reextract_store
from .factory import batch_drop, scan_folder
from .model_drop import (
    ModelDrop, probe_output_to_facts, probe_outputs_to_facts,
    model_fact_statement, model_content_hash,
)
from .cross_model import CrossModelVerifier, reliability_for_agreement, jaccard, cluster_facts
from .model_distill import ModelKnowledgeDistiller

try:  # single source of truth: version comes from package metadata (pyproject)
    from importlib.metadata import version as _pkg_version, PackageNotFoundError
    try:
        __version__ = _pkg_version("knowledgereduce")
    except PackageNotFoundError:  # not installed (e.g. raw source tree)
        __version__ = "0.2.0"
except ImportError:  # pragma: no cover - very old Python
    __version__ = "0.2.0"
__all__ = [
    'KnowledgeGraph',
    'ReliabilityRating',
    'EnhancedKnowledgeGraph',
    'SemanticKnowledgeGraph',
    'ShardedKnowledgeGraph',
    'VectorKnowledgeGraph',
    'StreamingKnowledgeGraph',
    'BlockchainKnowledgeGraph',
    'KnowledgeDistiller',
    'QAGenerator',
    'resolve_coreferences',
    'SVOExtractor',
    'FactQualityFilter',
    'load_text',
    'Extractor',
    'get_extractor',
    'split_records',
    'budget_records',
    'estimate_tokens',
    'FactStore',
    'KnowledgeStore',
    'Drop',
    'content_hash',
    'SCHEMA_VERSION',
    'Catalog',
    'promote_reliability',
    'find_contradictions',
    'reextract_store',
    'batch_drop',
    'scan_folder',
    'ModelDrop',
    'probe_output_to_facts',
    'probe_outputs_to_facts',
    'model_fact_statement',
    'model_content_hash',
    'CrossModelVerifier',
    'reliability_for_agreement',
    'jaccard',
    'cluster_facts',
    'ModelKnowledgeDistiller'
]
