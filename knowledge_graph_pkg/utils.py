"""
Utility functions for knowledge graph operations.

This module provides helper functions for common knowledge graph operations
such as importing data from various sources, filtering, and data transformation.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from urllib.parse import urlparse
from .core import KnowledgeGraph, ReliabilityRating


def _parse_reliability(val: Any) -> ReliabilityRating:
    if isinstance(val, ReliabilityRating):
        return val
    if isinstance(val, str):
        member = ReliabilityRating.__members__.get(val.strip().upper())
        if member is not None:
            return member
    return ReliabilityRating.UNVERIFIED


def _add_node_data_to_kg(kg: KnowledgeGraph, fact_id: str, raw_data: Dict[str, Any]) -> None:
    """Safely insert serialized or CSV-derived node attributes into ``kg``."""
    data = dict(raw_data)
    data.pop('quality_score', None)
    now = datetime.now().isoformat()

    raw_tags = data.get('tags', [])
    if isinstance(raw_tags, str):
        tags = [tag.strip() for tag in raw_tags.split(',') if tag.strip()]
    elif isinstance(raw_tags, (list, tuple)):
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    else:
        tags = []

    raw_related = data.get('related_facts', [])
    if isinstance(raw_related, str):
        related_facts = [item.strip() for item in raw_related.split(',') if item.strip()]
    elif isinstance(raw_related, (list, tuple)):
        related_facts = [str(item).strip() for item in raw_related if str(item).strip()]
    else:
        related_facts = []

    try:
        usage_count = int(data.get('usage_count') or 0)
    except (TypeError, ValueError):
        usage_count = 0

    kg.add_fact(
        fact_id=fact_id,
        fact_statement=str(data.get('fact_statement', '')),
        category=str(data.get('category') or 'General'),
        tags=tags,
        date_recorded=data.get('date_recorded') or now,
        last_updated=data.get('last_updated') or now,
        reliability_rating=_parse_reliability(data.get('reliability_rating')),
        source_id=str(data.get('source_id') or ''),
        source_title=str(data.get('source_title') or ''),
        author_creator=str(data.get('author_creator') or ''),
        publication_date=data.get('publication_date') or now,
        url_reference=str(data.get('url_reference') or ''),
        related_facts=related_facts,
        contextual_notes=str(data.get('contextual_notes') or ''),
        access_level=str(data.get('access_level') or 'public'),
        usage_count=usage_count,
    )
    # Preserve any extra semantic/custom attributes without breaking add_fact.
    node = kg.graph.nodes[fact_id]
    for extra_key in ('subject', 'subject_type', 'predicate', 'object', 'object_type', 'question', 'answer'):
        if extra_key in data:
            node[extra_key] = data[extra_key]


def import_from_json(kg: KnowledgeGraph, json_file: str) -> int:
    """
    Import facts from a JSON file into a knowledge graph.
    """
    imported_count = 0
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for fact_id, fact_data in data.items():
        try:
            _add_node_data_to_kg(kg, fact_id, fact_data)
            imported_count += 1
        except Exception as e:
            print(f"Error importing fact {fact_id}: {e}")
            
    return imported_count


def export_to_json(kg: KnowledgeGraph, json_file: str) -> int:
    """
    Export facts from a knowledge graph to a JSON file atomically.
    """
    export_data = {}
    
    for node, data in kg.graph.nodes(data=True):
        node_data = dict(data)
        
        if 'reliability_rating' in node_data:
            rel = node_data['reliability_rating']
            node_data['reliability_rating'] = (
                rel.name if isinstance(rel, ReliabilityRating) else str(rel)
            )
            
        for key, value in node_data.items():
            if isinstance(value, datetime):
                node_data[key] = value.isoformat()
                
        export_data[node] = node_data

    parent = os.path.dirname(json_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{json_file}.tmp.{os.getpid()}"
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    os.replace(tmp_path, json_file)
        
    return len(export_data)


def import_from_csv(kg: KnowledgeGraph, csv_file: str, 
                   id_column: str = 'fact_id',
                   reliability_column: Optional[str] = 'reliability_rating') -> int:
    """
    Import facts from a CSV file into a knowledge graph.
    """
    imported_count = 0
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for raw_row in reader:
            try:
                row = dict(raw_row)
                fact_id = row.pop(id_column)
                if reliability_column and reliability_column in row and reliability_column != 'reliability_rating':
                    row['reliability_rating'] = row.pop(reliability_column)
                _add_node_data_to_kg(kg, fact_id, row)
                imported_count += 1
            except Exception as e:
                print(f"Error importing row: {e}")
                
    return imported_count


def extract_facts_from_url(url: str, 
                          fact_extractor: Callable,
                          reliability: ReliabilityRating = ReliabilityRating.UNVERIFIED,
                          timeout: float = 15.0) -> List[Dict[str, Any]]:
    """
    Extract facts from an HTTP(S) web URL using a custom extractor function.
    """
    parsed = urlparse(str(url or ""))
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Unsupported or invalid URL: {url!r}")

    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise ImportError(
            "extract_facts_from_url requires the [ingest] extra (requests, beautifulsoup4)."
        ) from exc

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    facts = fact_extractor(soup)
    
    current_time = datetime.now()
    for fact in facts:
        if 'reliability_rating' not in fact:
            fact['reliability_rating'] = reliability
        if 'date_recorded' not in fact:
            fact['date_recorded'] = current_time
        if 'last_updated' not in fact:
            fact['last_updated'] = current_time
        if 'source_title' not in fact:
            fact['source_title'] = soup.title.text if soup.title else url
        if 'url_reference' not in fact:
            fact['url_reference'] = url
            
    return facts


def filter_facts_by_reliability(kg: KnowledgeGraph, 
                              min_reliability: ReliabilityRating) -> KnowledgeGraph:
    """
    Create a new knowledge graph containing only facts with at least the specified reliability.
    """
    filtered_kg = KnowledgeGraph()
    
    for node, data in kg.graph.nodes(data=True):
        if _parse_reliability(data.get('reliability_rating')).value >= min_reliability.value:
            _add_node_data_to_kg(filtered_kg, node, data)

    for u, v, edata in kg.graph.edges(data=True):
        if u in filtered_kg.graph and v in filtered_kg.graph:
            filtered_kg.graph.add_edge(u, v, **edata)
            
    return filtered_kg


def merge_knowledge_graphs(kg1: KnowledgeGraph, 
                          kg2: KnowledgeGraph,
                          conflict_strategy: str = 'higher_reliability') -> KnowledgeGraph:
    """
    Merge two knowledge graphs into a new one.
    """
    if conflict_strategy not in ['higher_reliability', 'keep_first', 'keep_second']:
        raise ValueError(f"Invalid conflict strategy: {conflict_strategy}")
        
    merged_kg = KnowledgeGraph()
    
    for node, data in kg1.graph.nodes(data=True):
        _add_node_data_to_kg(merged_kg, node, data)
        
    for node, data in kg2.graph.nodes(data=True):
        if node not in merged_kg.graph:
            _add_node_data_to_kg(merged_kg, node, data)
        else:
            if conflict_strategy == 'keep_first':
                continue
            elif conflict_strategy == 'keep_second':
                _add_node_data_to_kg(merged_kg, node, data)
            elif conflict_strategy == 'higher_reliability':
                existing_reliability = _parse_reliability(
                    merged_kg.graph.nodes[node].get('reliability_rating')
                ).value
                new_reliability = _parse_reliability(data.get('reliability_rating')).value
                
                if new_reliability > existing_reliability:
                    _add_node_data_to_kg(merged_kg, node, data)

    for u, v, edata in kg1.graph.edges(data=True):
        if u in merged_kg.graph and v in merged_kg.graph:
            merged_kg.graph.add_edge(u, v, **edata)
    for u, v, edata in kg2.graph.edges(data=True):
        if u in merged_kg.graph and v in merged_kg.graph:
            merged_kg.graph.add_edge(u, v, **edata)
                        
    return merged_kg
