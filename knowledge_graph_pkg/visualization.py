"""
Visualization utilities for knowledge graphs.

This module provides functions for visualizing knowledge graphs using
various plotting libraries and export formats.
"""

from enum import Enum
import json
from typing import Dict, Any, Tuple
import networkx as nx

_TIER_ORDER = ['UNVERIFIED', 'POSSIBLY_TRUE', 'LIKELY_TRUE', 'VERIFIED']
_TIER_COLORS = {
    'UNVERIFIED': 'red',
    'POSSIBLY_TRUE': 'orange',
    'LIKELY_TRUE': 'lightgreen',
    'VERIFIED': 'darkgreen',
}


def _sanitize_attr_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sanitized_export_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Return a copy of ``graph`` with GEXF/GraphML-safe primitive attributes."""
    copy_g = graph.__class__()
    for k, v in graph.graph.items():
        copy_g.graph[k] = _sanitize_attr_value(v)
    for node, data in graph.nodes(data=True):
        clean_attrs = {k: _sanitize_attr_value(v) for k, v in data.items()}
        copy_g.add_node(node, **clean_attrs)
    for u, v, data in graph.edges(data=True):
        clean_attrs = {k: _sanitize_attr_value(val) for k, val in data.items()}
        copy_g.add_edge(u, v, **clean_attrs)
    return copy_g


def plot_knowledge_graph(graph: nx.DiGraph, 
                         title: str = "Knowledge Graph", 
                         node_size: int = 800,
                         node_color: str = "skyblue",
                         edge_color: str = "gray",
                         font_size: int = 10,
                         with_labels: bool = True,
                         figsize: Tuple[int, int] = (12, 8)) -> Any:
    """
    Plot a knowledge graph using matplotlib and networkx.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    pos = nx.spring_layout(graph, seed=42)
    nx.draw(graph, pos, 
            with_labels=with_labels,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            font_size=font_size,
            ax=ax)
    plt.title(title)
    return fig


def export_to_gexf(graph: nx.DiGraph, filename: str) -> None:
    """
    Export a knowledge graph to GEXF format for visualization in tools like Gephi.
    """
    if not filename.endswith('.gexf'):
        filename += '.gexf'
    nx.write_gexf(_sanitized_export_graph(graph), filename)
    

def export_to_graphml(graph: nx.DiGraph, filename: str) -> None:
    """
    Export a knowledge graph to GraphML format.
    """
    if not filename.endswith('.graphml'):
        filename += '.graphml'
    nx.write_graphml(_sanitized_export_graph(graph), filename)


def get_graph_statistics(graph: nx.DiGraph) -> Dict[str, Any]:
    """
    Calculate and return statistics about the knowledge graph.
    """
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        return {
            'node_count': 0,
            'edge_count': 0,
            'density': 0.0,
            'is_directed': nx.is_directed(graph),
            'is_connected': False,
            'average_clustering': 0.0,
            'average_shortest_path_length': None,
        }

    is_connected = (
        nx.is_weakly_connected(graph) if nx.is_directed(graph) else nx.is_connected(graph)
    )
    stats = {
        'node_count': n_nodes,
        'edge_count': graph.number_of_edges(),
        'density': nx.density(graph),
        'is_directed': nx.is_directed(graph),
        'is_connected': is_connected,
        'average_clustering': nx.average_clustering(graph),
        'average_shortest_path_length': None,
    }
    
    if stats['is_connected']:
        try:
            stats['average_shortest_path_length'] = nx.average_shortest_path_length(graph)
        except nx.NetworkXError:
            stats['average_shortest_path_length'] = "Graph has multiple components"
    
    return stats


def plot_reliability_distribution(graph: nx.DiGraph, 
                                 figsize: Tuple[int, int] = (10, 6)) -> Any:
    """
    Plot the distribution of reliability ratings in the knowledge graph.
    """
    import matplotlib.pyplot as plt

    reliability_counts: Dict[str, int] = {}
    
    for _, data in graph.nodes(data=True):
        if 'reliability_rating' in data and data['reliability_rating'] is not None:
            raw = data['reliability_rating']
            rating = raw.name if isinstance(raw, Enum) else str(raw).strip().upper()
            if rating:
                reliability_counts[rating] = reliability_counts.get(rating, 0) + 1
    
    fig, ax = plt.subplots(figsize=figsize)
    
    def _rank(item: Tuple[str, int]) -> int:
        return _TIER_ORDER.index(item[0]) if item[0] in _TIER_ORDER else len(_TIER_ORDER)

    sorted_items = sorted(reliability_counts.items(), key=_rank)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [_TIER_COLORS.get(label, 'steelblue') for label in labels]
    
    if labels:
        ax.bar(labels, values, color=colors)
        for i, v in enumerate(values):
            ax.text(i, v + 0.1, str(v), ha='center')
    
    ax.set_xlabel('Reliability Rating')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Fact Reliability Ratings')
    
    return fig
