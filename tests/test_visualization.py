import importlib.util
import os
import tempfile
import unittest
from datetime import datetime
import networkx as nx
from knowledge_graph_pkg.core import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.visualization import (
    plot_knowledge_graph,
    export_to_gexf,
    export_to_graphml,
    get_graph_statistics,
    plot_reliability_distribution
)

_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


class TestVisualization(unittest.TestCase):

    @unittest.skipUnless(_HAS_MATPLOTLIB, "matplotlib not installed")
    def test_plot_knowledge_graph_and_tier_colors(self):
        import matplotlib.colors as mcolors

        g = nx.DiGraph()
        g.add_node("A", reliability_rating=ReliabilityRating.VERIFIED)
        g.add_node("B", reliability_rating=ReliabilityRating.VERIFIED)
        g.add_edge("A", "B")
        
        fig = plot_knowledge_graph(g, title="Test Plot")
        self.assertIsNotNone(fig)

        fig2 = plot_reliability_distribution(g)
        self.assertIsNotNone(fig2)
        patches = fig2.axes[0].patches
        self.assertEqual(len(patches), 1)
        # VERIFIED-only graph must plot in darkgreen, not red!
        self.assertEqual(patches[0].get_facecolor(), mcolors.to_rgba("darkgreen"))

    def test_exports_handle_enum_list_and_none_attributes(self):
        kg = KnowledgeGraph()
        now = datetime.now()
        kg.add_fact(
            "f1", "Earth is round.", "Science", ["earth", "planet"],
            now, now, ReliabilityRating.VERIFIED, "src", "Title", "Author",
            now, "https://example.com", ["f2"], "Notes", "public", 1
        )
        kg.graph.nodes["f1"]["optional_meta"] = None
        kg.graph.add_node("f2", reliability_rating="LIKELY_TRUE", related_facts=[])
        kg.graph.add_edge("f1", "f2", tags=["rel"], flag=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            gexf_path = os.path.join(tmpdir, "graph.gexf")
            export_to_gexf(kg.graph, gexf_path)
            self.assertTrue(os.path.exists(gexf_path))

            graphml_path = os.path.join(tmpdir, "graph.graphml")
            export_to_graphml(kg.graph, graphml_path)
            self.assertTrue(os.path.exists(graphml_path))

    def test_statistics_handles_populated_and_empty_graphs(self):
        empty = nx.DiGraph()
        empty_stats = get_graph_statistics(empty)
        self.assertEqual(empty_stats["node_count"], 0)
        self.assertFalse(empty_stats["is_connected"])
        self.assertEqual(empty_stats["average_clustering"], 0.0)

        g = nx.DiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        stats = get_graph_statistics(g)
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)
        self.assertIn("density", stats)
