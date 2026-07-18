import unittest
import pytest
pytest.importorskip("matplotlib")
import os
import tempfile
import networkx as nx
from knowledge_graph_pkg.visualization import (
    plot_knowledge_graph,
    export_to_gexf,
    export_to_graphml,
    get_graph_statistics,
    plot_reliability_distribution
)

class TestVisualization(unittest.TestCase):

    def test_plot_knowledge_graph(self):
        from knowledge_graph_pkg.core import ReliabilityRating
        g = nx.DiGraph()
        g.add_node("A", reliability_rating=ReliabilityRating.VERIFIED)
        g.add_node("B", reliability_rating=ReliabilityRating.VERIFIED)
        g.add_edge("A", "B")
        
        fig = plot_knowledge_graph(g, title="Test Plot")
        self.assertIsNotNone(fig)

        fig2 = plot_reliability_distribution(g)
        self.assertIsNotNone(fig2)

    def test_exports(self):
        g = nx.DiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        with tempfile.TemporaryDirectory() as tmpdir:
            gexf_path = os.path.join(tmpdir, "graph.gexf")
            export_to_gexf(g, gexf_path)
            self.assertTrue(os.path.exists(gexf_path))

            graphml_path = os.path.join(tmpdir, "graph.graphml")
            export_to_graphml(g, graphml_path)
            self.assertTrue(os.path.exists(graphml_path))

    def test_statistics(self):
        g = nx.DiGraph()
        g.add_node("A")
        g.add_node("B")
        g.add_edge("A", "B")

        stats = get_graph_statistics(g)
        self.assertEqual(stats["node_count"], 2)
        self.assertEqual(stats["edge_count"], 1)
        self.assertIn("density", stats)
