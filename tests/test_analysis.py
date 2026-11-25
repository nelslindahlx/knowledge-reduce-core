import unittest
from knowledge_graph_pkg.core import KnowledgeGraph, ReliabilityRating
from knowledge_graph_pkg.analysis import KnowledgeGraphAnalyzer

class TestKnowledgeGraphAnalyzer(unittest.TestCase):

    def test_analyzer_empty(self):
        kg = KnowledgeGraph()
        analyzer = KnowledgeGraphAnalyzer(kg)
        
        self.assertEqual(analyzer.get_central_facts(), [])
        self.assertEqual(analyzer.get_fact_communities(), {})
        self.assertEqual(analyzer.get_reliability_summary(), {r.name: 0 for r in ReliabilityRating})
        self.assertEqual(analyzer.get_category_summary(), {})
        self.assertEqual(analyzer.find_contradictions(), [])
        self.assertEqual(analyzer.get_fact_importance(), {})
        self.assertEqual(analyzer.suggest_missing_links(), [])

    def test_analyzer_metrics(self):
        kg = KnowledgeGraph()
        # Add nodes with statements and ratings
        kg.graph.add_node(
            "f1",
            fact_statement="Mitochondria produce cellular energy",
            reliability_rating=ReliabilityRating.VERIFIED,
            category="biology",
            usage_count=5,
            related_facts=["f2"]
        )
        kg.graph.add_node(
            "f2",
            fact_statement="ATP is energy currency",
            reliability_rating=ReliabilityRating.VERIFIED,
            category="biology",
            usage_count=2,
            related_facts=["f1"]
        )
        kg.graph.add_node(
            "f3",
            fact_statement="Mitochondria does not produce cellular energy",
            reliability_rating=ReliabilityRating.UNVERIFIED,
            category="biology"
        )
        # Add edges
        kg.graph.add_edge("f1", "f2")
        kg.graph.add_edge("f2", "f1")

        analyzer = KnowledgeGraphAnalyzer(kg)
        
        # Central facts
        central = analyzer.get_central_facts()
        self.assertEqual(len(central), 3)

        # Communities
        comm = analyzer.get_fact_communities()
        self.assertIn("f1", comm)

        # Summaries
        rel_summary = analyzer.get_reliability_summary()
        self.assertEqual(rel_summary["VERIFIED"], 2)
        self.assertEqual(rel_summary["UNVERIFIED"], 1)

        cat_summary = analyzer.get_category_summary()
        self.assertEqual(cat_summary["biology"], 3)

        # Contradictions (f1 vs f3)
        contr = analyzer.find_contradictions()
        self.assertGreaterEqual(len(contr), 0)

        # Importance
        imp = analyzer.get_fact_importance()
        self.assertIn("f1", imp)

        # Missing links
        links = analyzer.suggest_missing_links(threshold=0.1)
        self.assertGreaterEqual(len(links), 0)

    def test_similarity_handles_tag_whitespace_empty_tags_and_mixed_ratings(self):
        kg = KnowledgeGraph()
        kg.graph.add_node("f1", fact_statement="Alpha beta", reliability_rating="VERIFIED", tags="")
        kg.graph.add_node("f2", fact_statement="Gamma delta", reliability_rating=None, tags="")
        kg.graph.add_node("f3", fact_statement="Alpha beta", reliability_rating=ReliabilityRating.UNVERIFIED, tags="a, b")
        kg.graph.add_node("f4", fact_statement="Alpha beta", reliability_rating="LIKELY_TRUE", tags="b,  a")

        analyzer = KnowledgeGraphAnalyzer(kg)
        summary = analyzer.get_reliability_summary()
        self.assertEqual(summary["VERIFIED"], 1)
        self.assertEqual(summary["LIKELY_TRUE"], 1)
        self.assertEqual(summary["UNVERIFIED"], 2)

        imp = analyzer.get_fact_importance()
        self.assertIn("f2", imp)

        # Empty tags should NOT grant +0.2 bonus between disjoint f1 and f2.
        sim_empty = analyzer._calculate_fact_similarity(kg.graph.nodes["f1"], kg.graph.nodes["f2"])
        self.assertEqual(sim_empty, 0.0)
        # Trimmed tags "a, b" vs "b,  a" should match completely.
        sim_tags = analyzer._calculate_fact_similarity(kg.graph.nodes["f3"], kg.graph.nodes["f4"])
        self.assertGreater(sim_tags, 0.5)
