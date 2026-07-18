import unittest
from knowledge_graph_pkg.embeddings import VectorIndex

class TestVectorIndex(unittest.TestCase):

    def test_empty_index(self):
        index = VectorIndex()
        self.assertEqual(index.query([1.0, 0.0]), [])

    def test_query_and_similarity(self):
        index = VectorIndex()
        index.add_vector("v1", [1.0, 0.0])
        index.add_vector("v2", [0.0, 1.0])
        index.add_vector("v3", [0.707, 0.707])

        # Exact match query
        res = index.query([1.0, 0.0], top_k=1)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "v1")
        self.assertAlmostEqual(res[0][1], 1.0, places=4)

        # Multi-match query
        res_all = index.query([0.0, 1.0], top_k=3)
        self.assertEqual(len(res_all), 3)
        self.assertEqual(res_all[0][0], "v2")
        self.assertAlmostEqual(res_all[0][1], 1.0, places=4)
        self.assertEqual(res_all[1][0], "v3")

    def test_vector_update_and_removal(self):
        index = VectorIndex()
        index.add_vector("v1", [1.0, 0.0])
        index.add_vector("v2", [0.0, 1.0])

        # Update v1 to match query, make v2 orthogonal
        index.add_vector("v1", [0.0, 1.0])
        index.add_vector("v2", [1.0, 0.0])
        res = index.query([0.0, 1.0], top_k=1)
        self.assertEqual(res[0][0], "v1")

        # Remove v1
        index.remove_vector("v1")
        res = index.query([0.0, 1.0], top_k=2)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0], "v2")

    def test_zero_norms(self):
        index = VectorIndex()
        index.add_vector("v1", [0.0, 0.0])
        index.add_vector("v2", [1.0, 0.0])

        res = index.query([1.0, 0.0], top_k=2)
        self.assertEqual(res[0][0], "v2")
        self.assertAlmostEqual(res[0][1], 1.0, places=4)
        self.assertEqual(res[1][0], "v1")
        self.assertAlmostEqual(res[1][1], 0.0, places=4)

        # Zero query vector
        res_zero = index.query([0.0, 0.0], top_k=2)
        self.assertEqual(len(res_zero), 2)
        self.assertAlmostEqual(res_zero[0][1], 0.0)
