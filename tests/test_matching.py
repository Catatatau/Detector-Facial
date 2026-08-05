import unittest
import numpy as np
from matching.matcher import cosine_similarity, euclidean_distance

class TestMatching(unittest.TestCase):
    def test_cosine_similarity(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0)
        
        v3 = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(cosine_similarity(v1, v3), 0.0)

    def test_euclidean_distance(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([1.0, 0.0, 0.0])
        self.assertAlmostEqual(euclidean_distance(v1, v2), 0.0)
        
        v3 = np.array([1.0, 1.0, 0.0])
        self.assertAlmostEqual(euclidean_distance(v1, v3), 1.0)

if __name__ == '__main__':
    unittest.main()
