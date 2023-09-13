import unittest

import numpy as np
import pandas as pd
from y0.graph import NxMixedGraph

from src.eliater.algorithm.identify.conditional_independencies import evaluate_test, to_pgmpy_dag


class NxMixedGraphToDAG(unittest.TestCase):

    def test_to_pgmpy_dag(self):
        pass


class TestEvaluate(unittest.TestCase):

    def test_pearson(self):
        # Data Generation
        num_samples = 1000
        seed = 1
        np.random.seed(seed)
        W = np.random.normal(loc=10, scale=1, size=num_samples)
        X = np.random.normal(loc=W * 0.7, scale=3, size=num_samples)
        Z = np.random.normal(loc=X * 0.4, scale=2, size=num_samples)
        Y = np.random.normal(loc=Z * 0.5 + W * 0.3, scale=6)
        data = pd.DataFrame({'W': W, 'Z': Z, 'X': X, 'Y': Y})

        graph = NxMixedGraph.from_str_adj(
            directed={
                "W": ["X", "Y"],
                "X": ["Z"],
                "Z": ["Y"]
            }
        )
        # TODO: Finish the test case implementation

    def test_chi_square(self):
        pass
