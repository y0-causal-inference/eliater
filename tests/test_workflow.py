import unittest

import numpy as np
import pandas as pd

from src.eliater.workflow import choose_default_test, fix_graph, get_state_space_map
from y0.dsl import Variable
from y0.examples.frontdoor import generate_data_for_frontdoor
from y0.graph import NxMixedGraph
from y0.examples.frontdoor_backdoor import generate_data_for_frontdoor_backdoor
from src.eliater.examples import multi_med, multi_med_confounder
from src.eliater.examples.multi_med import generate_data_for_multi_med
from src.eliater.examples.multi_med_confounder import generate_data_for_multi_med_confounder


class TestWorkflow(unittest.TestCase):
    def test_get_space_map(self):
        frontdoor_data = generate_data_for_frontdoor(1000)
        expected_frontdoor_col_types = {
            Variable("X"): "discrete",
            Variable("Y"): "continuous",
            Variable("Z"): "continuous",
        }
        actual_frontdoor_col_types = get_state_space_map(frontdoor_data)
        self.assertEqual(expected_frontdoor_col_types, actual_frontdoor_col_types)

        frontdoor_backdoor_data = generate_data_for_frontdoor_backdoor(1000)
        expected_frontdoor_backdoor_col_types = {
            Variable("X"): "discrete",
            Variable("Y"): "discrete",
            Variable("W"): "discrete",
            Variable("Z"): "discrete",
        }
        actual_frontdoor_backdoor_col_types = get_state_space_map(frontdoor_backdoor_data)
        self.assertEqual(expected_frontdoor_backdoor_col_types, actual_frontdoor_backdoor_col_types)

    def test_choose_default_test(self):
        frontdoor_backdoor_data = generate_data_for_frontdoor_backdoor(1000)
        expected_default_test = "chi-square"
        actual_default_test = choose_default_test(frontdoor_backdoor_data)
        self.assertEqual(expected_default_test, actual_default_test)

        frontdoor_data = generate_data_for_frontdoor(1000)
        self.assertRaises(NotImplementedError, choose_default_test, frontdoor_data)

        num_samples = 1000
        seed = 1
        np.random.seed(seed)
        W = np.random.normal(loc=10, scale=1, size=num_samples)
        X = np.random.normal(loc=W * 0.7, scale=3, size=num_samples)
        Z = np.random.normal(loc=X * 0.4, scale=2, size=num_samples)
        Y = np.random.normal(loc=Z * 0.5 + W * 0.3, scale=6)
        data = pd.DataFrame({"W": W, "Z": Z, "X": X, "Y": Y})
        expected_default_test = "pearson"
        actual_default_test = choose_default_test(data)
        self.assertEqual(expected_default_test, actual_default_test)

    def test_fix_graph(self):

        actual_fixed_graph = fix_graph(multi_med, generate_data_for_multi_med(1000))
        expected_fixed_graph = NxMixedGraph.from_str_adj(
            directed={
                'X': ['M1'],
                'M1': ['M2'],
                'M2': ['Y']
            },
            undirected={
                'X': ['Y'],
                'M1': ['Y']
            }
        )
        self.assertEqual(actual_fixed_graph, expected_fixed_graph)

        actual_fixed_graph = fix_graph(multi_med_confounder, generate_data_for_multi_med_confounder(40))
        expected_fixed_graph = NxMixedGraph.from_str_adj(
            directed= {
                "Z1": ["X", "Z2"],
                "X": ["M1"],
                "M1": ["M2"],
                "M2": ["Y"],
                "Z2": ["Z3"],
                "Z3": ["Y"]
            },
            undirected={
                "Z1": ["X"],
                "Y": ["Z2"]
            }
        )
        self.assertEqual(actual_fixed_graph, expected_fixed_graph)
