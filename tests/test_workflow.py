import unittest

import numpy as np
import pandas as pd

from src.eliater.workflow import choose_default_test, fix_graph, get_state_space_map
from y0.dsl import Variable
from y0.examples.frontdoor import generate_data_for_frontdoor
from y0.examples.frontdoor_backdoor import generate_data_for_frontdoor_backdoor


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
        pass
