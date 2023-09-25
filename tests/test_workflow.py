import unittest
import warnings

from y0.dsl import Variable
from y0.examples import frontdoor_backdoor
from y0.examples.frontdoor import generate_data_for_frontdoor
from y0.examples.frontdoor_backdoor import generate_data_for_frontdoor_backdoor
from y0.graph import NxMixedGraph

from src.eliater.examples import continuous, multi_med, multi_med_confounder
from src.eliater.examples.continuous import generate_random_continuous_data
from src.eliater.examples.multi_med import generate_data_for_multi_med
from src.eliater.examples.multi_med_confounder import generate_data_for_multi_med_confounder
from src.eliater.workflow import choose_default_test, fix_graph, get_state_space_map


class TestWorkflow(unittest.TestCase):
    def test_get_space_map(self):
        """Test get_space_map for frontdoor and frontdoor_backdoor."""
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
        """Test choose_default_test for discrete, continuous and mixed data."""
        # Test on discrete data
        frontdoor_backdoor_data = generate_data_for_frontdoor_backdoor(1000)
        expected_default_test = "chi-square"
        actual_default_test = choose_default_test(frontdoor_backdoor_data)
        self.assertEqual(expected_default_test, actual_default_test)

        # Test on mixed data
        frontdoor_data = generate_data_for_frontdoor(1000)
        self.assertRaises(NotImplementedError, choose_default_test, frontdoor_data)

        # Test on continuous data
        expected_default_test = "pearson"
        actual_default_test = choose_default_test(generate_random_continuous_data(1000))
        self.assertEqual(expected_default_test, actual_default_test)

    def test_fix_graph(self):
        """Test fix_graph on invalid input, multi_med, multi_med_confounder."""
        # Continuous data and not pearson (Test the warning)
        with warnings.catch_warnings(record=True) as w:
            fix_graph(continuous, generate_random_continuous_data(1000), "chi-square")
            self.assertEqual(w[0].category, UserWarning)  # Check the warning category
            self.assertEqual(
                str(w[0].message),
                "The data is continuous. Either discretize and use chi-square or use "
                "the pearson.",
            )

        # Discrete data and pearson (Test the exception)
        self.assertRaises(
            Exception,
            fix_graph,
            frontdoor_backdoor,
            generate_data_for_frontdoor_backdoor(1000),
            "pearson",
        )

        # Input invalid test (Test the exception)
        self.assertRaises(Exception, fix_graph, multi_med, generate_data_for_multi_med(1000), "abc")

        actual_fixed_graph = fix_graph(multi_med, generate_data_for_multi_med(1000))
        expected_fixed_graph = NxMixedGraph.from_str_adj(
            directed={"X": ["M1"], "M1": ["M2"], "M2": ["Y"]}, undirected={"X": ["Y"], "M1": ["Y"]}
        )
        self.assertEqual(actual_fixed_graph, expected_fixed_graph)

        # FIXME: This test fails sometimes
        actual_fixed_graph = fix_graph(
            multi_med_confounder, generate_data_for_multi_med_confounder(40)
        )
        expected_fixed_graph = NxMixedGraph.from_str_adj(
            directed={
                "Z1": ["X", "Z2"],
                "X": ["M1"],
                "M1": ["M2"],
                "M2": ["Y"],
                "Z2": ["Z3"],
                "Z3": ["Y"],
            },
            undirected={"Z1": ["X"], "Y": ["Z2"]},
        )
        self.assertEqual(actual_fixed_graph, expected_fixed_graph)
