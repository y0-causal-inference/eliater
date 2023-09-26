import unittest
import warnings

from y0.dsl import Variable
from y0.examples import frontdoor_backdoor
from y0.examples.frontdoor import generate_data_for_frontdoor
from y0.examples.frontdoor_backdoor import generate_data_for_frontdoor_backdoor
from y0.graph import NxMixedGraph

from eliater.examples import (
    continuous,
    multi_med,
    multi_med_confounder,
    multi_med_confounder_nuisance_var,
)
from eliater.examples.continuous import generate_random_continuous_data
from eliater.examples.multi_med import generate_data_for_multi_med
from eliater.examples.multi_med_confounder import generate_data_for_multi_med_confounder
from eliater.examples.multi_med_confounder_nuisance_var import (
    generate_data_for_multi_med_confounder_nuisance_var,
)
from eliater.workflow import choose_default_test, fix_graph, get_state_space_map


class TestWorkflow(unittest.TestCase):
    def test_get_space_map_for_frontdoor(self):
        """Test get_space_map for frontdoor."""
        frontdoor_data = generate_data_for_frontdoor(1000)
        expected_frontdoor_col_types = {
            Variable("X"): "discrete",
            Variable("Y"): "continuous",
            Variable("Z"): "continuous",
        }
        actual_frontdoor_col_types = get_state_space_map(frontdoor_data)
        self.assertEqual(expected_frontdoor_col_types, actual_frontdoor_col_types)

    def test_get_space_map_for_frontdoor_backdoor(self):
        """Test get_space_map for frontdoor_backdoor."""
        frontdoor_backdoor_data = generate_data_for_frontdoor_backdoor(1000)
        expected_frontdoor_backdoor_col_types = {
            Variable("X"): "discrete",
            Variable("Y"): "discrete",
            Variable("W"): "discrete",
            Variable("Z"): "discrete",
        }
        actual_frontdoor_backdoor_col_types = get_state_space_map(frontdoor_backdoor_data)
        self.assertEqual(expected_frontdoor_backdoor_col_types, actual_frontdoor_backdoor_col_types)

    def test_choose_default_test_for_discrete_data(self):
        """Test choose_default_test for discrete data."""
        # Test on discrete data
        frontdoor_backdoor_data = generate_data_for_frontdoor_backdoor(1000)
        expected_default_test = "chi-square"
        actual_default_test = choose_default_test(frontdoor_backdoor_data)
        self.assertEqual(expected_default_test, actual_default_test)

    def test_choose_default_test_for_continuous_data(self):
        """Test choose_default_test for continuous data."""
        expected_default_test = "pearson"
        actual_default_test = choose_default_test(generate_random_continuous_data(1000))
        self.assertEqual(expected_default_test, actual_default_test)

    def test_choose_default_test_for_mixed_data(self):
        """Test choose_default_test for mixed data."""
        frontdoor_data = generate_data_for_frontdoor(1000)
        self.assertRaises(NotImplementedError, choose_default_test, frontdoor_data)

    def test_fix_graph_for_invalid_input_test(self):
        """Test fix_graph for invalid input test."""
        self.assertRaises(Exception, fix_graph, multi_med, generate_data_for_multi_med(1000), "abc")

    def test_fix_graph_for_continuous_data_and_not_pearson(self):
        """Test fix_graph for continuous data when pearson is not chosen."""
        with warnings.catch_warnings(record=True) as w:
            fix_graph(continuous, generate_random_continuous_data(1000), "chi-square")
            self.assertTrue(len(w) > 0)
            # Iterate through the captured warnings and check for the specific message
            specific_warning_found = False
            for warning in w:
                if issubclass(warning.category, UserWarning):
                    warning_message = str(warning.message)
                    expected_message = (
                        "The data is continuous. Either discretize and use chi-square or use the "
                        "pearson."
                    )
                    if warning_message == expected_message:
                        specific_warning_found = True
                        break
            # Assert that the specific warning was found
            self.assertTrue(specific_warning_found)

    def test_fix_graph_for_discrete_data_and_pearson(self):
        """Test fix_graph for discrete data when pearson is chosen."""
        self.assertRaises(
            Exception,
            fix_graph,
            frontdoor_backdoor,
            generate_data_for_frontdoor_backdoor(1000),
            "pearson",
        )

    def test_fix_graph_for_multi_med(self):
        """Test fix_graph for multi_med."""
        actual_fixed_graph = fix_graph(multi_med, generate_data_for_multi_med(1000))
        expected_fixed_graph = NxMixedGraph.from_str_adj(
            directed={"X": ["M1"], "M1": ["M2"], "M2": ["Y"]}, undirected={"X": ["Y"], "M1": ["Y"]}
        )
        self.assertEqual(actual_fixed_graph, expected_fixed_graph)

    def test_fix_graph_for_multi_med_confounder(self):
        """Test fix_graph for multi_med_confounder."""
        # FIXME: This test fails sometimes
        actual_fixed_graph = fix_graph(
            graph=multi_med_confounder,
            data=generate_data_for_multi_med_confounder(40),
            significance_level=0.001,
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

    def test_fix_graph_for_multi_med_confounder_nuisance_var(self):
        """Test fix_graph for multi_med_confounder_nuisance_var."""
        # FIXME: This test fails
        actual_fixed_graph = fix_graph(
            graph=multi_med_confounder_nuisance_var,
            data=generate_data_for_multi_med_confounder_nuisance_var(100, seed=2000),
            significance_level=0.001,
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
