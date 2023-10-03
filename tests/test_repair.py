"""This module tests the steps involved in repairing the network structure."""

import unittest
import warnings

from eliater.frontdoor_backdoor import (
    base_example,
    multi_mediators_confounders_nuisance_vars_example,
    multiple_mediators_confounders_example,
    multiple_mediators_single_confounder_example,
)
from eliater.repair import choose_default_test, get_state_space_map, repair_network
from y0.dsl import Variable
from y0.examples import frontdoor_backdoor
from y0.examples.frontdoor import generate_data_for_frontdoor
from y0.examples.frontdoor_backdoor import generate_data_for_frontdoor_backdoor
from y0.graph import NxMixedGraph

R1, R2, R3 = (Variable("R{i}") for i in (1, 2, 3))


class TestRepair(unittest.TestCase):
    """This class implements tests to verify the correctness of steps involved in repairing the network structure."""

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
        actual_default_test = choose_default_test(base_example.generate_data())
        self.assertEqual(expected_default_test, actual_default_test)

    def test_choose_default_test_for_mixed_data(self):
        """Test choose_default_test for mixed data."""
        frontdoor_data = generate_data_for_frontdoor(1000)
        self.assertRaises(NotImplementedError, choose_default_test, frontdoor_data)

    def test_repair_network_for_invalid_input_test(self):
        """Test repair_network for invalid input test."""
        self.assertRaises(
            Exception,
            repair_network,
            multiple_mediators_single_confounder_example,
            multiple_mediators_single_confounder_example.generate_data(),
            "abc",
        )

    def test_repair_network_for_continuous_data_and_not_pearson(self):
        """Test repair_network for continuous data when pearson is not chosen."""
        with warnings.catch_warnings(record=True) as w:
            repair_network(base_example.graph, base_example.generate_data(), "chi-square")
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

    def test_repair_network_for_discrete_data_and_pearson(self):
        """Test repair_network for discrete data when pearson is chosen."""
        self.assertRaises(
            Exception,
            repair_network,
            frontdoor_backdoor,
            generate_data_for_frontdoor_backdoor(1000),
            "pearson",
        )

    def test_repair_network_for_multi_mediators(self):
        """Test repair_network for multi_mediators."""
        actual_fixed_graph = repair_network(
            multiple_mediators_single_confounder_example.graph,
            multiple_mediators_single_confounder_example.generate_data(),
        )
        expected_fixed_graph = NxMixedGraph.from_str_adj(
            directed={"X": ["M1"], "M1": ["M2"], "M2": ["Y"]}, undirected={"X": ["Y"], "M1": ["Y"]}
        )
        self.assertEqual(actual_fixed_graph, expected_fixed_graph)

    def assert_graph_equal(self, g1: NxMixedGraph, g2: NxMixedGraph) -> None:
        """Assert two graphs are equal."""
        self.assertEqual(
            set(g1.directed.nodes()),
            set(g2.directed.nodes()),
            msg="Directed graphs have different nodes",
        )
        self.assertEqual(
            set(g1.undirected.nodes()),
            set(g2.undirected.nodes()),
            msg="Undirected graphs have different nodes",
        )
        self.assertEqual(
            g1.directed.edges(), g2.directed.edges(), msg="Graphs have different directed edges"
        )
        self.assertEqual(
            g1.undirected.edges(),
            g2.undirected.edges(),
            msg="Graphs have different undirected edges",
        )

    def test_repair_network_for_multi_mediators_confounder(self):
        """Test repair_network for multi_mediators_confounder."""
        actual_fixed_graph = repair_network(
            graph=multiple_mediators_confounders_example.graph,
            data=multiple_mediators_confounders_example.generate_data(100, seed=1),
            significance_level=0.05,
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
            undirected={"Y": ["Z2"]},
        )
        self.assert_graph_equal(actual_fixed_graph, expected_fixed_graph)

    def test_repair_network_for_multi_mediators_confounder_nuisance_var(self):
        """Test repair_network for multi_mediators_confounder_nuisance_var."""
        actual = repair_network(
            graph=multi_mediators_confounders_nuisance_vars_example.graph,
            data=multi_mediators_confounders_nuisance_vars_example.generate_data(100, seed=2),
            significance_level=0.01,
        )
        expected = NxMixedGraph.from_str_adj(
            directed={
                "Z1": ["X", "Z2"],
                "X": ["M1"],
                "M1": ["M2", "R1"],
                "M2": ["Y"],
                "Z2": ["Z3"],
                "Z3": ["Y"],
                "R1": ["R2"],
                "R2": ["R3"],
                "Y": ["R3"],
            },
            undirected={"Y": ["Z2"]},
        )
        self.assertNotIn(
            R1, set(actual.nodes()), msg="Nuisance variable R1 should have been removed"
        )
        self.assertNotIn(
            R2, set(actual.nodes()), msg="Nuisance variable R2 should have been removed"
        )
        self.assertNotIn(
            R3, set(actual.nodes()), msg="Nuisance variable R3 should have been removed"
        )
        self.assert_graph_equal(actual, expected)
