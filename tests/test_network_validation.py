"""This module tests the steps involved in repairing the network structure."""

import unittest

from eliater.frontdoor_backdoor import base_example, multiple_mediators_single_confounder_example
from eliater.network_validation import choose_default_test, get_state_space_map, validate_test
from y0.dsl import Variable
from y0.examples.frontdoor import generate_data_for_frontdoor
from y0.examples.frontdoor_backdoor import generate_data_for_frontdoor_backdoor
from y0.graph import NxMixedGraph

R1, R2, R3 = (Variable("R{i}") for i in (1, 2, 3))


class TestRepair(unittest.TestCase):
    """Tests for repairing the network structure.

    This class implements tests to verify the correctness of steps involved in repairing the network structure
    by conditional independence tests.
    """

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

    def test_validate_test_for_invalid_input_test(self):
        """Test validate_test for invalid input test."""
        self.assertRaises(
            ValueError,
            validate_test,
            multiple_mediators_single_confounder_example.generate_data(100),
            "abc",
        )

    def test_validate_test_for_continuous_data_and_not_pearson(self):
        """Test validate_test for continuous data when pearson is not chosen."""
        self.assertRaises(
            ValueError,
            validate_test,
            base_example.generate_data(),
            "chi-square",
        )

    def test_validate_test_for_discrete_data_and_pearson(self):
        """Test validate_test for discrete data when pearson is chosen."""
        self.assertRaises(
            ValueError,
            validate_test,
            generate_data_for_frontdoor_backdoor(1000),
            "pearson",
        )

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