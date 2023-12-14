"""Test the regression module."""

import unittest

import pandas as pd

from eliater.frontdoor_backdoor import (
    frontdoor_backdoor_example,
    multiple_mediators_confounders_example,
    multiple_mediators_single_confounder_example,
)
from eliater.regression import get_adjustment_sets, get_regression_coefficients
from y0.dsl import Z1, Z2, Z3, Variable, W, X, Y
from y0.graph import NxMixedGraph


class TestRegression(unittest.TestCase):
    """Test the regression module."""

    # TODO create other unit tests that rely on small, well-defined graphs where
    #  you can create a ground truth

    def test_frondoor_backdoor_regression(self):
        """Test getting regression coefficients for the frontdoor-backdoor graph."""
        graph: NxMixedGraph = frontdoor_backdoor_example.graph
        data: pd.DataFrame = frontdoor_backdoor_example.generate_data(1000)
        treatments: set[Variable] = {X}
        outcome: Variable = Y
        expected_x_coefficient: float = ...  # TODO
        expected_coefficients: set[Variable] = ...  # TODO

        name_to_coefficient = get_regression_coefficients(
            graph=graph, data=data, treatments=treatments, outcomes=outcome
        )
        self.assertEqual(
            {c.name for c in expected_coefficients},
            set(name_to_coefficient),
            msg="Coefficients were calculated for the wrong variables",
        )
        # TODO since there's some random aspect, set the appropriate delta
        self.assertAlmostEqual(expected_x_coefficient, name_to_coefficient[X.name])


class TestPGMpyAdjustmentSet(unittest.TestCase):
    """Tests for deriving adjustment sets with :mod:`pgmpy`."""

    def test_example1(self):
        """Test getting adjustment set for the frontdoor-backdoor graph."""
        graph = frontdoor_backdoor_example.graph
        expected = {frozenset([W])}
        actual = get_adjustment_sets(graph, X, Y, impl="pgmpy")
        self.assertEqual(expected, actual)

    def test_example2(self):
        """Test getting adjustment set for the multiple-mediators-single-confounder graph."""
        graph = multiple_mediators_single_confounder_example.graph
        self.assertRaises(ValueError, get_adjustment_sets, graph, X, Y, impl="pgmpy")

    def test_example3(self):
        """Test multiple possible adjustment sets for multiple-mediators-multiple-confounders graph."""
        graph = multiple_mediators_confounders_example.graph
        expected = {
            frozenset([Z1]),
            frozenset([Z2]),
            frozenset([Z3]),
        }
        actual = get_adjustment_sets(graph, X, Y, impl="pgmpy")
        self.assertEqual(expected, actual)
