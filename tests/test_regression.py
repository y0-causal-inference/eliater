"""Test the regression module."""

import unittest

import pandas as pd

from eliater.regression import get_regression_coefficients
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from eliater.frontdoor_backdoor import frontdoor_backdoor_example
from y0.dsl import X, Y


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
            graph=graph, data=data, treatments=treatments, outcome=outcome
        )
        self.assertEqual(
            {c.name for c in expected_coefficients},
            set(name_to_coefficient),
            msg="Coefficients were calculated for the wrong variables",
        )
        # TODO since there's some random aspect, set the appropriate delta
        self.assertAlmostEqual(expected_x_coefficient, name_to_coefficient[X.name])
