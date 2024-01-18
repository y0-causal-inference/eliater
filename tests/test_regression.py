"""Test the regression module."""

import unittest
from typing import Dict, List, Tuple

import pandas as pd

from eliater.examples import sars_cov_2_example, t_cell_signaling_example
from eliater.frontdoor_backdoor import (
    example_2,
    frontdoor_backdoor_example,
    multiple_mediators_confounders_nuisance_vars_example,
    multiple_mediators_single_confounder_example,
)
from eliater.regression import (
    RegressionResult,
    SummaryStatistics,
    estimate_query,
    fit_regression,
    get_adjustment_set,
    summary_statistics,
)
from y0.dsl import Z1, Z2, Z3, Variable, X, Y, Z
from y0.graph import NxMixedGraph


class TestRegression(unittest.TestCase):
    """Test the regression method."""

    def _compare_regression_result(self, result1: RegressionResult, result2: RegressionResult):
        """Compare two instances of :class:`eliater.regression.RegressionResult`."""
        expected_coefficients, expected_intercept = result1
        actual_coefficients, actual_intercept = result2
        self.assertEqual(set(expected_coefficients.keys()), set(actual_coefficients.keys()))
        for variable in expected_coefficients:
            self.assertAlmostEqual(
                actual_coefficients[variable], expected_coefficients[variable], delta=0.01
            )
        self.assertAlmostEqual(expected_intercept, actual_intercept, delta=0.01)

    def test_frondoor_backdoor_regression(self):
        """Test regression result for the frontdoor-backdoor graph."""
        graph: NxMixedGraph = frontdoor_backdoor_example.graph
        data: pd.DataFrame = frontdoor_backdoor_example.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_coefficients: Dict[Variable, float] = {X: 0.163, Z: 0.469}
        expected_intercept: float = -1.239
        expected_result = RegressionResult(expected_coefficients, expected_intercept)
        actual_result = fit_regression(
            graph=graph, data=data, treatments=treatments, outcome=outcome
        )
        self._compare_regression_result(expected_result, actual_result)

    def test_multiple_mediators_multiple_confounders_regression(self):
        """Test regression result for the multiple-mediators-multiple-confounders graph."""
        graph: NxMixedGraph = example_2.graph
        data: pd.DataFrame = example_2.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_coefficients: Dict[Variable, float] = {X: 0.165, Z3: 1.311}
        expected_intercept: float = 21.06
        expected_result = RegressionResult(expected_coefficients, expected_intercept)
        actual_result = fit_regression(
            graph=graph, data=data, treatments=treatments, outcome=outcome
        )
        self._compare_regression_result(expected_result, actual_result)

    def test_multiple_mediators_confounders_nuisance_vars_regression(self):
        """Test regression result for the multiple-mediators-confounders-nuisance-vars graph."""
        graph: NxMixedGraph = multiple_mediators_confounders_nuisance_vars_example.graph
        data: pd.DataFrame = multiple_mediators_confounders_nuisance_vars_example.generate_data(
            1000, seed=100
        )
        treatments = {X}
        outcome = Y
        expected_coefficients: Dict[Variable, float] = {X: 0.274, Z3: 0.578}
        expected_intercept: float = 5.578
        expected_result = RegressionResult(expected_coefficients, expected_intercept)
        actual_result = fit_regression(
            graph=graph, data=data, treatments=treatments, outcome=outcome
        )
        self._compare_regression_result(expected_result, actual_result)


class TestEstimateQuery(unittest.TestCase):
    """Test the estimate_query method."""

    def test_frondoor_backdoor_ate(self):
        """Test getting average treatment effect for the frontdoor-backdoor graph."""
        graph = frontdoor_backdoor_example.graph
        data = frontdoor_backdoor_example.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_ate: float = 0.163
        actual_ate = estimate_query(graph=graph, data=data, treatments=treatments, outcome=outcome)
        self.assertAlmostEqual(expected_ate, actual_ate, delta=0.01)

    def test_frondoor_backdoor_expected_value(self):
        """Test getting expected value for the frontdoor-backdoor graph."""
        graph = frontdoor_backdoor_example.graph
        data = frontdoor_backdoor_example.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_value: float = 3.476
        interventions = {X: 0}
        actual_value = estimate_query(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            query_type="expected_value",
            interventions=interventions,
        )
        self.assertAlmostEqual(expected_value, actual_value, delta=0.01)

    def test_multiple_mediators_multiple_confounders_ate(self):
        """Test getting average treatment effect for the multiple-mediators-multiple-confounders graph."""
        graph = example_2.graph
        data = example_2.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_ate: float = 0.165
        actual_ate = estimate_query(graph=graph, data=data, treatments=treatments, outcome=outcome)
        self.assertAlmostEqual(expected_ate, actual_ate, delta=0.01)

    def test_multiple_mediators_multiple_confounders_expected_value(self):
        """Test getting expected value for the multiple-mediators-multiple-confounders graph."""
        graph = example_2.graph
        data = example_2.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_value: float = 68.448
        interventions = {X: 0}
        actual_value = estimate_query(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            query_type="expected_value",
            interventions=interventions,
        )
        self.assertAlmostEqual(expected_value, actual_value, delta=0.01)

    def test_multiple_mediators_confounders_nuisance_vars_ate(self):
        """Test getting average treatment effect for the multiple-mediators-confounders-nuisance-vars graph."""
        graph = multiple_mediators_confounders_nuisance_vars_example.graph
        data = multiple_mediators_confounders_nuisance_vars_example.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_ate: float = 0.274
        actual_ate = estimate_query(graph=graph, data=data, treatments=treatments, outcome=outcome)
        self.assertAlmostEqual(expected_ate, actual_ate, delta=0.01)

    def test_multiple_mediators_confounders_nuisance_vars_expected_value(self):
        """Test getting expected value for the multiple-mediators-confounders-nuisance-vars graph."""
        graph = multiple_mediators_confounders_nuisance_vars_example.graph
        data = multiple_mediators_confounders_nuisance_vars_example.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        expected_value: float = 14.209
        interventions = {X: 0}
        actual_value = estimate_query(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            query_type="expected_value",
            interventions=interventions,
        )
        self.assertAlmostEqual(expected_value, actual_value, delta=0.01)

    def test_frontdoor_backdoor_probabilities(self):
        """Test getting probabilities for the frontdoor-backdoor graph."""
        graph = frontdoor_backdoor_example.graph
        data = frontdoor_backdoor_example.generate_data(10, seed=100)
        treatments = {X}
        outcome = Y
        expected_probabilities: List[float] = [
            -0.74,
            -2.18,
            -2.66,
            -2.43,
            -0.93,
            -2.95,
            -2.59,
            -2.59,
            -2.63,
            -2.99,
        ]
        interventions = {X: 0}
        actual_probabilities = estimate_query(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            query_type="probability",
            interventions=interventions,
        )
        self.assertEqual(len(expected_probabilities), len(actual_probabilities))
        for index in range(0, len(expected_probabilities)):
            self.assertAlmostEqual(
                expected_probabilities[index], actual_probabilities[index], delta=0.01
            )


class TestAdjustmentSet(unittest.TestCase):
    """Tests for deriving adjustment sets with :mod:`pgmpy` and :mod:`optimaladj`."""

    def _compare(
        self, actual: Tuple[frozenset[Variable], str], expected: Tuple[frozenset[Variable], str]
    ) -> None:
        """Compare the expected and actual adjustment sets."""
        expected_adjustment_set, expected_adjustment_set_type = expected
        actual_adjustment_set, actual_adjustment_set_type = actual
        self.assertEqual(expected_adjustment_set, actual_adjustment_set)
        self.assertEqual(expected_adjustment_set_type, actual_adjustment_set_type)

    def test_example1(self):
        """Test getting adjustment set for the frontdoor-backdoor graph."""
        graph = frontdoor_backdoor_example.graph
        expected = frozenset([Z]), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self._compare(actual, expected)

    def test_example2(self):
        """Test getting adjustment set for the multiple-mediators-single-confounder graph."""
        graph = multiple_mediators_single_confounder_example.graph
        with self.assertRaises(ValueError):
            get_adjustment_set(graph, X, Y)

    def test_example3(self):
        """Test getting adjustment set for multiple-mediators-multiple-confounders graph."""
        graph = example_2.graph
        expected = frozenset([Z3]), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self._compare(actual, expected)

    def test_example4(self):
        """Test getting adjustment set for a sample graph."""
        graph = NxMixedGraph.from_str_adj(
            directed={"V1": ["V2", "X"], "V3": ["V5"], "X": ["Y", "V5"], "V2": ["V5"], "V5": ["Y"]}
        )
        expected = frozenset([Variable(v) for v in ("V2", "V3")]), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self._compare(actual, expected)

    def test_example5(self):
        """Test getting adjustment set for a sample graph."""
        graph = NxMixedGraph.from_str_adj(
            directed={"A": ["B", "X"], "C": ["Y"], "X": ["Y"], "D": ["X"], "B": ["C"]}
        )
        expected = frozenset({Variable("C")}), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self._compare(actual, expected)

    def test_example6(self):
        """Test getting adjustment set for a sample graph."""
        graph = NxMixedGraph.from_str_adj(
            directed={
                "X": ["M1"],
                "M1": ["M2", "M3", "M4"],
                "M2": ["Y"],
                "M3": ["Y"],
                "M4": ["Y"],
                "Z1": ["X", "Z2"],
                "Z2": ["Z3"],
                "Z3": ["Y"],
            }
        )
        expected_adjustment_sets = {frozenset([Z3]), frozenset([Z1]), frozenset([Z2])}
        expected_adjustment_set_type = "Minimal Adjustment Set"
        actual_adjustment_set, actual_adjustment_set_type = get_adjustment_set(graph, X, Y)
        self.assertIn(actual_adjustment_set, expected_adjustment_sets)
        self.assertEqual(actual_adjustment_set_type, expected_adjustment_set_type)

    def test_example7(self):
        """Test getting adjustment set for a sample graph."""
        graph = NxMixedGraph.from_str_adj(
            directed={"A": ["X"], "B": ["X", "C"], "C": ["Y"], "X": ["D", "Y"]}
        )
        expected = frozenset([Variable("C")]), "Optimal Minimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self._compare(actual, expected)

    def test_t_cell_signaling_example(self):
        """Test getting adjustment set for the t_cell_signaling graph."""
        graph = t_cell_signaling_example.graph
        expected = (
            frozenset([Variable(v) for v in ("PKA", "PKC")]),
            "Optimal Minimal Adjustment Set",
        )
        actual = get_adjustment_set(graph, Variable("Raf"), Variable("Erk"))
        self._compare(actual, expected)

    def test_sars_cov_2_example(self):
        """Test getting adjustment set for the sars_cov_2 graph."""
        graph = sars_cov_2_example.graph
        expected = (
            frozenset([Variable(v) for v in ("IL6STAT3", "TNF", "SARS_COV2", "PRR")]),
            "Optimal Adjustment Set",
        )
        actual = get_adjustment_set(graph, Variable("EGFR"), Variable("cytok"))
        self._compare(actual, expected)


class TestSummaryStatistics(unittest.TestCase):
    """Test getting summary statistics of the estimated outcome probabilities."""

    def test_frondoor_backdoor_summary_stats(self):
        """Test getting summary statistics of the estimated outcome probabilities for the frontdoor-backdoor graph."""
        graph: NxMixedGraph = frontdoor_backdoor_example.graph
        data: pd.DataFrame = frontdoor_backdoor_example.generate_data(1000, seed=100)
        treatments = {X}
        outcome = Y
        interventions = {X: 0}
        expected_stats = SummaryStatistics(1000, 3.476, 0.473, 2.107, 3.145, 3.478, 3.802, 5.064)
        actual_stats = summary_statistics(graph, data, treatments, outcome, interventions)
        for value1, value2 in zip(expected_stats, actual_stats):
            self.assertAlmostEqual(value1, value2, delta=0.01)
