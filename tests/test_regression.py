"""Test the regression module."""

import unittest

import pandas as pd
from optimaladj.CausalGraph import CausalGraph
from pgmpy.models import BayesianNetwork

from eliater.examples import sars_cov_2_example, t_cell_signaling_example
from eliater.frontdoor_backdoor import (
    frontdoor_backdoor_example,
    multiple_mediators_confounders_example,
    multiple_mediators_single_confounder_example,
)
from eliater.regression import (
    get_adjustment_sets,
    get_regression_coefficients,
    to_bayesian_network,
    to_causal_graph,
)
from y0.dsl import Z3, Variable, X, Y
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
        from y0.dsl import Z

        graph = frontdoor_backdoor_example.graph
        expected = {frozenset([Z])}
        actual = get_adjustment_sets(graph, X, Y)
        self.assertEqual(expected, actual)

    def test_example2(self):
        """Test getting adjustment set for the multiple-mediators-single-confounder graph."""
        graph = multiple_mediators_single_confounder_example.graph
        self.assertRaises(ValueError, get_adjustment_sets, graph, X, Y)

    def test_example3(self):
        """Test multiple possible adjustment sets for multiple-mediators-multiple-confounders graph."""
        graph = multiple_mediators_confounders_example.graph
        expected = {frozenset([Z3])}
        actual = get_adjustment_sets(graph, X, Y)
        self.assertEqual(expected, actual)

    def test_t_cell_signaling_example(self):
        """Test getting adjustment set for the t_cell_signaling graph."""
        graph = t_cell_signaling_example.graph
        expected = {frozenset([Variable(v) for v in ("PKA", "PKC")])}
        actual = get_adjustment_sets(graph, Variable("Raf"), Variable("Erk"))
        self.assertEqual(expected, actual)

    def test_sars_cov_2_example(self):
        """Test getting adjustment set for the sars_cov_2 graph."""
        graph = sars_cov_2_example.graph
        expected = {frozenset([Variable(v) for v in ("IL6STAT3", "SARS_COV2", "TNF")])}
        actual = get_adjustment_sets(graph, Variable("EGFR"), Variable("cytok"))
        self.assertEqual(expected, actual)


class TestToBayesianNetwork(unittest.TestCase):
    """Tests converting a mixed graph to an equivalent :class:`pgmpy.BayesianNetwork`."""

    @staticmethod
    def _compare_bayesian_networks(
        bayesian_network_1: BayesianNetwork, bayesian_network_2: BayesianNetwork
    ) -> bool:
        """Compare two instances of :class:`pgmpy.BayesianNetwork`."""
        return (
            set(bayesian_network_1.edges) == set(bayesian_network_2.edges)
            and bayesian_network_1.latents == bayesian_network_2.latents
        )

    def test_graph_with_latents(self):
        """Tests converting a mixed graph with latents to an equivalent :class:`pgmpy.BayesianNetwork`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"}, undirected={"X": "Y"})
        expected = BayesianNetwork(
            ebunch=[("X", "Y"), ("U_X_Y", "X"), ("U_X_Y", "Y")], latents=["U_X_Y"]
        )
        actual = to_bayesian_network(graph)
        return self._compare_bayesian_networks(expected, actual)

    def test_graph_without_latents(self):
        """Tests converting a mixed graph without latents to an equivalent :class:`pgmpy.BayesianNetwork`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"})
        expected = BayesianNetwork(ebunch=[("X", "Y")])
        actual = to_bayesian_network(graph)
        return self._compare_bayesian_networks(expected, actual)


class TestToCausalGraph:
    """Tests converting a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""

    @staticmethod
    def _compare_causal_graphs(causal_graph_1: CausalGraph, causal_graph_2: CausalGraph) -> bool:
        """Compare two instances of :class:`optimaladj.CausalGraph.CausalGraph`."""
        return causal_graph_1.edges == causal_graph_2.edges

    def test_graph_with_latents(self):
        """Tests converting a mixed graph with latents to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"}, undirected={"X": "Y"})
        expected = CausalGraph()
        expected.add_edges_from([("X", "Y"), ("U1", "X"), ("U1", "Y")])
        actual = to_causal_graph(graph)
        return self._compare_causal_graphs(expected, actual)

    def test_graph_without_latents(self):
        """Test converting a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"})
        expected = CausalGraph()
        expected.add_edges_from([("X", "Y")])
        actual = to_causal_graph(graph)
        return self._compare_causal_graphs(expected, actual)
