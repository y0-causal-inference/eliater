"""Test the regression module."""

import unittest
from typing import Tuple

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
    get_adjustment_set,
    get_regression_coefficients,
    to_bayesian_network,
    to_causal_graph,
)
from y0.dsl import Z1, Z2, Z3, Variable, X, Y
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


class TestAdjustmentSet(unittest.TestCase):
    """Tests for deriving adjustment sets with :mod:`pgmpy` and :mod:`optimaladj`."""

    @staticmethod
    def _compare(
        actual: Tuple[frozenset[Variable], str], expected: Tuple[frozenset[Variable], str]
    ) -> bool:
        """Compare the expected and actual adjustment sets."""
        expected_adjustment_set, expected_adjustment_set_type = expected
        actual_adjustment_set, actual_adjustment_set_type = actual
        return (
            expected_adjustment_set == actual_adjustment_set
            and expected_adjustment_set_type == actual_adjustment_set_type
        )

    def test_example1(self):
        """Test getting adjustment set for the frontdoor-backdoor graph."""
        from y0.dsl import Z

        graph = frontdoor_backdoor_example.graph
        expected = frozenset([Z]), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self.assertTrue(self._compare(actual, expected))

    def test_example2(self):
        """Test getting adjustment set for the multiple-mediators-single-confounder graph."""
        graph = multiple_mediators_single_confounder_example.graph
        self.assertRaises(ValueError, get_adjustment_set, graph, X, Y)

    def test_example3(self):
        """Test getting adjustment set for multiple-mediators-multiple-confounders graph."""
        graph = multiple_mediators_confounders_example.graph
        expected = frozenset([Z3]), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self.assertTrue(self._compare(actual, expected))

    def test_example4(self):
        """Test getting adjustment set for a sample graph."""
        graph = NxMixedGraph.from_str_adj(
            directed={"V1": ["V2", "X"], "V3": ["V5"], "X": ["Y", "V5"], "V2": ["V5"], "V5": ["Y"]}
        )
        expected = frozenset([Variable(v) for v in ("V2", "V3")]), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self.assertTrue(self._compare(actual, expected))

    def test_example5(self):
        """Test getting adjustment set for a sample graph."""
        graph = NxMixedGraph.from_str_adj(
            directed={"A": ["B", "X"], "C": ["Y"], "X": ["Y"], "D": ["X"], "B": ["C"]}
        )
        expected = frozenset({Variable("C")}), "Optimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self.assertTrue(self._compare(actual, expected))

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
        self.assertTrue(
            actual_adjustment_set in expected_adjustment_sets
            and actual_adjustment_set_type == expected_adjustment_set_type
        )

    def test_example7(self):
        """Test getting adjustment set for a sample graph."""
        graph = NxMixedGraph.from_str_adj(
            directed={"A": ["X"], "B": ["X", "C"], "C": ["Y"], "X": ["D", "Y"]}
        )
        expected = frozenset([Variable("C")]), "Optimal Minimal Adjustment Set"
        actual = get_adjustment_set(graph, X, Y)
        self.assertTrue(self._compare(actual, expected))

    def test_t_cell_signaling_example(self):
        """Test getting adjustment set for the t_cell_signaling graph."""
        graph = t_cell_signaling_example.graph
        expected = (
            frozenset([Variable(v) for v in ("PKA", "PKC")]),
            "Optimal Minimal Adjustment Set",
        )
        actual = get_adjustment_set(graph, Variable("Raf"), Variable("Erk"))
        self.assertTrue(self._compare(actual, expected))

    def test_sars_cov_2_example(self):
        """Test getting adjustment set for the sars_cov_2 graph."""
        graph = sars_cov_2_example.graph
        expected = (
            frozenset([Variable(v) for v in ("IL6STAT3", "TNF", "SARS_COV2", "PRR")]),
            "Optimal Adjustment Set",
        )
        actual = get_adjustment_set(graph, Variable("EGFR"), Variable("cytok"))
        self.assertTrue(self._compare(actual, expected))


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
        self.assertTrue(self._compare_bayesian_networks(expected, actual))

    def test_graph_without_latents(self):
        """Tests converting a mixed graph without latents to an equivalent :class:`pgmpy.BayesianNetwork`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"})
        expected = BayesianNetwork(ebunch=[("X", "Y")])
        actual = to_bayesian_network(graph)
        self.assertTrue(self._compare_bayesian_networks(expected, actual))


class TestToCausalGraph(unittest.TestCase):
    """Tests converting a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""

    @staticmethod
    def _compare_causal_graphs(causal_graph_1: CausalGraph, causal_graph_2: CausalGraph) -> bool:
        """Compare two instances of :class:`optimaladj.CausalGraph.CausalGraph`."""
        return causal_graph_1.edges == causal_graph_2.edges

    def test_graph_with_latents(self):
        """Tests converting a mixed graph with latents to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"}, undirected={"X": "Y"})
        expected = CausalGraph()
        expected.add_edges_from([("X", "Y"), ("U_1", "X"), ("U_1", "Y")])
        actual = to_causal_graph(graph)
        self.assertTrue(self._compare_causal_graphs(expected, actual))

    def test_graph_without_latents(self):
        """Test converting a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"})
        expected = CausalGraph()
        expected.add_edges_from([("X", "Y")])
        actual = to_causal_graph(graph)
        self.assertTrue(self._compare_causal_graphs(expected, actual))


# class TestFitRegression(unittest.TestCase):
#
#     def test_ecoli(self):
#         graph = ecoli_transcription_example.graph
#         import pandas as pd
#         data = pd.read_csv("C:\\Users\\pnava\\PycharmProjects\\eliater\\src\\data\\EColi_obs_data.csv")
#         co_eff = fit_regression(graph=graph, data=data, treatments=Variable("fur"), outcome=Variable("dpiA"))
#         print(co_eff)
#         self.assertEqual(True, False)
