# -*- coding: utf-8 -*-

"""Tests for simplifying latent variable DAGs."""

import unittest

import networkx as nx

from eliater.simplify_latent import (
    DEFAULT_SUFFIX,
    iter_latents,
    remove_redundant_latents,
    remove_widow_latents,
    simplify_latent_dag,
    transform_latents_with_parents,
)
from y0.algorithm.taheri_design import taheri_design_dag
from y0.dsl import Y1, Y2, Y3, U, Variable, W
from y0.examples import igf_example
from y0.graph import set_latent

X1, X2, X3 = map(Variable, ["X1", "X2", "X3"])


class TestDesign(unittest.TestCase):
    """Test the design algorithm."""

    def test_design(self):
        """Test the design algorithm."""
        results = taheri_design_dag(igf_example.graph.directed, cause="PI3K", effect="Erk", stop=3)
        self.assertIsNotNone(results)
        # FIXME do better than this.


class TestSimplify(unittest.TestCase):
    """Tests for the Robin Evans simplification algorithms."""

    def assert_latent_variable_dag_equal(self, expected, actual) -> None:
        """Check two latent variable DAGs are the same."""
        self.assertEqual(
            sorted(expected),
            sorted(actual),
            msg="Nodes are incorrect",
        )
        self.assertEqual(
            dict(expected.nodes.items()),
            dict(actual.nodes.items()),
            msg="Tags are incorrect",
        )
        self.assertEqual(
            sorted(expected.edges()),
            sorted(actual.edges()),
        )

    def test_remove_widows(self):
        """Test simplification 1 - removing widows."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (X1, X2),
                (X2, W),
                (U, X2),
                (U, X3),
                (U, W),
            ]
        )
        latents = {U, W}
        set_latent(graph, latents)
        self.assertEqual(latents, set(iter_latents(graph)))

        # Apply the simplification
        _, removed = remove_widow_latents(graph)
        self.assertEqual({W}, removed)

        expected = nx.DiGraph()
        expected.add_edges_from(
            [
                (X1, X2),
                (U, X2),
                (U, X3),
            ]
        )
        set_latent(expected, U)

        self.assert_latent_variable_dag_equal(expected, graph)

    def test_transform_latents_with_parents(self):
        """Test simplification 2: latents with parents can be transformed."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (X1, U),
                (X2, U),
                (U, Y1),
                (U, Y2),
                (U, Y3),
            ]
        )
        set_latent(graph, U)
        # Apply the simplification
        transform_latents_with_parents(graph)

        expected = nx.DiGraph()
        expected.add_edges_from(
            [
                (X1, Y1),
                (X1, Y2),
                (X1, Y3),
                (X2, Y1),
                (X2, Y2),
                (X2, Y3),
                (Variable(f"U{DEFAULT_SUFFIX}"), Y1),
                (Variable(f"U{DEFAULT_SUFFIX}"), Y2),
                (Variable(f"U{DEFAULT_SUFFIX}"), Y3),
            ]
        )
        set_latent(expected, Variable(f"U{DEFAULT_SUFFIX}"))

        self.assert_latent_variable_dag_equal(expected, graph)

    def test_remove_redundant_latents(self):
        """Test simplification 3 - remove redundant latents."""
        graph = nx.DiGraph()
        graph.add_edges_from(
            [
                (U, X1),
                (U, X2),
                (U, X3),
                (W, X1),
                (W, X2),
            ]
        )
        set_latent(graph, [U, W])
        # Apply the simplification
        remove_redundant_latents(graph)

        expected = nx.DiGraph()
        expected.add_edges_from(
            [
                (U, X1),
                (U, X2),
                (U, X3),
            ]
        )
        set_latent(expected, [U])

        self.assert_latent_variable_dag_equal(expected, graph)

    def test_remove_unidirectional_latents(self):
        """Test simplification 4 - remove unidirectional latents."""
        from y0.graph import NxMixedGraph

        actual_graph = NxMixedGraph.from_str_adj(
            directed={"U1": ["U2"], "U2": ["U3"], "U3": ["U4"]}
        )
        set_latent(actual_graph.directed, [Variable("U2"), Variable("U3")])
        simplify_latent_dag(actual_graph.directed)
        expected_graph = NxMixedGraph.from_str_adj(directed={"U1": ["U4"]})
        self.assertEqual(actual_graph.directed.edges, expected_graph.directed.edges)
        self.assertEqual(set(actual_graph.nodes()), set(expected_graph.nodes()))

    def test_simplify_latent_dag_for_graph0(self):
        """Test simplification for a simple network."""
        from y0.graph import NxMixedGraph

        # Original graph
        actual_graph = NxMixedGraph.from_str_adj(
            directed={
                "U1": ["V1", "V2", "V3"],
                "U2": ["V2", "V3"],
                "U3": ["V4", "V5"],
                "U4": ["V5"],
                "V1": ["U3", "U5"],
                "V2": ["U3"],
                "V3": ["U3"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph.directed, [Variable("U" + str(num)) for num in range(1, 6)])
        # Simplify the network
        simplify_latent_dag(actual_graph.directed)
        # Expected graph after simplification
        expected_graph = NxMixedGraph.from_str_adj(
            directed={
                "U1": ["V1", "V2", "V3"],
                "U3": ["V4", "V5"],
                "V1": ["V4", "V5"],
                "V2": ["V4", "V5"],
                "V3": ["V4", "V5"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph.directed, [Variable("U" + str(num)) for num in range(1, 6)])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_graph1(self):
        """Test simplification for a simple network."""
        from y0.graph import NxMixedGraph

        # Original graph
        actual_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph.directed, [Variable("EGF"), Variable("IGF")])
        # Simplify the network
        simplify_latent_dag(actual_graph.directed)
        # Expected graph after simplification
        expected_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph.directed, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_graph2(self):
        """Test simplification for a simple network."""
        from y0.graph import NxMixedGraph

        # Original graph
        actual_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph.directed, [Variable("EGF"), Variable("IGF"), Variable("PI3K")])
        # Simplify the network
        simplify_latent_dag(actual_graph.directed)
        # Expected graph after simplification
        expected_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "Akt"],
                "SOS": ["Ras"],
                "Ras": ["Akt", "Raf"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph.directed, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_graph3(self):
        """Test simplification for a simple network."""
        from y0.graph import NxMixedGraph

        # Original graph
        actual_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(actual_graph.directed, [Variable("EGF"), Variable("IGF"), Variable("Ras")])
        # Simplify the network
        simplify_latent_dag(actual_graph.directed)
        # Expected graph after simplification
        expected_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["PI3K", "Raf"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph.directed, [Variable("EGF"), Variable("Ras")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_graph4(self):
        """Test simplification for a simple network."""
        from y0.graph import NxMixedGraph

        # Original graph
        actual_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(
            actual_graph.directed,
            [Variable("EGF"), Variable("IGF"), Variable("Akt"), Variable("Erk")],
        )
        # Simplify the network
        simplify_latent_dag(actual_graph.directed)
        # Expected graph after simplification
        expected_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Raf"],
                "Raf": ["Mek"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph.directed, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)

    def test_simplify_latent_dag_for_graph5(self):
        """Test simplification for a simple network."""
        from y0.graph import NxMixedGraph

        # Original graph
        actual_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "IGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Raf"],
                "PI3K": ["Akt"],
                "Akt": ["Raf"],
                "Raf": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Mark the latent nodes
        set_latent(
            actual_graph.directed,
            [Variable("EGF"), Variable("IGF"), Variable("Raf"), Variable("Akt")],
        )
        # Simplify the network
        simplify_latent_dag(actual_graph.directed)
        # Expected graph after simplification
        expected_graph = NxMixedGraph.from_str_adj(
            directed={
                "EGF": ["SOS", "PI3K"],
                "SOS": ["Ras"],
                "Ras": ["PI3K", "Mek"],
                "PI3K": ["Mek"],
                "Mek": ["Erk"],
            }
        )
        # Expected latent nodes after simplification
        set_latent(expected_graph.directed, [Variable("EGF")])
        self.assertEqual(actual_graph, expected_graph)
