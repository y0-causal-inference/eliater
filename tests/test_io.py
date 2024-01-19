"""Tests for IO functions."""

import unittest

from optimaladj.CausalGraph import CausalGraph

from eliater.io import to_causal_graph
from y0.graph import NxMixedGraph


class TestToCausalGraph(unittest.TestCase):
    """Tests converting a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""

    def assert_causal_equal(self, expected: CausalGraph, actual: CausalGraph) -> None:
        """Compare two instances of :class:`optimaladj.CausalGraph.CausalGraph`."""
        self.assertEqual(expected.edges, actual.edges)

    def test_graph_with_latents(self):
        """Tests converting a mixed graph with latents to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"}, undirected={"X": "Y"})
        expected = CausalGraph()
        expected.add_edges_from([("X", "Y"), ("U_1", "X"), ("U_1", "Y")])
        actual = to_causal_graph(graph)
        self.assert_causal_equal(expected, actual)

    def test_graph_without_latents(self):
        """Test converting a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
        graph = NxMixedGraph.from_str_adj(directed={"X": "Y"})
        expected = CausalGraph()
        expected.add_edges_from([("X", "Y")])
        actual = to_causal_graph(graph)
        self.assert_causal_equal(expected, actual)
