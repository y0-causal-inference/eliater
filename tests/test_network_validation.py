"""This module tests the steps involved in repairing the network structure."""

import unittest

from y0.dsl import Variable
from y0.graph import NxMixedGraph

R1, R2, R3 = (Variable("R{i}") for i in (1, 2, 3))


class TestRepair(unittest.TestCase):
    """Tests for repairing the network structure.

    This class implements tests to verify the correctness of steps involved in repairing the network structure
    by conditional independence tests.
    """

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
