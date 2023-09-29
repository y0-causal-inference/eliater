"""This module tests the methods involved in discovering and marking latent nodes."""

import unittest
from copy import deepcopy

from y0.dsl import Variable
from y0.graph import set_latent

from eliater.discover_latent_nodes import find_all_nodes_in_causal_paths, mark_latent
from eliater.frontdoor_backdoor import (
    multiple_mediators_single_confounder_example,
    multi_mediators_confounders_nuisance_vars_example,
)


class TestDiscoverAndMarkLatentNodes(unittest.TestCase):
    """This class implements tests to verify the correctness of methods to discover and mark latent nodes."""

    def test_find_nodes_on_all_paths_for_multi_med(self):
        """Tests finding nodes in all causal paths for multi_med."""
        expected_nodes = {Variable("X"), Variable("M2"), Variable("M1"), Variable("Y")}
        actual_nodes = find_all_nodes_in_causal_paths(
            multiple_mediators_single_confounder_example.graph, Variable("X"), Variable("Y")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_mark_latent_for_multi_med_confounder_nuisance_var(self):
        """Tests marking nodes as latent.

        Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
        nodes for multi_med_confounder_nuisance_var.
        """
        expected_graph = deepcopy(multi_mediators_confounders_nuisance_vars_example.graph)
        set_latent(expected_graph.directed, {Variable("R1"), Variable("R2"), Variable("R3")})
        actual_graph = mark_latent(
            graph=multi_mediators_confounders_nuisance_vars_example.graph,
            treatments=Variable("X"),
            outcome=Variable("Y"),
        )
        self.assertEqual(expected_graph, actual_graph)
