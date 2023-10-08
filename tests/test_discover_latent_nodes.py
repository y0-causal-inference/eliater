"""This module tests the methods involved in discovering and marking latent nodes."""

import unittest
from copy import deepcopy

from eliater.discover_latent_nodes import find_all_nodes_in_causal_paths, mark_latent
from eliater.examples import ecoli, sars, t_cell_signaling_pathway
from eliater.frontdoor_backdoor import (
    multi_mediators_confounders_nuisance_vars_example,
    multiple_mediators_single_confounder_example,
)
from y0.dsl import Variable
from y0.graph import set_latent


class TestDiscoverAndMarkLatentNodes(unittest.TestCase):
    """This class implements tests to verify the correctness of methods to discover and mark latent nodes."""

    def test_find_nodes_on_all_paths_for_multi_med(self):
        """Test finding nodes in all causal paths for multi_med."""
        expected_nodes = {Variable("X"), Variable("M2"), Variable("M1"), Variable("Y")}
        actual_nodes = find_all_nodes_in_causal_paths(
            multiple_mediators_single_confounder_example.graph, Variable("X"), Variable("Y")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_mark_latent_for_multi_med_confounder_nuisance_var(self):
        """Test marking nodes as latent.

        Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
        nodes for multi_med_confounder_nuisance_var.
        """
        expected_graph = deepcopy(multi_mediators_confounders_nuisance_vars_example.graph)
        set_latent(expected_graph.directed, {Variable("R1"), Variable("R2"), Variable("R3")})
        actual_graph = mark_latent(
            graph=multi_mediators_confounders_nuisance_vars_example.graph,
            treatments=Variable("X"),
            outcomes=Variable("Y"),
        )
        self.assertEqual(expected_graph, actual_graph)

    def test_find_nodes_on_all_paths_for_t_cell_signaling_pathway(self):
        """Test finding nodes in all causal paths for t_cell_signaling_pathway."""
        expected_nodes = {Variable("Raf"), Variable("Mek"), Variable("Erk")}
        actual_nodes = find_all_nodes_in_causal_paths(
            t_cell_signaling_pathway.graph, {Variable("Raf"), Variable("Mek")}, Variable("Erk")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_mark_latent_for_t_cell_signaling_pathway(self):
        """Test marking nodes as latent.

        Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
        nodes for t_cell_signaling_pathway.
        """
        expected_graph = deepcopy(t_cell_signaling_pathway.graph)
        set_latent(expected_graph.directed, {Variable("Akt")})
        actual_graph = mark_latent(
            graph=t_cell_signaling_pathway.graph,
            treatments={Variable("Raf"), Variable("Mek")},
            outcomes=Variable("Erk"),
        )
        self.assertEqual(expected_graph, actual_graph)

    def test_find_nodes_on_all_paths_for_sars(self):
        """Test finding nodes in all causal paths for sars."""
        expected_nodes = {Variable("EGFR"), Variable("NFKB"), Variable("IL6AMP"), Variable("cytok")}
        actual_nodes = find_all_nodes_in_causal_paths(
            sars.graph, Variable("EGFR"), Variable("cytok")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_mark_latent_for_sars(self):
        """Test marking nodes as latent.

        Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
        nodes for sars.
        """
        expected_graph = deepcopy(sars.graph)
        actual_graph = mark_latent(
            graph=sars.graph, treatments=Variable("EGFR"), outcomes=Variable("cytok")
        )
        self.assertEqual(expected_graph, actual_graph)

    def test_find_nodes_on_all_paths_for_ecoli(self):
        """Test finding nodes in all causal paths for ecoli."""
        expected_nodes = {
            Variable("dcuR"),
            Variable("dpiA"),
            Variable("fnr"),
            Variable("fur"),
            Variable("narL"),
        }
        actual_nodes = find_all_nodes_in_causal_paths(
            ecoli.graph, Variable("fur"), Variable("dpiA")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_mark_latent_for_ecoli(self):
        """Test marking nodes as latent.

        Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
        nodes for ecoli.
        """
        expected_graph = deepcopy(ecoli.graph)
        descendants_not_ancestors = {
            "citX",
            "dpiB",
            "srIR",
            "citC",
            "citD",
            "appB",
            "appX",
            "hyaF",
            "hyaA",
            "cydD",
            "aceE",
            "hyaB",
            "mdh",
            "exuT",
            "hcp",
            "aspC",
            "gutM",
            "appY",
            "ydeO",
            "appA",
            "amtB",
            "cyoA",
            "gadX",
            "cirA",
            "hns",
            "yjjQ",
        }
        descendants_not_ancestors = {Variable(node) for node in descendants_not_ancestors}
        set_latent(expected_graph.directed, descendants_not_ancestors)
        actual_graph = mark_latent(
            graph=ecoli.graph, treatments=Variable("fur"), outcomes=Variable("dpiA")
        )
        self.assertEqual(expected_graph, actual_graph)

    def test_find_nodes_on_all_causal_paths_for_t_cell_signaling_pathway_with_multiple_outcomes(
        self,
    ):
        """Test finding nodes in all causal paths for t_cell_signaling_pathway with multiple outcomes."""
        expected_nodes = {
            Variable("PIP2"),
            Variable("PIP3"),
            Variable("PKC"),
            Variable("PKA"),
            Variable("Raf"),
            Variable("Mek"),
            Variable("Erk"),
            Variable("Akt"),
        }
        actual_nodes = find_all_nodes_in_causal_paths(
            graph=t_cell_signaling_pathway.graph,
            treatments={Variable("PIP2"), Variable("PIP3")},
            outcomes={Variable("Erk"), Variable("Akt")},
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_mark_latent_for_t_cell_signaling_pathway_with_multiple_outcomes(self):
        """Test marking nodes as latent.

        Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
        nodes for t_cell_signaling_pathway with multiple outcomes.
        """
        expected_graph = deepcopy(t_cell_signaling_pathway.graph)
        set_latent(expected_graph.directed, {Variable("Jnk"), Variable("P38")})
        actual_graph = mark_latent(
            graph=t_cell_signaling_pathway.graph,
            treatments={Variable("PIP2"), Variable("PIP3")},
            outcomes={Variable("Erk"), Variable("Akt")},
        )
        self.assertEqual(expected_graph, actual_graph)
