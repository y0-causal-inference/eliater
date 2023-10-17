"""This module tests the methods involved in discovering and marking latent nodes."""

import unittest

from eliater.discover_latent_nodes import find_all_nodes_in_causal_paths
from eliater.examples import sars, simple, t_cell_signaling_pathway
from eliater.frontdoor_backdoor import multiple_mediators_single_confounder_example
from y0.dsl import Variable

# from copy import deepcopy


# from y0.graph import set_latent


class TestDiscoverAndMarkLatentNodes(unittest.TestCase):
    """This class implements tests to verify the correctness of methods to discover and mark latent nodes."""

    def test_find_nodes_on_all_causal_paths_for_simple_graph(self):
        """Test finding nodes in all causal paths for a simple graph.

        A causal path is defined as any simple directed path from any of the treatments to any of the outcomes.
        In the example provided in this test case, there are three treatments,
        namely X1, X2, and X3, and one outcome, Y.

        The following causal paths exist from treatments to outcomes:
        1. X1 -> X3 -> A -> B -> Y
        2. X1 -> A -> B -> Y
        3. X2 -> A -> C -> D -> Y

        Therefore, the nodes present in all causal paths are: X1, X2, X3, A, B, C, D and Y.
        """
        expected_nodes = {Variable(x) for x in ["X1", "X2", "X3", "A", "B", "C", "D", "Y"]}
        actual_nodes = find_all_nodes_in_causal_paths(
            graph=simple.graph,
            treatments={Variable("X1"), Variable("X2"), Variable("X3")},
            outcomes={Variable("Y")},
        )
        self.assertEqual(expected_nodes, actual_nodes)

    # def test_mark_latent_for_simple_graph(self):
    #     """Test marking nodes as latent for a simple graph.
    #
    #     In this test case, we mark the descendants of nodes on causal paths
    #     that are not ancestors of the outcome variable as latent.
    #
    #     Considering the example provided, the nodes on all causal paths are: X1, X2, X3, A, B, C, D, and Y.
    #
    #     The descendants of these nodes that are not ancestors of the outcome variable are: E, F, G, and H.
    #     Consequently, we mark these nodes as latent.
    #     """
    #     expected_graph = deepcopy(simple.graph)
    #     set_latent(expected_graph.directed, {Variable(v) for v in ["E", "F", "G", "H"]})
    #     actual_graph = mark_latent(
    #         graph=simple.graph,
    #         treatments={Variable("X1"), Variable("X2"), Variable("X3")},
    #         outcomes={Variable("Y")},
    #     )
    #     self.assertEqual(expected_graph, actual_graph)

    def test_find_nodes_on_all_paths_for_multi_med(self):
        """Test finding nodes in all causal paths for multi_med."""
        expected_nodes = {Variable("X"), Variable("M2"), Variable("M1"), Variable("Y")}
        actual_nodes = find_all_nodes_in_causal_paths(
            multiple_mediators_single_confounder_example.graph, Variable("X"), Variable("Y")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    # def test_mark_latent_for_multi_med_confounder_nuisance_var(self):
    #     """Test marking nodes as latent.
    #
    #     Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
    #     nodes for multi_med_confounder_nuisance_var.
    #     """
    #     expected_graph = deepcopy(multi_mediators_confounders_nuisance_vars_example.graph)
    #     set_latent(expected_graph.directed, {Variable("R1"), Variable("R2"), Variable("R3")})
    #     actual_graph = mark_latent(
    #         graph=multi_mediators_confounders_nuisance_vars_example.graph,
    #         treatments=Variable("X"),
    #         outcomes=Variable("Y"),
    #     )
    #     self.assertEqual(expected_graph, actual_graph)

    def test_find_nodes_on_all_paths_for_t_cell_signaling_pathway(self):
        """Test finding nodes in all causal paths for t_cell_signaling_pathway."""
        expected_nodes = {Variable("Raf"), Variable("Mek"), Variable("Erk")}
        actual_nodes = find_all_nodes_in_causal_paths(
            t_cell_signaling_pathway.graph, {Variable("Raf"), Variable("Mek")}, Variable("Erk")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    # def test_mark_latent_for_t_cell_signaling_pathway(self):
    #     """Test marking nodes as latent.
    #
    #     Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
    #     nodes for t_cell_signaling_pathway.
    #     """
    #     expected_graph = deepcopy(t_cell_signaling_pathway.graph)
    #     set_latent(expected_graph.directed, {Variable("Akt")})
    #     actual_graph = mark_latent(
    #         graph=t_cell_signaling_pathway.graph,
    #         treatments={Variable("Raf"), Variable("Mek")},
    #         outcomes=Variable("Erk"),
    #     )
    #     self.assertEqual(expected_graph, actual_graph)

    def test_find_nodes_on_all_paths_for_sars(self):
        """Test finding nodes in all causal paths for sars."""
        expected_nodes = {Variable("EGFR"), Variable("NFKB"), Variable("IL6AMP"), Variable("cytok")}
        actual_nodes = find_all_nodes_in_causal_paths(
            sars.graph, Variable("EGFR"), Variable("cytok")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    # def test_mark_latent_for_sars(self):
    #     """Test marking nodes as latent.
    #
    #     Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
    #     nodes for sars.
    #     """
    #     expected_graph = deepcopy(sars.graph)
    #     actual_graph = mark_latent(
    #         graph=sars.graph, treatments=Variable("EGFR"), outcomes=Variable("cytok")
    #     )
    #     self.assertEqual(expected_graph, actual_graph)

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

    # def test_mark_latent_for_t_cell_signaling_pathway_with_multiple_outcomes(self):
    #     """Test marking nodes as latent.
    #
    #     Test marking the descendants of nodes in all causal paths that are not ancestors of the outcome as latent
    #     nodes for t_cell_signaling_pathway with multiple outcomes.
    #     """
    #     expected_graph = deepcopy(t_cell_signaling_pathway.graph)
    #     set_latent(expected_graph.directed, {Variable("Jnk"), Variable("P38")})
    #     actual_graph = mark_latent(
    #         graph=t_cell_signaling_pathway.graph,
    #         treatments={Variable("PIP2"), Variable("PIP3")},
    #         outcomes={Variable("Erk"), Variable("Akt")},
    #     )
    #     self.assertEqual(expected_graph, actual_graph)

    def test_find_nuisance_variables_for_simple_graph(self):
        """Test find nuisance variable for a simple graph."""
        pass

    def test_find_nuisance_variables_for_multi_med(self):
        """Test find nuisance variable for multi_med example."""
        pass

    def test_find_nuisance_variables_for_sars(self):
        """Test find nuisance variable for sars example."""
        pass

    def test_find_nuisance_variables_for_t_cell(self):
        """Test find nuisance variable for t_cell_signaling_pathway example."""
        pass
