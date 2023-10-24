"""This module tests the methods involved in discovering and marking latent nodes."""

import unittest
from copy import deepcopy

from eliater.discover_latent_nodes import (
    find_all_nodes_in_causal_paths,
    find_nuisance_variables,
    remove_latent_variables,
)
from eliater.examples import multi_causal_path, sars, t_cell_signaling_pathway
from eliater.frontdoor_backdoor import multiple_mediators_single_confounder_example
from y0.dsl import Variable


class TestDiscoverAndMarkLatentNodes(unittest.TestCase):
    """This class implements tests to verify the correctness of methods to discover and mark latent nodes."""

    def test_find_nodes_on_all_causal_paths_for_simple_graph(self):
        """Test finding nodes in all causal paths for a simple network with multiple causal paths.

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
            graph=multi_causal_path.graph,
            treatments={Variable("X1"), Variable("X2"), Variable("X3")},
            outcomes={Variable("Y")},
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_find_nodes_on_all_paths_for_multi_med(self):
        """Test finding nodes in all causal paths for multi_med."""
        expected_nodes = {Variable("X"), Variable("M2"), Variable("M1"), Variable("Y")}
        actual_nodes = find_all_nodes_in_causal_paths(
            multiple_mediators_single_confounder_example.graph, Variable("X"), Variable("Y")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_find_nodes_on_all_paths_for_t_cell_signaling_pathway(self):
        """Test finding nodes in all causal paths for t_cell_signaling_pathway."""
        expected_nodes = {Variable("Raf"), Variable("Mek"), Variable("Erk")}
        actual_nodes = find_all_nodes_in_causal_paths(
            t_cell_signaling_pathway.graph, {Variable("Raf"), Variable("Mek")}, Variable("Erk")
        )
        self.assertEqual(expected_nodes, actual_nodes)

    def test_find_nodes_on_all_paths_for_sars(self):
        """Test finding nodes in all causal paths for sars."""
        expected_nodes = {Variable("EGFR"), Variable("NFKB"), Variable("IL6AMP"), Variable("cytok")}
        actual_nodes = find_all_nodes_in_causal_paths(
            sars.graph, Variable("EGFR"), Variable("cytok")
        )
        self.assertEqual(expected_nodes, actual_nodes)

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

    def test_find_nuisance_variables_for_simple_graph(self):
        """Test finding nuisance variables for a simple network with multiple causal paths."""
        expected_nuisance_variables = {Variable(v) for v in ["E", "F", "G", "H"]}
        actual_nuisance_variables = find_nuisance_variables(
            graph=multi_causal_path.graph,
            treatments={Variable(v) for v in ["X1", "X2", "X3"]},
            outcomes={Variable("Y")},
        )
        self.assertEqual(expected_nuisance_variables, actual_nuisance_variables)

    def test_find_nuisance_variables_for_multi_med(self):
        """Test finding nuisance variables for multi_med example."""
        expected_nuisance_variables = set()
        actual_nuisance_variables = find_nuisance_variables(
            graph=multiple_mediators_single_confounder_example.graph,
            treatments={Variable("X")},
            outcomes={Variable("Y")},
        )
        self.assertEqual(expected_nuisance_variables, actual_nuisance_variables)

    def test_find_nuisance_variables_for_sars(self):
        """Test finding nuisance variables for sars example."""
        expected_nuisance_variables = set()
        actual_nuisance_variables = find_nuisance_variables(
            graph=sars.graph, treatments=Variable("EGFR"), outcomes=Variable("cytok")
        )
        self.assertEqual(expected_nuisance_variables, actual_nuisance_variables)

    def test_find_nuisance_variables_for_t_cell(self):
        """Test finding nuisance variables for t_cell_signaling_pathway example."""
        expected_nuisance_variables = {Variable("Akt")}
        actual_nuisance_variables = find_nuisance_variables(
            graph=t_cell_signaling_pathway.graph,
            treatments={Variable(v) for v in ["Raf", "Mek", "Erk"]},
            outcomes={Variable("Erk")},
        )
        self.assertEqual(expected_nuisance_variables, actual_nuisance_variables)

    def test_find_nuisance_variables_for_t_cell_signaling_pathway_with_multiple_outcomes(
        self,
    ):
        """Test finding nuisance variables for t_cell_signaling_pathway with multiple outcomes."""
        expected_nuisance_variables = set()
        actual_nuisance_variables = find_nuisance_variables(
            graph=t_cell_signaling_pathway.graph,
            treatments={Variable(v) for v in ["Raf", "Mek", "Erk"]},
            outcomes={Variable("Erk"), Variable("Akt")},
        )
        self.assertEqual(expected_nuisance_variables, actual_nuisance_variables)

    def test_remove_latent_variables_for_simple_graph(self):
        """Test removing latents variables for a simple network with multiple causal paths."""
        expected_graph = deepcopy(multi_causal_path.graph)
        expected_graph = expected_graph.remove_in_edges([Variable(v) for v in ["E", "F", "G", "H"]])
        expected_graph = expected_graph.remove_nodes_from(
            [Variable(v) for v in ["E", "F", "G", "H"]]
        )
        actual_graph = remove_latent_variables(
            graph=multi_causal_path.graph,
            treatments={Variable(v) for v in ["X1", "X2", "X3"]},
            outcomes={Variable("Y")},
        )
        self.assertEqual(actual_graph, expected_graph)

    def test_remove_latent_variables_for_multi_med(self):
        """Test removing latents variables for multi_med example."""
        expected_graph = deepcopy(multiple_mediators_single_confounder_example.graph)
        actual_graph = remove_latent_variables(
            graph=multiple_mediators_single_confounder_example.graph,
            treatments={Variable("X")},
            outcomes={Variable("Y")},
        )
        self.assertEqual(actual_graph, expected_graph)

    def test_remove_latent_variables_for_sars(self):
        """Test removing latents variables for sars example."""
        expected_graph = deepcopy(sars.graph)
        actual_graph = remove_latent_variables(
            graph=sars.graph, treatments=Variable("EGFR"), outcomes=Variable("cytok")
        )
        self.assertEqual(actual_graph, expected_graph)

    def test_remove_latent_variables_for_t_cell(self):
        """Test removing latents variables for t_cell_signaling_pathway example."""
        expected_graph = deepcopy(t_cell_signaling_pathway.graph)
        expected_graph = expected_graph.remove_in_edges(Variable("Akt"))
        expected_graph = expected_graph.remove_nodes_from(Variable("Akt"))
        actual_graph = remove_latent_variables(
            graph=t_cell_signaling_pathway.graph,
            treatments={Variable(v) for v in ["Raf", "Mek", "Erk"]},
            outcomes={Variable("Erk")},
        )
        self.assertEqual(actual_graph, expected_graph)

    def test_remove_latent_variables_for_t_cell_signaling_pathway_with_multiple_outcomes(
        self,
    ):
        """Test removing latents variables for t_cell_signaling_pathway with multiple outcomes."""
        expected_graph = deepcopy(t_cell_signaling_pathway.graph)
        actual_graph = remove_latent_variables(
            graph=t_cell_signaling_pathway.graph,
            treatments={Variable(v) for v in ["Raf", "Mek", "Erk"]},
            outcomes={Variable("Erk"), Variable("Akt")},
        )
        self.assertEqual(actual_graph, expected_graph)
