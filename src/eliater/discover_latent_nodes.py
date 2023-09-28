"""This module contains methods to discover and mark latent nodes in a network."""

from typing import Set, Union

import networkx as nx
from y0.dsl import Variable
from y0.graph import NxMixedGraph, set_latent


def find_all_nodes_in_causal_paths(
    graph: NxMixedGraph, treatments: Union[Variable, Set[Variable]], outcome: Variable
) -> Set[Variable]:
    """Find all the nodes in causal paths from source to destination."""
    if isinstance(treatments, Variable):
        treatments = {treatments}
    nodes = set()
    for treatment in treatments:
        for causal_path in nx.all_simple_paths(graph.directed, treatment, outcome):
            for node in causal_path:
                nodes.add(node)
    return nodes


def mark_latent(
    graph: NxMixedGraph, treatments: Union[Variable, Set[Variable]], outcome: Variable
) -> NxMixedGraph:
    """Mark the latent nodes in the graph.

    Marks the descendants of nodes in all causal paths that are not ancestors of the outcome variable as latent
    nodes.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcome: the outcome variable
    :returns: The modified graph marked with latent nodes.
    """
    if isinstance(treatments, Variable):
        treatments = {treatments}
    # Find the nodes on causal paths
    nodes_on_causal_paths = find_all_nodes_in_causal_paths(
        graph=graph, treatments=treatments, outcome=outcome
    )
    # Find the descendants for the nodes on the causal paths
    descendants_of_nodes_on_causal_paths = graph.descendants_inclusive(nodes_on_causal_paths)
    # Find the ancestors of the outcome variable
    ancestors_of_outcome = graph.ancestors_inclusive(outcome)
    # Descendants of nodes on causal paths that are not ancestors of the outcome variable
    descendants_not_ancestors = descendants_of_nodes_on_causal_paths.difference(
        ancestors_of_outcome
    )
    # Remove treatments and outcome
    descendants_not_ancestors = descendants_not_ancestors.difference(treatments.union({outcome}))
    # Mark nodes as latent
    set_latent(graph.directed, descendants_not_ancestors)
    return graph
