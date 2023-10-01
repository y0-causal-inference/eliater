"""This module contains methods to discover and mark nuisance nodes in a network.

Given an acyclic directed mixed graph (ADMG), along with the treatment and the outcome
of interest, certain observable variables can be regarded as nuisances. This
classification arises because they do not have any impact on the outcome and should not
be involved in the estimation of the treatment's effect on the outcome. These specific
variables are descendants of the variables on all causal paths that are not ancestors of
the outcome. A causal path, in this context, refers to a directed path that starts from the
treatment and leads to the outcome such that all the arrows on the path have the same direction.
This module is designed to identify these variables and produce a new ADMG in which they are
designated as latent.

This process enables us to concentrate on the fundamental variables needed to estimate the
treatment's impact on the outcome. This focus results in more precise estimates with reduced
variance and bias.

Here is an example:

.. todo::

    Don't just give some random example. Motivate it. Explain the characteristics of the
    example ADMG that are important. Explain what the algorithm does to it.

.. code-block:: python

    graph = NxMixedGraph.from_edges(
        directed=[
            (X, M1),
            (M1, M2),
            (M2, Y),
            (M1, R1),
            (R1, R2),
            (R2, R3),
            (Y, R3),
        ],
        undirected=[
            (X, Y),
        ],
    )

    new_graph = mark_latent(graph, treatments = 'X', outcome: 'Y')

.. todo::

    And then what? what do we do with an ADMG that has variables marked as latent?
"""

from typing import Set, Union

import networkx as nx
from y0.dsl import Variable
from y0.graph import NxMixedGraph, set_latent


__all__ = [
    "find_all_nodes_in_causal_paths",
    "mark_latent",
]


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
