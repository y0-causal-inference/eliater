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

Here is an example of an ADMG where X is the treatment and Y is the outcome is Y. This ADMG has
only one causal path from X to Y which is X -> M1 -> M2 -> Y. The descendants of these variables
that are ancestors of the outcome are R1, R2, and R3. The goal of this example is to identify these
nuisance variables and mark them as latent.

.. code-block:: python
    import y0
    import eliater
    from y0.graph import NxMixedGraph
    from y0.dsl import Variable, X, Y
    from eliater.discover_latent_nodes import mark_latent

    M1 = Variable("M1")
    M2 = Variable("M2")
    R1 = Variable("R1")
    R2 = Variable("R2")
    R3 = Variable("R3")

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

    new_graph = mark_latent(graph, treatments = Variable("X"), outcome: Variable("Y"))

The new graph now has R1, R2, and R3 marked as latent. Hence, these variables can't be
part of the estimation of the causal query. This decreases the estimation variance and
increases the accuracy of the query estimation.
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
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
) -> Set[Variable]:
    """Find all the nodes in causal paths from treatments to outcomes.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcomes: a list of outcomes
    :return: the nodes on all causal paths from treatments to outcomes.
    """
    if isinstance(treatments, Variable):
        treatments = {treatments}
    if isinstance(outcomes, Variable):
        outcomes = {outcomes}
    nodes = set()
    for treatment in treatments:
        for outcome in outcomes:
            for causal_path in nx.all_simple_paths(graph.directed, treatment, outcome):
                for node in causal_path:
                    nodes.add(node)
    return nodes


def mark_latent(
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
) -> NxMixedGraph:
    """Mark the latent nodes in the graph.

    Marks the descendants of nodes in all causal paths that are not ancestors of the outcome variables as latent
    nodes.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcomes: a list of outcomes
    :returns: The modified graph marked with latent nodes.
    """
    if isinstance(treatments, Variable):
        treatments = {treatments}
    if isinstance(outcomes, Variable):
        outcomes = {outcomes}
    # Find the nodes on causal paths
    nodes_on_causal_paths = find_all_nodes_in_causal_paths(
        graph=graph, treatments=treatments, outcomes=outcomes
    )
    # Find the descendants for the nodes on the causal paths
    descendants_of_nodes_on_causal_paths = graph.descendants_inclusive(nodes_on_causal_paths)
    # Find the ancestors of outcome variables
    ancestors_of_outcome = graph.ancestors_inclusive(outcomes)
    # Descendants of nodes on causal paths that are not ancestors of outcome variables
    descendants_not_ancestors = descendants_of_nodes_on_causal_paths.difference(
        ancestors_of_outcome
    )
    # Remove treatments and outcomes
    descendants_not_ancestors = descendants_not_ancestors.difference(treatments.union(outcomes))
    # Mark nodes as latent
    # FIXME this operation is currently meaningless in ADMGs, it's supposed to be used on graphs
    #  going through the Latent DAG workflow
    if descendants_not_ancestors:
        set_latent(graph.directed, descendants_not_ancestors)
    return graph

