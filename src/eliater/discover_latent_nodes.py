"""This module contains methods to discover nuisance nodes in a network.

Given an acyclic directed mixed graph (ADMG), along with the treatment and the outcome
of interest, certain observable variables can be regarded as nuisances. This
classification arises because they do not have any impact on the outcome and should not
be involved in the estimation of the treatment's effect on the outcome. These specific
variables are descendants of the variables on all causal paths that are not ancestors of
the outcome. A causal path, in this context, refers to a directed path that starts from the
treatment and leads to the outcome such that all the arrows on the path have the same direction.
This module is designed to identify these variables.

This process enables us to concentrate on the fundamental variables needed to estimate the
treatment's impact on the outcome. This focus results in more precise estimates with reduced
variance and bias. In addition, if this process is combined with the simplification module in y0:
y0.algorithm.simplify_latent.simplify_latent_dag() it can help to remove the nuisance variables
from the graph which leads to simpler, more interpretable, and visually more appealing result.

Here is an example of an ADMG where X is the treatment and Y is the outcome. This ADMG has
only one causal path from X to Y which is X -> M1 -> M2 -> Y. The descendants of these variables
that are not ancestors of the outcome are R1, R2, and R3. The goal of this example is to identify these
nuisance variables.

.. code-block:: python

    from eliater import mark_nuisance_variables_as_latent
    from eliater.simplify_latent import simplify_latent_dag
    from y0.algorithm.identify import identify_outcomes
    from y0.dsl import Variable, X, Y
    from y0.graph import NxMixedGraph

    M1, M2, R1, R2, R3 = (Variable(x) for x in ("M1", "M2", "R1", "R2", "R3"))

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

    new_graph = remove_latent_variables(graph, treatments=X, outcomes=Y)

The nuisance variables are identified as R1, R2, and R3. The input ADMG is converted to a latent variable DAG where
bi-directed edges are assigned as latent nodes upstream of their two incident nodes. R1, R2, and R3 are
marked as latent in the latent variable DAG.

.. code-block:: python

    simplified_latent_dag = simplify_latent_dag(lv_dag)
    new_graph = NxMixedGraph.from_latent_variable_dag(simplified_latent_dag.graph, tag=tag)

The simplification rules can then be applied to the latent variable DAG to simplify the dag. The simplified
latent dag does not include the nuisance variables. The latent variable DAG can then be converted back to an ADMG.
The new graph is simpler than the original graph and only contains variables necessary for estimation of the
causal effect of interest.

.. code-block:: python

    estimand = identify_outcomes(new_graph, treatments=X, outcomes=Y)

The new graph can be used to check if the query is identifiable, and if so, generate an estimand for it.
"""
import itertools
from typing import Iterable, Optional, Set, Union

import networkx as nx

from eliater.simplify_latent import simplify_latent_dag
from y0.dsl import Variable
from y0.graph import DEFAULT_TAG, NxMixedGraph

__all__ = [
    "remove_latent_variables",
    "mark_nuisance_variables_as_latent",
    "find_all_nodes_in_causal_paths",
    "find_nuisance_variables",
]


def remove_latent_variables(
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
    tag: Optional[str] = None,
) -> NxMixedGraph:
    """Find all nuissance variables and remove them based on Evans' simplification rules.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcomes: a list of outcomes
    :param tag: The tag for which variables are latent
    :return: the new graph after simplification
    """
    # This is the high-level access point, the only function anyone will ever want
    # to directly use from this module.
    lv_dag = mark_nuisance_variables_as_latent(
        graph=graph, treatments=treatments, outcomes=outcomes, tag=tag
    )
    simplified_latent_dag = simplify_latent_dag(lv_dag, tag=tag)
    return NxMixedGraph.from_latent_variable_dag(simplified_latent_dag.graph, tag=tag)


def mark_nuisance_variables_as_latent(
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
    tag: Optional[str] = None,
) -> nx.DiGraph:
    """Find all the nuisance variables and mark them as latent.

    Mark nuisance variables as latent by first identifying them, then creating a new graph where these
    nodes are marked as latent. Nuisance variables are the descendants of nodes in all proper causal paths
    that are not ancestors of the outcome variables nodes. A proper causal path is a directed path from
    treatments to the outcome. Nuisance variables should not be included in the estimation of the causal
    effect as they increase the variance.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcomes: a list of outcomes
    :param tag: The tag for which variables are latent
    :return: the modified graph after simplification, in place
    """
    if tag is None:
        tag = DEFAULT_TAG
    nuisance_variables = find_nuisance_variables(graph, treatments=treatments, outcomes=outcomes)
    lv_dag = NxMixedGraph.to_latent_variable_dag(graph, tag=tag)
    # Set nuisance variables as latent
    for node, data in lv_dag.nodes(data=True):
        if Variable(node) in nuisance_variables:
            data[tag] = True
    # simplified_latent_dag = simplify_latent_dag(lv_dag, tag=tag)
    # return NxMixedGraph.from_latent_variable_dag(simplified_latent_dag.graph, tag=tag)
    return lv_dag


def find_all_nodes_in_causal_paths(
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
) -> Set[Variable]:
    """Find all the nodes in proper causal paths from treatments to outcomes.

    A proper causal path is a directed path from treatments to the outcome.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcomes: a list of outcomes
    :return: the nodes on all causal paths from treatments to outcomes.
    """
    if isinstance(treatments, Variable):
        treatments = {treatments}
    if isinstance(outcomes, Variable):
        outcomes = {outcomes}

    return {
        node
        for treatment, outcome in itertools.product(treatments, outcomes)
        for causal_path in nx.all_simple_paths(graph.directed, treatment, outcome)
        for node in causal_path
    }


def find_nuisance_variables(
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
) -> Iterable[Variable]:
    """Find the nuisance variables in the graph.

    Nuisance variables are the descendants of nodes in all proper causal paths that are
    not ancestors of the outcome variables' nodes. A proper causal path is a directed path
    from treatments to the outcome. Nuisance variables should not be included in the estimation
    of the causal effect as they increase the variance.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcomes: a list of outcomes
    :returns: The nuisance variables.
    """
    if isinstance(treatments, Variable):
        treatments = {treatments}
    if isinstance(outcomes, Variable):
        outcomes = {outcomes}

    # Find the nodes on all causal paths
    nodes_on_causal_paths = find_all_nodes_in_causal_paths(
        graph=graph, treatments=treatments, outcomes=outcomes
    )

    # Find the descendants of interest
    descendants_of_nodes_on_causal_paths = graph.descendants_inclusive(nodes_on_causal_paths)

    # Find the ancestors of outcome variables
    ancestors_of_outcomes = graph.ancestors_inclusive(outcomes)

    descendants_not_ancestors = descendants_of_nodes_on_causal_paths.difference(
        ancestors_of_outcomes
    )

    nuisance_variables = descendants_not_ancestors.difference(treatments.union(outcomes))
    return nuisance_variables
