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
variance and bias. In addition, if this process is combined with the simplification function
:func:`y0.algorithm.simplify_latent.simplify_latent_dag` it can help to remove the nuisance variables
from the graph which leads to simpler, more interpretable, and visually more appealing result.

Example
-------
We'll work with the following example where $X$ is the treatment, $Y$ is the outcome,
and the estimand from the ID algorithm is $\\sum\\limits_{M_1} P(M_1 | X) \\sum\\limits_{X} P(X) P(Y | M_1, X)$.

.. code-block:: python

    from eliater.discover_latent_nodes import remove_nuisance_variables
    from y0.algorithm.identify import identify_outcomes
    from y0.dsl import Variable, X, Y
    from y0.graph import NxMixedGraph

    M1, R1, R2, R3 = (Variable(x) for x in ("M1", "R1", "R2", "R3"))

    graph = NxMixedGraph.from_edges(
        directed=[
            (X, M1),
            (M1, Y),
            (M1, R1),
            (R1, R2),
            (R2, R3),
            (Y, R3),
        ],
        undirected=[
            (X, Y),
        ],
    )
    estimand = identify_outcomes(graph, X, Y)
    graph.draw()

.. figure:: img/nuisance/original.svg
    :scale: 70%

There is one causal path between the treatment ($X$ ) and outcome ($Y$): $X$ -> $M_1$ -> $Y$.

The descendants of these variables that are not ancestors of the outcome are $R_1$, $R_2$, and $R_3$.
The :func:`eliater.remove_nuisance_variables` function identifies and removes these variables from the
graph.

.. code-block:: python

    new_graph = remove_nuisance_variables(graph, treatments=X, outcomes=Y)
    new_graph.draw()

.. figure:: img/nuisance/reduced.svg
   :scale: 70%

The new graph can be used to check if the query is identifiable, and if so, generate an estimand for it.
Note that the estimand does not change since the nodes removed did not contribute to it.

.. code-block:: python

    identify_outcomes(new_graph, treatments=X, outcomes=Y)

$\\sum\\limits_{M_1} P(M_1 | X) \\sum\\limits_{X} P(X) P(Y | M_1, X)$

Explanation
-----------

.. code-block:: python

    # Minimal example for evans rule 1 (Transforming latent nodes)
    from y0.algorithm.simplify_latent import simplify_latent_dag
    import networkx as nx
    from y0.dsl import X, Y, Z1
    from y0.graph import set_latent

    graph = nx.DiGraph()
    graph.add_edges_from([(X, Z1), (Z1, Y), (Z1, Z2)])
    set_latent(graph, [Z1])
    simplified_graph = simplify_latent_dag(graph).graph

The edges in the resultant graph are [($X$, $Z_2$), ($X$, $Y$), ($Z_1^{prime}$, $Z_2$), ($Z_1^{prime}$, $Y$)].
The parent of the latent node $Z_1$ becomes attached to latter's children ($Z_2$ and $Y$).
The edge between $X$ and $Z_1$ is removed, and $Z_1$ is transformed into $Z_1^{prime}$ while remaining connected
to its children.

.. code-block:: python

    # Minimal example for evans rule 2 (Removing widow latents)
    from y0.algorithm.simplify_latent import simplify_latent_dag
    import networkx as nx
    from y0.dsl import X, Y, Z1
    from y0.graph import set_latent

    graph = nx.DiGraph()
    graph.add_edges_from([(X, Z1), (X, Y)])
    set_latent(graph, [Z1])
    simplified_graph = simplify_latent_dag(graph).graph

The edges in the resultant graph are [($X$, $Y$)].
$Z_1$ is removed as it is a latent node with no children.

.. code-block:: python

    # Minimal example for evans rule 3 (Removing unidirectional latents)
    from y0.algorithm.simplify_latent import simplify_latent_dag
    import networkx as nx
    from y0.dsl import X, Y, Z1
    from y0.graph import set_latent

    graph = nx.DiGraph()
    graph.add_edges_from([(X, Z1), (Z1, Y), (X, Y)])
    set_latent(graph, [Z1])
    simplified_graph = simplify_latent_dag(graph).graph

The edges in the resultant graph are [($X$, $Y$)].
$Z_1$ is removed as it is a latent node with a single child.

.. code-block:: python

    # Minimal example for evans rule 4 (Removing redundant latents)
    from y0.algorithm.simplify_latent import simplify_latent_dag
    import networkx as nx
    from y0.dsl import X, Y, Z1, Z2, Z3, Z4
    from y0.graph import set_latent

    graph = nx.DiGraph()
    graph.add_edges_from([(X, Y), (Z1, Y), (Z1, Z2), (Z1, Z3), (Z4, Z2), (Z4, Z3)])
    set_latent(graph, [Z1, Z4])
    simplified_graph = simplify_latent_dag(graph).graph

The edges in the resultant graph are [($X$, $Y$), ($Z_1$, $Y$), ($Z_1$, $Z_2$), ($Z_1$, $Z_3$)].
$Z_4$ is removed as its children are a subset of $Z_1$'s children.
"""

# FIXME @sara the explanation section is not easy to follow. please revise it.

import itertools
from typing import Iterable, Optional, Set, Union

import networkx as nx

from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.dsl import Variable
from y0.graph import DEFAULT_TAG, NxMixedGraph

__all__ = [
    "remove_nuisance_variables",
]


def remove_nuisance_variables(
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
    tag: Optional[str] = None,
) -> NxMixedGraph:
    """Find all nuisance variables and remove them based on Evans' simplification rules.

    :param graph: an NxMixedGraph
    :param treatments: a list of treatments
    :param outcomes: a list of outcomes
    :param tag: The tag for which variables are latent
    :return: the new graph after simplification
    """
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
