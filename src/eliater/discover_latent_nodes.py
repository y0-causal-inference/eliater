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

    import eliater
    import y0
    from y0.algorithm.identify import identify_outcomes
    from y0.dsl import Variable, X, Y
    from y0.graph import NxMixedGraph, set_latent
    from y0.algorithm.simplify_latent import simplify_latent_dag
    from eliater.discover_latent_nodes import find_nuisance_variables

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
    
    nuisance_variables = find_nuisance_variables(graph, treatments=X, outcomes=Y)

    lv_dag = NxMixedGraph.to_latent_variable_dag(graph)

    set_latent(lv_dag, nuisance_variables)  # set the nuisance variables as latent

    simplified_lv_dag = simplify_latent_dag(lv_dag)

    #convert the simplified latent variable dag to a NxMixedGraph object
    new_graph = NxMixedGraph.from_latent_variable_dag(simplified_lv_dag)

The nuisance variables are identified as R1, R2, and R3. The input ADMG is converted to a latent variable DAG where
bi-directed edges are assigned as latent nodes upstream of their two incident nodes. R1, R2, and R3 are
marked as latent in the latent variable DAG. The simplification rules is then applied to the latent variable DAG to
remove the nuisance variables from the graph. The latent variable DAG is then converted back to an ADMG. The new graph
is simpler than the original graph and only contains variables necessary for estimation of the causal effect of interest.

.. code-block:: python

    estimand = identify_outcomes(new_graph, treatments=X, outcomes=Y)

The new graph can be used to check if the query is identifiable, and if so, generate an estimand for it.

"""

from typing import Set, Union

import networkx as nx

from y0.dsl import Variable
from y0.graph import NxMixedGraph, set_latent
from y0.algorithm.simplify_latent import simplify_latent_dag

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

def find_nuisance_variables(
    graph: NxMixedGraph,
    treatments: Union[Variable, Set[Variable]],
    outcomes: Union[Variable, Set[Variable]],
) -> Iterable[Variable]:
    """find the nuisance nodes in the graph.

        finds the descendants of nodes in all causal paths that are not ancestors of the outcome variables'
        nodes. These nodes should not be included in the estimation of the causal effect.

        :param graph: an NxMixedGraph
        :param treatments: a list of treatments
        :param outcomes: a list of outcomes
        :returns: The nuisance variables.
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

    return descendants_not_ancestors



def mark_latent(graph: NxMixedGraph,
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
    
    nuisance_variables = find_nuisance_variables(graph, treatments, outcomes)

    #  going through the Latent DAG workflow
    if nuisance_variables:
        lv_dag = NxMixedGraph.to_latent_variable_dag(graph)
        set_latent(lv_dag, nuisance_variables)  # set the nuisance variables as latent
        simplified_lv_dag = simplify_latent_dag(lv_dag)
        #convert the simplified lavent variable dag to a NxMixedGraph object
        graph = NxMixedGraph.from_latent_variable_dag(simplified_lv_dag)
        
    return graph
