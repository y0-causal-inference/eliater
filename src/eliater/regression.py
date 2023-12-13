"""
The goal is to determine the parameters of a structural causal model.

With linear SCMs, each edge has its own parameter.  How can you learn the
value of the parameter of the edge between X and Y?  If you naively regress X on Y,
then the coefficient Beta describes the total association between X and Y. This
includes not only the direct effect X -> Y, but also the association from all
backdoor paths between X and Y. This can be seen by generating data from a graph where
the edge between X and Y has been cut. The difference between Beta in the original
dataset and the coefficient Beta' you get from regressing Y on X in this new dataset
is the direct effect X -> Y.

.. warning::

    But there is an easier way to do it!  If you include a set of variables
    that block all backdoor paths between X and Y in the original regression,
    then the Beta coefficient associated with X will be the direct effect.
    But be careful! If you include extra variables in the regression that open a
    backdoor path between X and Y, then the Beta regression coefficient associated
    with X will no longer represent the direct effect
"""

from itertools import count
from operator import attrgetter
from typing import Sequence

import pandas as pd
from optimaladj.CausalGraph import CausalGraph
from pgmpy.inference.CausalInference import CausalInference
from pgmpy.models import BayesianNetwork
from sklearn.linear_model import LinearRegression

from y0.dsl import Variable
from y0.graph import NxMixedGraph, _ensure_set

__all__ = [
    "get_regression_coefficients",
]


def get_regression_coefficients(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcomes: Variable | set[Variable],
    conditions: None | Variable | set[Variable] = None,
) -> dict[Variable, dict[Variable, float]]:
    """Get the regression coefficients from fitting to the optimal adjustment set over the treatments and outcomes."""
    rv = {}
    for outcome in _ensure_set(outcomes):
        variables, model = fit_regression(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            conditions=conditions,
        )
        rv[outcomes] = dict(zip(variables, model.coef_))
    return rv


def fit_regression(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    conditions: None | Variable | set[Variable] = None,
) -> tuple[Sequence[Variable], LinearRegression]:
    """Fit a regression model to the optimal adjustment set over the treatments and a given outcome."""
    if conditions is not None:
        raise NotImplementedError
    treatments = _ensure_set(treatments)

    variable_set = (
        get_optimal_adjustment_set(graph=graph, treatments=treatments, outcome=outcome)
        .union(treatments)
        .difference({outcome})
    )
    variables = sorted(variable_set, key=attrgetter("name"))
    model = LinearRegression()
    model.fit(data[[v.name for v in variable_set]], data[outcome.name])
    return variables, model


def to_causal_graph(graph: NxMixedGraph) -> CausalGraph:
    """Convert y0 NxMixedGraph to equivalent optimaladj CausalGraph."""
    causal_graph = CausalGraph()
    ananke_admg = graph.to_admg()
    causal_graph.add_edges_from(ananke_admg.di_edges)
    counter = count(start=1)
    for node1, node2 in ananke_admg.bi_edges:
        latent = "U_{}".format(next(counter))
        causal_graph.add_edge(latent, node1)
        causal_graph.add_edge(latent, node2)
    return causal_graph


def to_bayesian_network(graph: NxMixedGraph) -> BayesianNetwork:
    """Convert y0 NxMixedGraph to equivalent pgmpy BayesianNetwork."""
    ananke_admg = graph.to_admg()
    ananke_dag = ananke_admg.canonical_dag()
    di_edges = ananke_dag.di_edges
    bi_edges = ananke_admg.bi_edges
    latents = ["U_{}_{}".format(*sorted([node1, node2])) for node1, node2 in bi_edges]
    model = BayesianNetwork(ebunch=di_edges, latents=latents)
    return model


def get_optimal_adjustment_set(
    graph: NxMixedGraph, treatments: Variable | set[Variable], outcome: Variable
) -> set[Variable]:
    """Get the optimal adjustment set for estimating the direct effect of treatments on a given outcome."""
    treatments = _ensure_set(treatments)
    if len(treatments) > 1:
        raise NotImplementedError(
            "Support for multiple treatments will be released in future versions."
        )
    treatments = list(treatments)
    try:
        causal_graph = to_causal_graph(graph)
        non_latent_nodes = graph.to_admg().vertices
        adjustment_set = causal_graph.optimal_minimum_adj_set(
            treatment=treatments[0].name, outcome=outcome.name, L=[], N=non_latent_nodes
        )
    except Exception:
        model = to_bayesian_network(graph)
        inference = CausalInference(model)
        adjustment_sets = inference.get_all_backdoor_adjustment_sets(
            treatments[0].name, outcome.name
        )
        adjustment_set = min(adjustment_sets, key=len)
    return adjustment_set


def _demo():
    from eliater.frontdoor_backdoor import frontdoor_backdoor_example
    from y0.dsl import X, Y

    graph = frontdoor_backdoor_example.graph
    data = frontdoor_backdoor_example.generate_data(1000)
    treatments = {X}
    outcome = Y
    coefficients = get_regression_coefficients(
        graph=graph, data=data, treatments=treatments, outcomes=outcome
    )
    print(coefficients)  # noqa:T201


if __name__ == "__main__":
    _demo()
