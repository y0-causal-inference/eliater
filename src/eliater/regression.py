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

from operator import attrgetter
from typing import TYPE_CHECKING, Literal, Optional, Sequence

import pandas as pd
from sklearn.linear_model import LinearRegression

from y0.dsl import Variable
from y0.graph import NxMixedGraph, _ensure_set

if TYPE_CHECKING:
    import optimaladj.CausalGraph
    import pgmpy.models


__all__ = [
    "get_eliater_regression",
    "get_regression_coefficients",
    "get_adjustment_sets",
]


Impl = Literal["pgmpy", "optimaladj"]


def get_eliater_regression(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatment: Variable,
    outcome: Variable,
    impl: Optional[Impl] = None,
) -> float:
    """Return a simplified view for a simplified regression scenario.

    1. Gets all adjustment sets returned by the implementation. If using :mod:`pgmpy`, then
       returns multiple. The shortest is non-deterministically chosen. If using  :mod:`optimaladj`,
       then a single optimal set is chosen (though might error if not possible)
    2. Fits a linear regression between the union of the treatment and variables in the adjustment
       set against the outcome
    3. Returns the coefficient for the treatment in the linear regression, which represents the direct
       effect of the treatment on the outcome

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatment: The treatment variable
    :param outcome: The outcome variable
    :param impl: The implementation for getting adjustment sets.
    :returns: The coefficient for the treatment in the linear regression between
        the union of the treatment and (optimal/chosen) adjustment set and the outcome
        as the response
    """
    adjustment_set_to_variable_to_coefficient = get_regression_coefficients(
        graph=graph,
        data=data,
        treatments=treatment,
        outcomes=outcome,
    )[outcome]
    if impl == "pgmpy":
        # TODO how else to do this aggregation?
        #  Return a distribution of all treatment coefficients?
        #  Average them?
        adjustment_set = min(adjustment_set_to_variable_to_coefficient, key=len)
    elif impl == "optimaladj":
        assert len(adjustment_set_to_variable_to_coefficient) == 1
        adjustment_set = list(adjustment_set_to_variable_to_coefficient)[0]
    else:
        raise TypeError
    return adjustment_set_to_variable_to_coefficient[adjustment_set][treatment]


def get_regression_coefficients(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcomes: Variable | set[Variable],
    conditions: None | Variable | set[Variable] = None,
) -> dict[Variable, dict[frozenset[Variable], dict[Variable, float]]]:
    """Get the regression coefficients for potentially multiple adjustment sets w.r.t. the treatments and outcomes.

    :returns:
        A three level dictionary from outcome -> adjustment set -> variable -> coefficient
    """
    rv = {}
    for outcome in _ensure_set(outcomes):
        regressions = fit_regressions(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            conditions=conditions,
        )
        rv[outcomes] = {
            adjustment_set: dict(zip(variables, model.coef_))
            for adjustment_set, variables, model in regressions
        }
    return rv


def fit_regressions(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    conditions: None | Variable | set[Variable] = None,
    impl: Optional[Impl] = None,
) -> list[tuple[frozenset[Variable], Sequence[Variable], LinearRegression]]:
    """Fit a regression model to each adjustment set over the treatments and a given outcome."""
    if conditions is not None:
        raise NotImplementedError
    treatments = _ensure_set(treatments)
    rv = []
    adjustment_sets = get_adjustment_sets(
        graph=graph, treatments=treatments, outcome=outcome, impl=impl
    )
    for adjustment_set in adjustment_sets:
        variable_set = adjustment_set.union(treatments).difference({outcome})
        variables = sorted(variable_set, key=attrgetter("name"))
        model = LinearRegression()
        model.fit(data[[v.name for v in variables]], data[outcome.name])
        rv.append((adjustment_set, variables, model))
    return rv


def to_causal_graph(graph: NxMixedGraph) -> "optimaladj.CausalGraph.CausalGraph":
    """Convert a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
    from optimaladj.CausalGraph import CausalGraph

    causal_graph = CausalGraph()
    ananke_admg = graph.to_admg()
    causal_graph.add_edges_from(ananke_admg.di_edges)
    for i, (node1, node2) in enumerate(ananke_admg.bi_edges, start=1):
        latent = "U_{}".format(i)
        causal_graph.add_edge(latent, node1)
        causal_graph.add_edge(latent, node2)
    return causal_graph


def to_bayesian_network(graph: NxMixedGraph) -> "pgmpy.models.BayesianNetwork":
    """Convert a mixed graph to an equivalent :class:`pgmpy.BayesianNetwork`."""
    from pgmpy.models import BayesianNetwork

    ananke_admg = graph.to_admg()
    ananke_dag = ananke_admg.canonical_dag()
    di_edges = ananke_dag.di_edges
    bi_edges = ananke_admg.bi_edges
    # TODO test this carefully
    latents = ["U_{}_{}".format(*sorted([node1, node2])) for node1, node2 in bi_edges]
    model = BayesianNetwork(ebunch=di_edges, latents=latents)
    return model


def get_adjustment_sets(
    graph: NxMixedGraph,
    treatments: Variable | set[Variable],
    outcome: Variable,
    impl: Optional[Impl] = None,
) -> set[frozenset[Variable]]:
    """Get the optimal adjustment set for estimating the direct effect of treatments on a given outcome."""
    treatments = list(_ensure_set(treatments))
    if len(treatments) > 1:
        raise NotImplementedError
    if impl is None or impl == "pgmpy":
        from pgmpy.inference.CausalInference import CausalInference

        model = to_bayesian_network(graph)
        inference = CausalInference(model)
        adjustment_sets = inference.get_all_backdoor_adjustment_sets(
            treatments[0].name, outcome.name
        )
        return {
            frozenset(Variable(v) for v in adjustment_set) for adjustment_set in adjustment_sets
        }
    elif impl == "optimaladj":
        causal_graph = to_causal_graph(graph)
        non_latent_nodes = graph.to_admg().vertices
        adjustment_set = causal_graph.optimal_minimum_adj_set(
            treatment=treatments[0].name, outcome=outcome.name, L=[], N=non_latent_nodes
        )
        return {frozenset(Variable(v) for v in adjustment_set)}
    else:
        raise TypeError(f"Unknown implementation: {impl}")


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
