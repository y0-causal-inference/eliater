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

import statistics
from operator import attrgetter
from typing import Dict, Literal, NamedTuple, Optional, Sequence, Tuple

import networkx.exception
import pandas as pd
from sklearn.linear_model import LinearRegression

from eliater.io import to_bayesian_network, to_causal_graph
from y0.dsl import Variable
from y0.graph import NxMixedGraph, _ensure_set

__all__ = [
    # High-level functions
    "estimate_query",
    "estimate_ate",
    "estimate_probabilities",
    # Helper functions
    "get_regression_results",
    "get_adjustment_sets",
    "fit_regressions",
]


Impl = Literal["pgmpy", "optimaladj"]


class RegressionResult(NamedTuple):
    """Represents a regression."""

    coefficients: dict[Variable, float]
    intercept: float


def estimate_ate(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatment: Variable,
    outcome: Variable,
    *,
    impl: Optional[Impl] = None,
    conditions: None | Variable | set[Variable] = None,
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
    :param conditions: Conditions to apply to the query
    :param impl: The implementation for getting adjustment sets.
    :returns: The coefficient for the treatment in the linear regression between
        the union of the treatment and (optimal/chosen) adjustment set and the outcome
        as the response
    """
    _adjustment_set, results = _get_regression_result(
        graph=graph,
        data=data,
        treatments=treatment,
        outcome=outcome,
        impl=impl,
        conditions=conditions,
    )
    # TODO how would you aggregate the coefficients on multiple treatments?
    return results.coefficients[treatment]


def _get_regression_result(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    *,
    impl: Optional[Impl] = None,
    conditions: None | Variable | set[Variable] = None,
) -> tuple[frozenset[Variable], RegressionResult]:
    """Return the adjustment set, the coefficients, and the intercept."""
    adjustment_set_to_variable_to_coefficient = get_regression_results(
        graph=graph,
        data=data,
        treatments=treatments,
        outcomes=outcome,
        conditions=conditions,
        impl=impl,
    )[outcome]
    if impl == "pgmpy" or impl is None:
        # TODO how else to do this aggregation over multiple adjustment sets?
        #  Return a distribution of coefficients for each treatment?
        #  Average them?
        adjustment_set = min(adjustment_set_to_variable_to_coefficient, key=len)
    elif impl == "optimaladj":
        if 1 != len(adjustment_set_to_variable_to_coefficient):
            raise RuntimeError  # this shouldn't be possible
        adjustment_set = list(adjustment_set_to_variable_to_coefficient)[0]
    else:
        raise TypeError(f"Invalid implementation: {impl}")
    return adjustment_set, adjustment_set_to_variable_to_coefficient[adjustment_set]


def get_regression_results(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcomes: Variable | set[Variable],
    *,
    conditions: None | Variable | set[Variable] = None,
    impl: Optional[Impl] = None,
) -> dict[Variable, dict[frozenset[Variable], RegressionResult]]:
    """Get the regression coefficients for potentially multiple adjustment sets w.r.t. the treatments and outcomes.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatments: The treatment variable(s)
    :param outcomes: The outcome variable(s)
    :param conditions: Conditions to apply to the query
    :param impl: The implementation for getting adjustment sets.
    :returns:
        A two-level dictionary from outcome -> adjustment set -> regression result where
        the regression result contains a dictionary of variables to coefficient values and
        the regression's intercept value
    """
    rv = {}
    for outcome in _ensure_set(outcomes):
        regressions = fit_regressions(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            conditions=conditions,
            impl=impl,
        )
        rv[outcomes] = {
            adjustment_set: RegressionResult(dict(zip(variables, model.coef_)), model.intercept_)
            for adjustment_set, variables, model in regressions
        }
    return rv


def fit_regressions(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    *,
    conditions: None | Variable | set[Variable] = None,
    impl: Optional[Impl] = None,
) -> Sequence[tuple[frozenset[Variable], Sequence[Variable], LinearRegression]]:
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


def get_adjustment_sets(
    graph: NxMixedGraph,
    treatments: Variable | set[Variable],
    outcome: Variable,
    *,
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


def get_adjustment_set(
    graph: NxMixedGraph, treatments: Variable | set[Variable], outcome: Variable
) -> Tuple[frozenset[Variable], str]:
    """Get the optimal adjustment set for estimating the direct effect of treatments on a given outcome."""
    import optimaladj.CausalGraph

    treatments = list(_ensure_set(treatments))
    if len(treatments) > 1:
        raise NotImplementedError

    causal_graph = to_causal_graph(graph)
    observable_nodes = graph.to_admg().vertices

    try:
        adjustment_set = causal_graph.optimal_adj_set(
            treatment=treatments[0].name, outcome=outcome.name, L=[], N=observable_nodes
        )
        adjustment_set_type = "Optimal Adjustment Set"
    except (
        networkx.exception.NetworkXError,
        optimaladj.CausalGraph.NoAdjException,
        optimaladj.CausalGraph.ConditionException,
    ):
        try:
            adjustment_set = causal_graph.optimal_minimal_adj_set(
                treatment=treatments[0].name, outcome=outcome.name, L=[], N=observable_nodes
            )
            adjustment_set_type = "Optimal Minimal Adjustment Set"
        except (
            networkx.exception.NetworkXError,
            optimaladj.CausalGraph.NoAdjException,
            optimaladj.CausalGraph.ConditionException,
        ):
            from pgmpy.inference.CausalInference import CausalInference

            model = to_bayesian_network(graph)
            inference = CausalInference(model)
            adjustment_sets = inference.get_all_backdoor_adjustment_sets(
                treatments[0].name, outcome.name
            )
            adjustment_set = min(adjustment_sets, key=len)
            adjustment_set_type = "Minimal Adjustment Set"
    return frozenset(Variable(v) for v in adjustment_set), adjustment_set_type


def fit_regression(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    conditions: None | Variable | set[Variable] = None,
) -> RegressionResult:
    """Fit a regression model to the adjustment set over the treatments and a given outcome."""
    # TODO this is duplicating existing functionality, can delete this entire function
    if conditions is not None:
        raise NotImplementedError
    treatments = _ensure_set(treatments)
    adjustment_set = get_adjustment_set(graph=graph, treatments=treatments, outcome=outcome)[0]
    variable_set = adjustment_set.union(treatments).difference({outcome})
    variables = sorted(variable_set, key=attrgetter("name"))
    model = LinearRegression()
    model.fit(data[[v.name for v in variables]], data[outcome.name])
    return RegressionResult(dict(zip(variables, model.coef_)), model.intercept_)


def estimate_query(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    *,
    query_type: Literal["ate", "expected_value", "probability"] = "ate",
    conditions: None | Variable | set[Variable] = None,
    interventions: Dict[Variable, float] | None = None,
) -> float | list[float]:
    """Estimate treatment effects using Linear Regression."""
    treatments = _ensure_set(treatments)

    if query_type == "ate":
        if len(treatments) > 1:
            raise NotImplementedError
        treatment = list(treatments)[0]
        return estimate_ate(
            graph=graph,
            data=data,
            treatment=treatment,
            outcome=outcome,
            conditions=conditions,
        )

    elif query_type in {"expected_value", "probability"}:
        if interventions is None:
            raise ValueError(f"interventions must be given for query type: {query_type}")
        y = estimate_probabilities(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            conditions=conditions,
            interventions=interventions,
        )
        if query_type == "probability":
            return y
        return statistics.fmean(y)

    else:
        raise TypeError(f"Unknown query type {query_type}")


def estimate_probabilities(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    interventions: Dict[Variable, float],
    *,
    conditions: None | Variable | set[Variable] = None,
) -> list[float]:
    treatments = _ensure_set(treatments)
    missing = set(interventions).difference(treatments)
    if missing:
        raise ValueError(f"Missing treatments: {missing}")

    # TODO reuse existing function
    # _, (coefficients, intercept) = _get_regression_result(
    #     graph, data, treatments=treatments, outcome=outcome, conditions=conditions
    # )
    coefficients, intercept = fit_regression(
        graph, data, treatments=treatments, outcome=outcome, conditions=conditions
    )

    y = [
        intercept
        + sum(
            coefficients[variable]
            * (interventions[variable] if variable in treatments else row[variable])
            for variable in coefficients
        )
        for row in data
    ]
    return y


def _demo():
    from eliater.frontdoor_backdoor import frontdoor_backdoor_example
    from y0.dsl import X, Y

    graph = frontdoor_backdoor_example.graph
    data = frontdoor_backdoor_example.generate_data(1000)
    treatments = {X}
    outcome = Y
    coefficients = get_regression_results(
        graph=graph, data=data, treatments=treatments, outcomes=outcome
    )
    print(coefficients)  # noqa:T201


if __name__ == "__main__":
    _demo()
