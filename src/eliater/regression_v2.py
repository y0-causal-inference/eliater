"""V2 of regression, to be finished later."""

from typing import Literal, Optional, Sequence, NamedTuple

import pandas as pd
from _operator import attrgetter
from sklearn.linear_model import LinearRegression

from eliater.io import to_bayesian_network, to_causal_graph
from y0.dsl import Variable
from y0.graph import NxMixedGraph, _ensure_set

Impl = Literal["pgmpy", "optimaladj"]


class RegressionResult(NamedTuple):
    """Represents a regression."""

    coefficients: dict[Variable, float]
    intercept: float


class MultipleTreatmentsNotImplementedError(NotImplementedError):
    """Raised when multiple treatments aren't yet allowed."""


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
        raise MultipleTreatmentsNotImplementedError
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


def fit_regressions(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    *,
    impl: Optional[Impl] = None,
) -> Sequence[tuple[frozenset[Variable], Sequence[Variable], LinearRegression]]:
    """Fit a regression model to each adjustment set over the treatments and a given outcome."""
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


def get_regression_results(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcomes: Variable | set[Variable],
    *,
    impl: Optional[Impl] = None,
) -> dict[Variable, dict[frozenset[Variable], RegressionResult]]:
    """Get the regression coefficients for potentially multiple adjustment sets w.r.t. the treatments and outcomes.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatments: The treatment variable(s)
    :param outcomes: The outcome variable(s)
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
            impl=impl,
        )
        rv[outcomes] = {
            adjustment_set: RegressionResult(dict(zip(variables, model.coef_)), model.intercept_)
            for adjustment_set, variables, model in regressions
        }
    return rv


def _get_regression_result(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    *,
    impl: Optional[Impl] = None,
) -> tuple[frozenset[Variable], RegressionResult]:
    """Return the adjustment set, the coefficients, and the intercept."""
    adjustment_set_to_variable_to_coefficient = get_regression_results(
        graph=graph,
        data=data,
        treatments=treatments,
        outcomes=outcome,
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


def estimate_ate(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatment: Variable,
    outcome: Variable,
    *,
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
    _adjustment_set, results = _get_regression_result(
        graph=graph,
        data=data,
        treatments=treatment,
        outcome=outcome,
        impl=impl,
    )
    # TODO how would you aggregate the coefficients on multiple treatments?
    return results.coefficients[treatment]
