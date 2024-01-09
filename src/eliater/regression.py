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

.. todo:: Questions to answer in documentation:

    1. How does estimation with linear regression work? Rework the text above from JZ. Remember the point
       is to explain to someone who doesn't really care about the math but wants to decide if they should
       use it for their situation
    2. What's the difference between estimation with this module and what's available in Ananke?
    3. What are the limitations of estimation with this methodology?
    4. What does it look like to actually use this code? Give a self-contained example of doing estimation
       with this module and include an explanation on how users should interpret the results.

    PLEASE DO NOT DELETE THIS LIST. Leave it at the bottom of the module level docstring so we
    can more easily check if all of the points have been addressed.
"""

import statistics
from operator import attrgetter
from typing import Dict, Literal, NamedTuple, Tuple

import networkx.exception
import optimaladj
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
    "estimated_probabilities_summary_statistics",
    # Classes
    "RegressionResult",
    "MultipleTreatmentsNotImplementedError",
]


class RegressionResult(NamedTuple):
    """Represents a regression."""

    coefficients: dict[Variable, float]
    intercept: float


class SummaryStatistics(NamedTuple):
    """Represents the summary statistics of a distribution."""

    size: float
    mean: float
    std: float
    min: float
    first_quartile: float
    second_quartile: float
    third_quartile: float
    max: float


class MultipleTreatmentsNotImplementedError(NotImplementedError):
    """Raised when multiple treatments aren't yet allowed."""


def get_adjustment_set(
    graph: NxMixedGraph, treatments: Variable | set[Variable], outcome: Variable
) -> Tuple[frozenset[Variable], str]:
    """Get the optimal adjustment set for estimating the direct effect of treatments on a given outcome.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param treatments: The treatment variable(s)
    :param outcome: The outcome variable

    :raises MultipleTreatmentsNotImplementedError: when there are multiple treatments in the input

    :returns: the optimal adjustment set
    """
    treatments = list(_ensure_set(treatments))
    if len(treatments) > 1:
        raise MultipleTreatmentsNotImplementedError

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
) -> RegressionResult:
    """Fit a regression model to the adjustment set over the treatments and a given outcome.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatments: The treatment variable(s)
    :param outcome: The outcome variable

    :raises MultipleTreatmentsNotImplementedError: when there are multiple treatments in the input

    :returns:
        regression result where the regression result contains a dictionary of variables
        to coefficient values and the regression's intercept value
    """
    treatments = _ensure_set(treatments)
    if len(treatments) > 1:
        raise MultipleTreatmentsNotImplementedError
    adjustment_set, _ = get_adjustment_set(graph=graph, treatments=treatments, outcome=outcome)
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
    interventions: Dict[Variable, float] | None = None,
) -> float | list[float]:
    """Estimate treatment effects using Linear Regression.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatments: The treatment variable(s)
    :param outcome: The outcome variable
    :param query_type: The operation to perform
    :param interventions: The interventions for the given query

    :raises TypeError: when the query type is unknown
    :raises ValueError: when the interventions are missing

    :returns:
        the average treatment effect or the outcome probabilities or the expected value
        based on the given query type.
    """
    if query_type == "ate":
        return estimate_ate(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
        )

    elif query_type in {"expected_value", "probability"}:
        if interventions is None:
            raise ValueError(f"interventions must be given for query type: {query_type}")
        y = estimate_probabilities(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcome,
            interventions=interventions,
        )
        if query_type == "probability":
            return y
        return statistics.fmean(y)

    else:
        raise TypeError(f"Unknown query type {query_type}")


def estimate_ate(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
) -> float:
    """Estimate the average treatment effect (ATE) using Linear Regression.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatments: The treatment variable(s)
    :param outcome: The outcome variable

    :raises MultipleTreatmentsNotImplementedError: if multiple treatments are given

    :returns:
        the outcome probabilities
    """
    treatments = _ensure_set(treatments)
    if len(treatments) > 1:
        raise MultipleTreatmentsNotImplementedError
    coefficients, _intercept = fit_regression(graph, data, treatments=treatments, outcome=outcome)
    return coefficients[list(treatments)[0]]


def estimate_probabilities(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    interventions: Dict[Variable, float],
) -> list[float]:
    """Estimate the outcome probabilities using Linear Regression.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatments: The treatment variable(s)
    :param outcome: The outcome variable
    :param interventions: The interventions for the given query

    :raises ValueError: when certain treatments are missing in the interventions

    :returns:
        the outcome probabilities
    """
    treatments = _ensure_set(treatments)
    missing = set(interventions).difference(treatments)
    if missing:
        raise ValueError(f"Missing treatments: {missing}")
    coefficients, intercept = fit_regression(graph, data, treatments=treatments, outcome=outcome)
    y = [
        intercept
        + sum(
            coefficients[variable]
            * (interventions[variable] if variable in treatments else row[variable.name])
            for variable in coefficients
        )
        for row in data.to_dict(orient="records")
    ]
    return y


def estimated_probabilities_summary_statistics(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Variable | set[Variable],
    outcome: Variable,
    interventions: Dict[Variable, float],
) -> SummaryStatistics:
    """Get the summary statistics of the estimated outcome probabilities.

    :param graph: An acyclic directed mixed graph (ADMG)
    :param data: Observational data corresponding to the ADMG
    :param treatments: The treatment variable(s)
    :param outcome: The outcome variable
    :param interventions: The interventions for the given query

    :returns:
        the summary statistics of the estimated outcome probabilities
    """
    y = pd.Series(estimate_probabilities(graph, data, treatments, outcome, interventions))
    summary_stats = y.describe()
    return SummaryStatistics(
        summary_stats["count"],
        summary_stats["mean"],
        summary_stats["std"],
        summary_stats["min"],
        summary_stats["25%"],
        summary_stats["50%"],
        summary_stats["75%"],
        summary_stats["max"],
    )
