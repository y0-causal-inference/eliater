"""Implementation of Eliater workflow."""

from typing import Optional, Union

import pandas as pd

from y0.algorithm.estimation import estimate_ace
from y0.algorithm.identify import identify_outcomes
from y0.dsl import Expression, Variable
from y0.graph import NxMixedGraph, _ensure_set
from y0.struct import CITest

from .discover_latent_nodes import remove_nuisance_variables
from .network_validation import add_ci_undirected_edges

__all__ = [
    "workflow",
]


def workflow(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Union[Variable, set[Variable]],
    outcomes: Union[Variable, set[Variable]],
    *,
    ci_method: Optional[CITest] = None,
    ci_significance_level: Optional[float] = None,
    ace_bootstraps: int | None = None,
    ace_significance_level: float | None = None,
) -> tuple[NxMixedGraph, Expression, float]:
    """Run the Eliater workflow.

    This workflow has two parts:

    1. Add undirected edges between d-separated nodes for which a data-driven conditional independency test fails
    2. Remove nuissance variables.
    3. Estimates the average causal effect (ACE) of the treatments on outcomes

    :param graph: An acyclic directed mixed graph
    :param data: Data associated with nodes in the graph
    :param treatments: The node or nodes that are treated
    :param outcomes: The node or nodes that are outcomes
    :param ci_method:
        The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data
        or :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param ci_significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :param ace_bootstraps: The number of bootstraps for calculating the ACE. Defaults to 0 (i.e., not used by default)
    :param ace_significance_level: The significance level for the ACE. Defaults to 0.05.
    :returns: A triple with a modified graph, the estimand, and the ACE value.
    :raises ValueError: If the graph becomes unidentifiable throughout the workflow
    """
    graph = add_ci_undirected_edges(
        graph, data, method=ci_method, significance_level=ci_significance_level
    )
    treatments = _ensure_set(treatments)
    outcomes = _ensure_set(outcomes)
    estimand = identify_outcomes(graph, treatments=treatments, outcomes=outcomes)
    if estimand is None:
        raise ValueError("not identifiable after adding CI edges")

    # TODO extend this to consider condition variables
    graph = remove_nuisance_variables(graph, treatments=treatments, outcomes=outcomes)
    estimand = identify_outcomes(graph, treatments=treatments, outcomes=outcomes)
    if not estimand:
        raise ValueError("not identifiable after removing nuisance variables")

    ace = estimate_ace(
        graph=graph,
        treatments=list(treatments),
        outcomes=list(outcomes),
        data=data,
        bootstraps=ace_bootstraps,
        alpha=ace_significance_level,
    )
    return graph, estimand, ace
