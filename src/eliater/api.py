"""Implementation of Eliater workflow."""

from typing import Optional, Union

import pandas as pd

from y0.dsl import Variable
from y0.graph import NxMixedGraph, _ensure_set
from y0.struct import CITest
from .network_validation import add_ci_undirected_edges
from .discover_latent_nodes import remove_nuisance_variables
from y0.algorithm.identify import identify_outcomes

__all__ = [
    "workflow",
]


def workflow(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Union[Variable, set[Variable]],
    outcomes: Union[Variable, set[Variable]],
    # TODO what about conditions?
    *,
    ci_method: Optional[CITest] = None,
    ci_significance_level: Optional[float] = None,
):
    graph = add_ci_undirected_edges(
        graph, data, method=ci_method, significance_level=ci_significance_level
    )
    treatments = _ensure_set(treatments)
    outcomes = _ensure_set(outcomes)
    estimand = identify_outcomes(graph, treatments=treatments, outcomes=outcomes)
    if estimand is None:
        raise ValueError("not identifiable")

    graph = remove_nuisance_variables(graph, treatments=treatments, outcomes=outcomes)
    estimand = identify_outcomes(graph, treatments=treatments, outcomes=outcomes)
    return estimand
