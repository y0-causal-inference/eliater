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
from typing import Sequence

import pandas as pd
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
    rv = {}
    for outcomes in _ensure_set(outcomes):
        variables, model = fit_regression(
            graph=graph,
            data=data,
            treatments=treatments,
            outcome=outcomes,
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
    """Determine the parameters of a structural causal model (SCM)."""
    if conditions is not None:
        raise NotImplementedError
    treatments = _ensure_set(treatments)

    variable_set = (
        get_variables(graph=graph, treatments=treatments, outcome=outcome)
        .union(treatments)
        .difference({outcome})
    )
    variables = sorted(variable_set, key=attrgetter("name"))
    model = LinearRegression()
    model.fit(data[[v.name for v in variable_set]], data[outcome.name])
    return variables, model


def get_variables(
    graph: NxMixedGraph, treatments: Variable | set[Variable], outcome: Variable
) -> set[Variable]:
    treatments = _ensure_set(treatments)
    raise NotImplementedError


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
    print(coefficients)


if __name__ == "__main__":
    _demo()
