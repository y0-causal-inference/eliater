r"""
The goal is to estimate causal effects using regression on the exposure (treatment) variable.

In this module we want to estimate the causal effect of a hypothesized treatment or intervention
of the exposure variable ($X$) on the outcome variable ($Y$) using linear regression. The causal effect
types that this module support is in the following forms:

1. Probability distribution over the outcome variable given an intervention on the exposure $P(Y \mid do(X=x)$
   where $X$ can take discrete or continuous values.
2. Expected value of the outcome given an intervention on the exposure $\mathbb{E}[Y \mid do(X=x)]$, where $X$ can take
   discrete or continuous values.
3. Average treatment effect (ATE), which is defined as $\mathbb{E}[Y \mid do(X=x+1)] - \mathbb{E}[Y \mid do(X=x)]$
   where $X$ can take discrete or continuous values. In the case of a binary exposure, where X only takes 1 (meaning
   that the treatment has been received) or 0 (meaning that treatment has not been received), the ATE is defined as
   $\mathbb{E}[Y \mid do(X=1)] - \mathbb{E}[Y \mid do(X=0)]$. In this module, we support both continuous and discrete
   values of $X$. In general the ATE varies depending on levels of $x$, but in linear models, the ATE reduces to a
   single number. Hence, it does not depend on the value of $x$. This is shown below.

In order to have an intuition for how to use linear regression on the treatment variable, we can create a
Gaussian linear structural causal model (SCM). With Gaussian linear SCMs, each variable is defined as a
linear combination of its parents. For example, consider this graph:

.. code-block:: python

    from y0.dsl import X, Y, Z
    from y0.graph import NxMixedGraph

    graph = NxMixedGraph.from_edges([
        (X, Y),
        (Z, Y),
        (Z, X),
    ])
    graph.draw()

.. figure:: img/backdoor.png
   :scale: 70%

The goal is to find the causal effect of X on Y. In this graph a Gaussian linear SCM can be defined as below:

$Z = U_Z; U_Z \sim \mathcal{N}(0, \sigma^2_Z)$

$X = \lambda_{zx} Z + U_X; U_X \sim \mathcal{N}(0, \sigma^2_X)$

$Y = \lambda_{xy} X + \lambda_{zy} Z + U_Y; U_Y \sim \mathcal{N}(0, \sigma^2_Y)$

Hence the probability distribution over the outcome variable given an intervention on the exposure
$P(Y \mid do(X=x))$ involves substituting estimated values of $\lambda_{xy}$ and $\lambda_{zy}$, fixing
the value of $X$ to $x$, generating samples from $U_Y$, and finally generating samples for $Y$.
Averaging across different values of sampled $Y$
values will lead to an estimate for the query in the form of $\mathbb{E}[Y \mid do(X=x)]$. The ATE amounts to,

$ATE = E[Y \mid do(X=x+1) - E[Y \mid do(X=x)] = \lambda_{xy}$.

However, if one naively regress X on Y, then the regression coefficient of $Y$ on $X$, denoted by $\gamma_{yx}$
is computed as follows:

$\gamma_{yx} = \frac{Cov(Y,X)}{Var(X)} = \lambda_{xy} + \lambda_{zx} \lambda_{zy}$

the estimated $\gamma_{yx} = \lambda_{xy} + \lambda_{zx} \lambda_{zy}$ differs from the true value of
ATE which amounts to $\lambda_{xy}$. Hence, the estimation of ATE is biased. This discrepancy arises because the
observed association of $X$ and $Y$ encapsulates both the causal relationship, indicated by the path X → Y, and the
non-causal relationship due
to the confounder $Z$, indicated by the path X ← Z → Y . Such confounding paths initiating with an arrow directed
towards $X$ are termed *back-door paths*.
Nevertheless, it's noteworthy that the regression coefficient of $Y$ on $X$ when adjusted for $Z$
(denoted by $\gamma_{yx.z}$) simplifies to:

$\gamma_{yx.z} = \lambda_{xy}$

This means that adjusting for $Z$ blocks the back-door path, and allows us to directly estimate the effect of $X$
on $Y$, which leads to an unbiased estimate of the ATE.
The set of variables blocking the backdoor paths are called adjustment sets.

The module identifies an adjustment set with the following priority:

1. An optimal adjustment set - the adjustment set that leads to an estimate of ATE with least
   asymptotic variance, if it exist
2. An optimal minimal adjustment set - identify all possible adjustment sets and choose one with the minimum cardinality
   (i.e., number of elements) that results in the least asymptomatic variance in the estimation of the ATE
   (choosing the one resulting in the ATE still needs to be implemented).
3. A randomly chosen adjustment set

Once the adjustment set is selected, this module use it to perform a regression with $X$ and the variables in the
adjustment set as the inputs and $Y$ as the output to find an unbiased
estimate of the $P(Y \mid do(X=x))$, $\mathbb{E}[Y \mid do(X=x)]$, and the ATE.

Linear regression is known for its simplicity, speed, and high interpretability.
However, linear regression is most appropriate when the variables exhibit linear relationships. In addition, it only
uses the variables from the back-door adjustment set and does not utilize useful variables such as mediators.

Example
-------
We'll work with the example :data:`eliater.frontdoor_backdoor.example_2` in which $X$ is the treatment and $Y$ is
the outcome. We will explore estimating causal effects in the forms of $P(Y \mid do(X=x))$,
$\mathbb{E}[Y \mid do(X=x)]$, and ATE.

.. code-block:: python

    from eliater.examples import example_2

    example_2.draw()

.. figure:: img/regression_example.png
   :scale: 70%

Typically, estimation of ATE is more popular and desirable than estimation of $\mathbb{E}[Y \mid do(X=x)]$, because ATE
compares the difference in the expected value of the outcome when the person has and has not received the treatment.
This is particularly useful when one is interested in the effect that receiving a medicine (treatment) has on a specific
disease. However, if there is not enough range of observational data for $X$, estimation of ATE will become inprecise.
In such cases, estimation of $\mathbb{E}[Y \mid do(X=x)]$ is more accurate and provides details of the expected value
of the outcome when the treatment is intervened upon. Estimation of $P(Y \mid do(X=x))$ is useful when one is interested
in an overall distribution over the outcome, and wants to get summary statistics (mean, median, quantiles, min, max) of
the output.


.. code-block:: python

    from y0.dsl import Variable, X, Y
    from eliater.frontdoor_backdoor import example_2
    from eliater.regression import estimate_query

    graph = example_2.graph
    data = example_2.generate_data(100, seed=100)
    estimate_query(
        graph=graph,
        data=data,
        treatments={X},
        outcome=Y,
        query_type="expected_value",
        interventions={X: 1},
    )

The output of the query type in the form of "expected value" ($\mathbb{E}[Y \mid do(X=1)]$) is 69.78.
This means that if one intervene on $X$ and fix its value of 1, the expected value over $Y$ will be around 70
However, if one estimates the expected value of $Y$ without any intervention on $X$, it will amount to 73.61.

.. code-block:: python

    from eliater.frontdoor_backdoor import example_2

    data = example_2.generate_data(100, seed=100)
    data.mean()['Y']

This means that, intervention on $X$ and fixing its value to 0 causes a decrease in the expected value of $Y$. Now
we will explore the output of the query in the form of ATE:
$\mathbb{E}[Y \mid do(X=x+1)] - \mathbb{E}[Y \mid do(X=x+1)]$. Note that the value of $x$ is indifferent in the
output, because ATE amounts to the estimate for the coefficient of $X$ (\hat{\lambda_{xy}}), in a regression equation
where $X$, and the optimal adjustment set $Z_3$ are regressed on $Y$ as follows:

$Y = \lambda_{xy} X + \lambda_{z_3Y} Z_3 + U_Y; U_Y \sim \mathcal{N}(0, \sigma^2_Y)$

Note that, instead of $Z_3$ in the equation above, one could have used $Z_1$ or $Z_2$, or any combination of $Z_i$ for
$i \in \{1,2,3\}$. However, this module is designed such that if the optimal adjustment set exists, it picks it. Hence,
in this example $Z_3$ is selected and used.

.. code-block:: python

    from y0.dsl import Variable, X, Y
    from eliater.frontdoor_backdoor import example_2
    from eliater.regression import estimate_query

    graph = example_2.graph
    data = example_2.generate_data(100, seed=100)

    estimate_query(
        graph=graph,
        data=data,
        treatments={X},
        outcome=Y,
        query_type="ate"
    )

The estimated value for ATE (average contrast of Y under two distinct interventions
on $X$) amounts to 0.123. Finally, we estimate the query in the form of "probability".

.. code-block:: python

    from y0.dsl import Variable, Z, X, Y
    from eliater.frontdoor_backdoor import example_2
    from eliater.regression import estimate_query

    graph = example_2.graph
    data = example_2.generate_data(100, seed=100)

    estimate_query(
        graph=graph,
        data=data,
        treatments={X},
        outcome=Y,
        query_type="probability",
        interventions={X: 0},
    )

The output is a vector of $Y$ values (100 data points). The mean over these values is equal to the
expected value obtained above when the query was in the form of "expected value". This type of query
can be used to get summary statistics for the distribution of $Y$, when $X$ has been intervened upon.
For example, we can get a summary statistics over $Y$ as follows:

.. code-block:: python

    from y0.dsl import Variable, Z, X, Y
    from eliater.frontdoor_backdoor import example_2
    from eliater.regression import estimate_query

    graph = example_2.graph
    data = example_2.generate_data(100, seed=100)
    summary_statistics(
        graph=graph,
        data=data,
        treatments={X},
        outcome=Y,
        interventions={X: 0},
    )

The output is as follows:

'''SummaryStatistics(size=100.0, mean=69.65468808666209, std=7.134828083971124, min=51.47221859593694,
first_quartile=64.48297054774694, second_quartile=69.74351236828335, third_quartile=74.41515094449318,
max=91.8646554122394)'''

Unanswered Questions for Later
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. What's the difference between estimation with this module and what's available in Ananke?
2. What are the limitations of estimation with this methodology?
3. Where do all of these random numbers throughout the examples of using the code come from? Can we have those
   propagated/calculated automatically? I don't trust any numbers that were typed by hand.
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
    "summary_statistics",
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


def summary_statistics(
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
