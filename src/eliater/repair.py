"""This module defines the steps for repairing the network structure.

Given an acyclic directed mixed graph (ADMG) and corresponding observational data,
one can assess whether the conditional independences implied by the structure of the
ADMG are supported by the data with a statistical test for conditional independence.
By default, this workflow uses a chi-square test for discrete data and a Pearson test
for continuous data from :mod:`pgmpy.estimators.CITests`.

Any conditional independency implied by the ADMG that fails to reject the null hypothesis
of the statistical test suggests the presence of a latent confounder between the two
variables for which the test failed. In such cases, this workflow adds a bidirectional
edge between the affected variables.

This process allows for unbiased estimation of causal queries in cases where the overall ADMG
structure over the observed variables is correct, but the number and location of
latent variables is unknown.

Here is an example of an ADMG with its corresponding observational data. This ADMG has one
bi-directed edge between X and Y. The observational data is generated with the assumption
that there exists an additional bidirectional edge between M2 and Y, which is not depicted
in the given ADMG. The incorporation of this additional edge is for testing purposes. The
goal of this example is to detect this missed bidirectional edge by testing the conditional
independences implied by the network against the data. Once the missed edge is detected it
will be incorporated into the ADMG. Hence the output is the rapired ADMG containing the
bidirected edge ('M2', 'Y').

.. todo::

    Test that the code block below actually works (it doesn't)
    by pasting it into a Jupyter notebook or python REPL. Fix the issues such
    as broken imports.

.. code-block:: python

    from y0.graph import NxMixedGraph
    from y0.dsl import Variable, X, Y
    M1 = Variable("M1")
    M2 = Variable("M2")
    from frontdoor_backdoor.multiple_mediators_single_confounder import generate
    from eliater.repair import repair_network

    graph = NxMixedGraph.from_edges(
        directed=[
            (X, M1),
            (M1, M2),
            (M2, Y),
        ],
        undirected=[
            (X, Y),
        ],
    )

    # Generate observational data for this graph (this is a special example)
    observational_data = generate(100)

    repaired_graph = add_conditional_dependency_edges(graph, observational_data)

.. todo::

    Sara said on Slack:

    > This is also explained in Robert's book and it is not a step that one should
    > highly rely on for several reasons. First, these tests make distributional
    > assumptions about the data generation process. For example assume they follow
    > the X-square distribution where in reality it may not be the case, or for
    > continuous data make the assumption that the data is from Gaussian distribution.

    Let's see some examples where this methodology doesn't work that also includes
    documentation on what a user should do in this situation.
    DO NOT DELETE THIS TO-DO until several end-to-end runnable examples are given below

This module relies on statistical tests, and statistical tests always have chances
of producing false negatives, i.e., a pair of variables that are conditionally
independent, be concluded as conditional dependent by the test, or producing false
positives, i.e., a pair of variables that are conditionally dependent be concluded
as conditionally independent by the test. Hence, the results obtained from this module
should be regarded more as heuristics approach rather than a systematic, strict step that
provides precise results.

Here are some reasons that the result of the test may be false negative or false positive:

1) In pgmpy, the conditional independence tests assume that the
alternative hypothesis is dependence, while the null hypothesis is conditional
independence. However, when dealing with an ADMG and hypothetically assuming that
the ADMG has the correct structure, it is more appropriate for the null hypothesis
to be the hypothesis of dependence. This distinction can be significant as the p-value
relies on the null hypothesis.

It's worth noting that this module employs traditional tests where the null hypothesis is
conditional independence.

2) In addition, p-values decrease as the number of data points used in the conditional
independency test increases, i.e., the larger the data, more conditional independences
implied by the network will be considered as dependent. Hence, chances of false negatives
increases. Here is an example that illustrates this:

.. code-block:: python

    from y0.graph import NxMixedGraph
    from y0.dsl import Variable, X, Y
    M1 = Variable("M1")
    M2 = Variable("M2")
    from frontdoor_backdoor.multiple_mediators_single_confounder import generate

    graph = NxMixedGraph.from_edges(
        directed=[
            (X, M1),
            (M1, M2),
            (M2, Y),
        ],
        undirected=[
            (X, Y),
        ],
    )

    # Generate observational data for this graph (this is a special example)
    observational_data = generate(100)

    data_size = range(50, 10000, 100)
    p_vals, lower_errors, higher_errors, probs_conclude_indep = zip(
        *[
            estimate_p_val(
                full_data=full_data,
                sample_size=size,
                left="M1",
                right="Y",
                conditions=["M2", "X],
                test="pearson",
                significance_level=0.05,
                boot_size=1000,
            )
            for size in data_size
        ]
    )

    plt.title("Independece of M1 & Y given M2 & X")
    plt.xlabel("number of data points")
    plt.ylabel("expected p-value")
    plt.errorbar(
    data_size, p_vals, yerr=np.array([lower_errors, higher_errors]), ecolor="grey", elinewidth=0.5
    )
    plt.hlines(0.05, 0, 10000, linestyles="dashed")
    plt.show()


3) Conditional independence tests rely on probability assumptions regarding the data distribution.
For instance, when dealing with discrete data, employing the chi-square test generates a test statistic
that conforms to the Chi-squared probability distribution. Similarly, in the case of continuous data,
utilizing the Pearson test yields a test statistic that adheres to the Normal distribution. If these
assumptions are not satisfied by the data, the outcomes may lead to both false positives and false negatives.

"""

import warnings
from typing import Dict, Literal, Optional

import pandas as pd

from y0.algorithm.falsification import get_conditional_independencies
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import get_conditional_independence_tests

__all__ = [
    "add_conditional_dependency_edges",
    "get_state_space_map",
    "is_data_discrete",
    "is_data_continuous",
    "CITest",
    "choose_default_test"
]



def get_state_space_map(
    data: pd.DataFrame, threshold: Optional[int] = 10
) -> Dict[Variable, Literal["discrete", "continuous"]]:
    """Get a dictionary from each variable to its type.

    .. todo:: document parameters

    .. todo:: This thresholding doesn't make sense. what if the values are 0, 1, 2?
    """
    column_values_unique_count = {
        column_name: data[column_name].nunique() for column_name in data.columns
    }
    return {
        Variable(column): "discrete"
        if column_values_unique_count[column] <= threshold
        else "continuous"
        for column in data.columns
    }


def is_data_discrete(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has discrete data."""
    variable_types = set(get_state_space_map(data=data).values())
    return variable_types == {"discrete"}


def is_data_continuous(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has continuous data."""
    variable_types = set(get_state_space_map(data).values())
    return variable_types == {"continuous"}


CITest = Literal[
    "pearson",
    "chi-square",
    "cressie_read",
    "freeman_tuckey",
    "g_sq",
    "log_likelihood",
    "modified_log_likelihood",
    "power_divergence",
    "neyman",
]


def choose_default_test(data: pd.DataFrame) -> CITest:
    """Choose the default statistical test for testing conditional independencies based on the data."""
    if is_data_discrete(data):
        return "chi-square"
    if is_data_continuous(data):
        return "pearson"
    raise NotImplementedError(
        "Mixed data types are not allowed. Either all of the columns of data should be discrete / continuous."
    )


def add_conditional_dependency_edges(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    test: Optional[CITest] = None,
    significance_level: Optional[float] = None,
) -> NxMixedGraph:
    """Repairs the network structure.

    Repairs the network structure by introducing bidirectional edges between
    any pairs of variables when the conditional independence implied by the network
    is not supported by the data through a statistical conditional independence test.

    :param graph: an NxMixedGraph
    :param data: observational data corresponding to the graph
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :returns: The repaired network, in place
    :raises ValueError: if the passed test is invalid / unsupported
    :raises Exception: if the data is discrete and the chosen test is pearson

    .. todo:: Use a real exception and not the base Exception. ValueError is fine.

    .. todo:: Why is it warning when there is continuous data and pearson test is used? Raise an error.
    """
    if significance_level is None:
        significance_level = 0.01
    if not test:
        test = choose_default_test(data)

    tests = get_conditional_independence_tests()
    if test not in tests:
        raise ValueError(f"`{test}` is invalid. Supported CI tests are: {sorted(tests)}")

    if is_data_continuous(data) and test != "pearson":
        warnings.warn(
            message="The data is continuous. Either discretize and use chi-square or use the pearson.",
            stacklevel=2,
        )

    if is_data_discrete(data) and test == "pearson":
        raise Exception("Cannot run pearson on discrete data. Use chi-square instead.")

    for conditional_independency in get_conditional_independencies(graph):
        if not conditional_independency.test(
            data, boolean=True, method=test, significance_level=significance_level
        ):
            graph.add_undirected_edge(conditional_independency.left, conditional_independency.right)

    return graph

from eliater.