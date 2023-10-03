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

This process allows for unbiased estimation of causal queries in cases where the
overall ADMG structure over the observed variables is correct, but the number and location of
latent variables is unknown.

Here is an example of an ADMG with its corresponding observational data. This ADMG has one
bi-directed edge between X and Y. The observational data is generated with the assumption
that there exists an additional bidirectional edge between M2 and Y, which is not depicted
in the given ADMG. Our objective is to identify all the conditional independencies implied
by this ADMG and pinpoint any inconsistencies with the data. The ultimate aim is to detect
this overlooked bidirectional edge and incorporate it into the corrected ADMG.

.. code-block:: python
    from y0.graph import NxMixedGraph
    graph = NxMixedGraph.from_edges(
        directed=[
            ('X', 'M1'),
            ('M1', 'M2'),
            ('M2', 'Y'),
        ],
        undirected=[
            ('X', 'Y'),
        ],
    )

    # Generate observational data for this graph (this is a special example)
    from frontdoor_backdoor.multiple_mediators_single_confounder import generate
    observational_data = generate(100)

    from eliater.repair import fix_graph
    repaired_graph = fix_graph(graph, observational_data)

"""

import warnings
from typing import Dict, Literal, Optional

import pandas as pd

from y0.algorithm.falsification import get_conditional_independencies
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import get_conditional_independence_tests

__all__ = [
    "repair_network",
]


def get_state_space_map(
    data: pd.DataFrame, threshold: Optional[int] = 10
) -> Dict[Variable, Literal["discrete", "continuous"]]:
    """Get a dictionary from each variable to its type."""
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


def repair_network(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    test: Optional[CITest] = None,
    significance_level: Optional[float] = 0.05,
) -> NxMixedGraph:
    """Repairs the network by adding undirected edges between the nodes that fail the conditional independency test.

    :param graph: an NxMixedGraph
    :param data: data
    :param test: the conditional independency test to use.
        Supported values are ["pearson", "chi-square", "cressie_read",
        "freeman_tuckey", "g_sq", "log_likelihood", "modified_log_likelihood",
        "power_divergence", "neyman"]
    :param significance_level: significance level. Default is 0.05
    :returns: The repaired network, in place
    :raises ValueError: if the passed test is invalid / unsupported
    :raises Exception: if the data and discrete and the chosen test is pearson
    """
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
