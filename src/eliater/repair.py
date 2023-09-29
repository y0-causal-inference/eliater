"""This module defines the steps for repairing the network structure.

"Given an Acyclic Directed Mixed Graph (ADMG) and the corresponding data,
you can assess the correctness of the graph by checking whether the conditional
independencies implied by the graph are supported by the data. This can be
performed by using a statistical conditional independence test. For all the
conditional independencies implied by the graph, if any of them, such as X
being independent of Y given Z, is not supported by the data, then the network
structure should be repaired. In the case of a failed test, the network likely
misses confounders between the affected nodes (e.g., X and Y). This module adds
bidirectional edges between the affected nodes to indicate the presence of confounders
and outputs a repaired ADMG."

Here is an example:

.. code-block:: python

    #Get the input ADMG
    from eliater.examples import multi_mediators

    #Get the data associated with the input ADMG
    from eliater.examples.multi_med import generate_data_for_multi_mediators

    repaired_graph = fix_graph(multi_mediators, generate_data_for_multi_mediators(100))

"""

import warnings
from typing import Dict, Literal, Optional

import pandas as pd

from y0.algorithm.falsification import get_conditional_independencies
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import get_conditional_independence_tests


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


def choose_default_test(data: pd.DataFrame) -> str:
    """Choose the default statistical test for testing conditional independencies based on the data."""
    if is_data_discrete(data):
        return "chi-square"
    if is_data_continuous(data):
        return "pearson"
    raise NotImplementedError(
        "Mixed data types are not allowed. Either all of the columns of data should be discrete / continuous."
    )


def fix_graph(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    test: Optional[str] = None,
    significance_level: Optional[float] = 0.05,
) -> NxMixedGraph:
    """Repairs the graph by adding undirected edges between the nodes that fail the conditional independency test."""
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

