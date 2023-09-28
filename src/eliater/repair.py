"""This module defines the steps for repairing the network structure.

The prior knowledge of the network may contain errors, missing edges
and omitted variables. It is crucial to repair the structure of the
prior network, by examining the conditional independencies implied
by the network with the ones implied by the data.

If a conditional independency implied by the network is not supported by the data,
it fails. This failure suggests that there is probably a latent
confounder between the variables for which the test has failed.
In this case, we suggest putting a bi-directed edge between those
variables to indicate the presence of latent confounder.

The inputs are the prior knowledge of the graph in the form of an ADMG, and the observational
and/or experimental data. The output is the repaired ADMG.

Here is an example:

#import a prior knowledge network X -> M1 -> M2 -> Y, X <-> Y, where X is the treatment,
#Y is the outcome. There is a bi-directed edge between X and Y.

from eliater.examples import multi_med

#Generate data for this prior network with the assumption that there is a  bi-directed
#edge between M1 and Y that is missed from the prior knowledge network. The goal is to
#recover that edge.

from eliater.examples.multi_med import generate_data_for_multi_med

repaired_graph = fix_graph(multi_med, generate_data_for_multi_med(100))

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
