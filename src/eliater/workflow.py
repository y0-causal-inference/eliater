import warnings
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd
from y0.algorithm.falsification import get_conditional_independencies
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import get_conditional_independence_tests


def get_state_space_map(
    data: pd.DataFrame, threshold: Optional[int] = 10
) -> Dict[Variable, Literal["discrete", "continuous"]]:
    """Get a dictionary from each variable to its type."""
    unique_count = {column_name: data[column_name].nunique() for column_name in data.columns}
    return {
        Variable(column): "discrete" if unique_count[column] <= threshold else "continuous"
        for column in data.columns
    }


def is_data_discrete(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has discrete data."""
    variable_types = get_state_space_map(data=data)
    is_discrete = np.array([col_type == "discrete" for column, col_type in variable_types.items()])
    return is_discrete.all()


def is_data_continuous(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has continuous data."""
    variable_types = get_state_space_map(data=data)
    is_continuous = np.array(
        [col_type == "continuous" for column, col_type in variable_types.items()]
    )
    return is_continuous.all()


def choose_default_test(data: pd.DataFrame) -> str:
    """Choose the default statistical test for testing conditional independencies based on the data."""
    is_discrete = is_data_discrete(data)
    is_continuous = is_data_continuous(data)

    if not is_discrete and not is_continuous:
        raise NotImplementedError(
            "Mixed data types are not allowed. Either all of the columns of data should be discrete / continuous."
        )

    if is_continuous:
        test = "pearson"
    else:
        test = "chi-square"

    return test


def fix_graph(graph: NxMixedGraph, data: pd.DataFrame, test: Optional[str] = None) -> NxMixedGraph:
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
            data, boolean=True, method=test, significance_level=0.001
        ):
            graph.add_undirected_edge(conditional_independency.left, conditional_independency.right)

    return graph
