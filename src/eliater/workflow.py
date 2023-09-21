from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

from y0.algorithm.falsification import get_conditional_independencies
from y0.dsl import Variable
from y0.graph import NxMixedGraph


def get_state_space_map(
    data: pd.DataFrame, threshold: Optional[int] = 10
) -> Dict[Variable, Literal["discrete", "continuous"]]:
    """Get a dictionary from each variable to its type."""
    unique_count = {column_name: data[column_name].nunique() for column_name in data.columns}
    return {
        Variable(column): "discrete" if unique_count[column] <= threshold else "continuous"
        for column in data.columns
    }


def choose_default_test(data: pd.DataFrame) -> str:
    """Choose the default statistical test for conditional independencies based on the data."""
    col_data_type = get_state_space_map(data=data)
    is_discrete = np.array([col_type == "discrete" for column, col_type in col_data_type.items()])
    is_continuous = ~is_discrete

    if not is_discrete.all() and not is_continuous.all():
        raise NotImplementedError(
            "Mixed data types are not allowed. Either the data should be discrete / continuous"
        )

    if is_continuous.all():
        test = "pearson"
    elif is_discrete.all():
        test = "chi-square"

    return test


def fix_graph(graph: NxMixedGraph, data: pd.DataFrame, test: Optional[str] = None):
    """Repairs the graph by adding undirected edges between variables that fail the conditional independency test"""

    if not test:
        test = choose_default_test(data)

    for conditional_independency in get_conditional_independencies(graph):
        if not conditional_independency.test(
            data, boolean=True, method=test, significance_level=0.05
        ):
            graph.add_undirected_edge(conditional_independency.left, conditional_independency.right)
