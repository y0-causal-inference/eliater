import warnings
from typing import Dict, Literal, Optional, Set, Union

import networkx as nx
import pandas as pd
from y0.algorithm.falsification import get_conditional_independencies
from y0.dsl import Variable
from y0.graph import NxMixedGraph, set_latent
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


def find_all_nodes_in_causal_paths(
    graph: NxMixedGraph, treatments: Union[Variable, Set[Variable]], outcome: Variable
) -> Set[Variable]:
    """Find all the nodes in causal paths from source to destination."""
    if isinstance(treatments, Variable):
        treatments = {treatments}
    nodes = set()
    for treatment in treatments:
        for causal_path in nx.all_simple_paths(graph.directed, treatment, outcome):
            for node in causal_path:
                nodes.add(node)
    return nodes


def mark_latent(
    graph: NxMixedGraph, treatments: Union[Variable, Set[Variable]], outcome: Variable
) -> NxMixedGraph:
    """Marks latent nodes.

    Marks the descendants of nodes in all causal paths that are not ancestors of the outcome variable as latent
    nodes.
    """
    if isinstance(treatments, Variable):
        treatments = {treatments}
    # Find the nodes on the causal path
    nodes_on_causal_paths = find_all_nodes_in_causal_paths(
        graph=graph, treatments=treatments, outcome=outcome
    )
    # Find the descendants for the nodes on the causal paths
    descendants_of_nodes_on_causal_paths = graph.descendants_inclusive(nodes_on_causal_paths)
    # Find the ancestors of the outcome variable
    ancestors_of_outcome = graph.ancestors_inclusive(outcome)
    # Descendants of nodes on the causal paths that are not ancestors of the outcome variable
    descendants_not_ancestors = descendants_of_nodes_on_causal_paths.difference(
        ancestors_of_outcome
    )
    # Remove treatments and outcome
    descendants_not_ancestors = descendants_not_ancestors.difference(treatments.union({outcome}))
    # Mark nodes as latent
    set_latent(graph.directed, descendants_not_ancestors)
    return graph
