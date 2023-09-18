from typing import Optional, Dict, Literal

import pandas as pd
from y0.algorithm.falsification import get_conditional_independencies
from y0.graph import NxMixedGraph


def fix_graph(graph: NxMixedGraph, data: pd.DataFrame, test: Optional[str]):
    """Repairs the graph by adding undirected edges between variables that fail the conditional independency test"""
    for conditional_independency in get_conditional_independencies(graph):
        if not(conditional_independency.test(data, boolean=True, method=test, significance_level=0.05)):
            graph.add_undirected_edge(conditional_independency.left, conditional_independency.right)
