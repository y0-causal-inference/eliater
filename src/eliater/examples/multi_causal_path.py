"""A simple network with multiple causal paths."""

from y0.graph import NxMixedGraph

graph = NxMixedGraph.from_str_adj(
    directed={
        "X1": ["X3", "A", "E", "F"],
        "X3": ["A"],
        "A": ["B", "C", "G"],
        "B": ["Y"],
        "X2": ["A"],
        "C": ["D"],
        "D": ["Y"],
        "Y": ["H"],
    }
)
