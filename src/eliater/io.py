"""Converters from y0 to external packages."""

from typing import TYPE_CHECKING

from y0.graph import NxMixedGraph

if TYPE_CHECKING:
    import optimaladj.CausalGraph

__all__ = [
    "to_causal_graph",
]


def to_causal_graph(graph: NxMixedGraph) -> "optimaladj.CausalGraph.CausalGraph":
    """Convert a mixed graph to an equivalent :class:`optimaladj.CausalGraph.CausalGraph`."""
    from optimaladj.CausalGraph import CausalGraph

    causal_graph = CausalGraph()
    ananke_admg = graph.to_admg()
    causal_graph.add_edges_from(ananke_admg.di_edges)
    for i, (node1, node2) in enumerate(ananke_admg.bi_edges, start=1):
        latent = "U_{}".format(i)
        causal_graph.add_edge(latent, node1)
        causal_graph.add_edge(latent, node2)
    return causal_graph
