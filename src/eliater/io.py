"""Converters from y0 to external packages."""

from typing import TYPE_CHECKING

from y0.graph import NxMixedGraph

if TYPE_CHECKING:
    import optimaladj.CausalGraph
    import pgmpy.models

__all__ = [
    "to_causal_graph",
    "to_bayesian_network",
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


def to_bayesian_network(graph: NxMixedGraph) -> "pgmpy.models.BayesianNetwork":
    """Convert a mixed graph to an equivalent :class:`pgmpy.BayesianNetwork`."""
    from pgmpy.models import BayesianNetwork

    ananke_admg = graph.to_admg()
    ananke_dag = ananke_admg.canonical_dag()
    di_edges = ananke_dag.di_edges
    bi_edges = ananke_admg.bi_edges
    # TODO test this carefully
    latents = ["U_{}_{}".format(*sorted([node1, node2])) for node1, node2 in bi_edges]
    model = BayesianNetwork(ebunch=di_edges, latents=latents)
    return model
