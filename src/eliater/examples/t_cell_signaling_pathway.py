"""Examples for T cell signaling pathway."""

from y0.graph import NxMixedGraph

graph = NxMixedGraph.from_str_adj(
    directed={
        "PKA": ["Raf", "Mek", "Erk", "Akt", "Jnk", "P38"],
        "PKC": ["Mek", "Raf", "PKA", "Jnk", "P38"],
        "Raf": ["Mek"],
        "Mek": ["Erk"],
        "Erk": ["Akt"],
        "Plcg": ["PKC", "PIP2", "PIP3"],
        "PIP3": ["PIP2", "Akt"],
        "PIP2": ["PKC"],
    }
)
