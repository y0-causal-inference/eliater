"""Examples for T cell signaling pathway.

.. todo::

    1. wrap these with the :class:`y0.example.Example` class and detailed biological context.
    2. Where did this network come from? Give reference
    3. In what organism is this happening?
    4. What is the biological phenomena described here?
    5. How was this network constructed?
    6. Is there associated data to go with this graph?
"""

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
