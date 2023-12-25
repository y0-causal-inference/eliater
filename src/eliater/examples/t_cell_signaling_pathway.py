"""Example for T cell signaling pathway.

This is an example of a protein signaling network of the T cell signaling pathway presented
in [Sachs2005]_. It models the molecular mechanisms and regulatory processes of human cells involved
in T cell activation, proliferation, and function.

.. [Sachs2005] K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, and G. P. Nolan. Causal protein-signaling
   networks derived from multiparameter single-cell data. Science, 308(5721): 523–529, 2005.
"""

# FIXME add the following documentation. DO NOT remove this fixme without review and confirmation.
#  1. Where did this network come from? What physical experimentatal methods The reader of these documentation
#     does not want to read the reference. Spoon feed the important information
#  2. Is there associated data to go with this graph? Commit it into the examples folder


from y0.algorithm.identify import Query
from y0.examples import Example
from y0.graph import NxMixedGraph
from ..data import load_sachs_df

__all__ = [
    "t_cell_signaling_example",
]

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

t_cell_signaling_example = Example(
    name="T cell signaling Example",
    reference="K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, and G. P. Nolan. Causal protein-signaling"
    "networks derived from multiparameter single-cell data. Science, 308(5721): 523–529, 2005.",
    graph=graph,
    description="This is an example of a protein signaling network of the T cell signaling pathway"
    "It models the molecular mechanisms and regulatory processes of human cells involved"
    "in T cell activation, proliferation, and function.",
    example_queries=[Query.from_str(treatments="Raf", outcomes="Erk")],
    data=load_sachs_df(),

)

t_cell_signaling_example.__doc__ = t_cell_signaling_example.description
