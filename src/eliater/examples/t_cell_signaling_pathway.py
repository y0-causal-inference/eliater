"""Example for T cell signaling pathway.

This is an example of a protein signalling network of the T cell signaling pathway presented
in [Sachs2005]_. It models the molecular mechanisms and regulatory processes of human cells involved
in T cell activation, proliferation, and function.

Here is the data associated with this network:

.. code-block:: python

    data = pd.read_csv(
    "../eliater/src/eliater/data/sachs_discretized_2bin.csv",
    index_col=False,
    )

[Sachs2005] K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, and G. P. Nolan. Causal protein-signaling
networks derived from multiparameter single-cell data. Science, 308(5721): 523–529, 2005.
"""

from y0.algorithm.identify import Query
from y0.examples import Example
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

base_example = Example(
    name="T cell signaling Example",
    reference="K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, and G. P. Nolan. Causal protein-signaling"
    "networks derived from multiparameter single-cell data. Science, 308(5721): 523–529, 2005.",
    graph=graph,
    description="This is an example of a protein signalling network of the T cell signaling pathway"
    "It models the molecular mechanisms and regulatory processes of human cells involved"
    "in T cell activation, proliferation, and function.",
    example_queries=[Query.from_str(treatments="Raf", outcomes="Erk")],
)

base_example.__doc__ = base_example.description
