"""Examples for SARS-CoV-2 and COVID19.

.. todo::

    1. Isn't this the same as something built in to the y0 package?
    2. wrap these with the :class:`y0.example.Example` class and detailed biological context.
    3. Where did this network come from? Give reference
    4. What is the biological phenomena described here?
    5. How was this network constructed?
    6. Is there associated data to go with this graph?
"""

from y0.graph import NxMixedGraph

graph = NxMixedGraph.from_str_edges(
    directed=[
        ("SARS_COV2", "ACE2"),
        ("ACE2", "Ang"),
        ("Ang", "AGTR1"),
        ("AGTR1", "ADAM17"),
        ("ADAM17", "EGF"),
        ("ADAM17", "TNF"),
        ("ADAM17", "Sil6r"),
        ("SARS_COV2", "PRR"),
        ("PRR", "NFKB"),
        ("EGFR", "NFKB"),
        ("TNF", "NFKB"),
        ("Sil6r", "IL6STAT3"),
        ("Toci", "Sil6r"),
        ("NFKB", "IL6AMP"),
        ("IL6AMP", "cytok"),
        ("IL6STAT3", "IL6AMP"),
        ("EGF", "EGFR"),
        ("Gefi", "EGFR"),
    ],
    undirected=[
        ("SARS_COV2", "Ang"),
        ("ADAM17", "Sil6r"),
        ("PRR", "NFKB"),
        ("EGF", "EGFR"),
        ("EGFR", "TNF"),
        ("EGFR", "IL6STAT3"),
    ],
)
