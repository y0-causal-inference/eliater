"""Examples for SARS-CoV-2.

The data associated with this example is as follows:

.. code-block:: python

    from src.eliater.frontdoor_backdoor import sars_cov2_discrete
    data = sars_cov2_discrete.generate(num_samples=1000, seed=1),
"""

from y0.graph import NxMixedGraph
from y0.examples import Example
from y0.algorithm.identify import Query


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

base_example = Example(
    name="SARS-CoV-2 Graph",
    reference="Mohammad-Taheri, S., Zucker, J., Hoyt, C. T., Sachs, K., Tewari, V., Ness, R., & Vitek,"
              " O. (2022). Do-calculus enables estimation of causal effects in partially observed"
              " biomolecular pathways. Bioinformatics, 38(Supplement_1), i350-i358.",
    graph=graph,
    description="This system models activation of Cytokine Release Syndrome (Cytokine Storm), known to"
                " cause tissue damage in severely ill SARS-CoV-2  patients. The network was extracted"
                " from COVID-19 Open Research Dataset (CORD-19) document corpus using the Integrated Dynamical"
                " Reasoner and Assembler (INDRA) workflow, and by quering and expressing the corresponding causal"
                " statements in the Biological Expression Language (BEL). Presence of latent variables was determined"
                " by querying pairs of entities in the network for common causes in the corpus.",
    example_queries=[Query.from_str(treatments="Sil6r", outcomes="cytok"),
                     Query.from_str(treatments="EGFR", outcomes="cytok")]
)

base_example.__doc__ = base_example.description