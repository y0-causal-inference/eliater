"""This module contains a method to generate testing data for sars_large_example case study."""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Z1, Z2, Z3, Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "sars_large_example",
]

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


def _r_exp(x):
    return 1 / (1 + np.exp(x))


def generate(
    num_samples: int, treatments: dict[Variable, float] | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Generate testing data for the SARS-CoV-2 large graph.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names SARS-CoV-2 large graph
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    u_adam17_sil6r_value = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    u_il6_stat_egfr_value = generator.normal(loc=44.0, scale=10.0, size=num_samples)
    u_tnf_egfr_value = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    u_adam17_cytok_value = generator.normal(loc=44.0, scale=10.0, size=num_samples)

    beta0_adam17 = -1
    beta_u_adam17_cytok = 0.04
    beta_u_adam17_sil6r = 0.04
    adam17 = generator.normal(
        loc=100
        * _r_exp(
            -beta0_adam17
            - u_adam17_cytok_value * beta_u_adam17_cytok
            - u_adam17_sil6r_value * beta_u_adam17_sil6r
        ),
        scale=1,
        size=num_samples,
    )

    beta0_sil6r = -1.9
    beta_adam17_to_sil6r = 0.03
    beta_u_adam17_sil6r = 0.05
    sil6r_value = generator.normal(
        loc=100
        * _r_exp(
            -beta0_sil6r
            - adam17 * beta_adam17_to_sil6r
            - u_adam17_sil6r_value * beta_u_adam17_sil6r
        ),
        scale=1,
    )

    beta0_tnf = -1.8
    beta_adam17_to_tnf = 0.05
    beta_u_tnf_egfr = 0.06
    tnf = generator.normal(
        loc=100
        * _r_exp(-beta0_tnf - adam17 * beta_adam17_to_tnf - u_tnf_egfr_value * beta_u_tnf_egfr),
        scale=1,
    )

    beta0_egfr = -1.9
    beta_adam17_egfr = 0.03
    beta_u_il6_stat_egfr = -0.04
    beta_u_tnf_egfr = 0.02
    if Variable("EGFR") in treatments:
        egfr = np.full(num_samples, treatments[Variable("EGFR")])
    else:
        p = _r_exp(
            -beta0_egfr
            - adam17 * beta_adam17_egfr
            - u_il6_stat_egfr_value * beta_u_il6_stat_egfr
            - u_tnf_egfr_value * beta_u_tnf_egfr
        )
        egfr = generator.binomial(1, p, size=num_samples)

    beta0_il6_stat3 = -1.6
    beta_u_il6_stat_egfr = -0.05
    beta_sil6r_to_il6_stat3 = 0.04
    il6_stat3 = generator.normal(
        loc=100
        * _r_exp(
            -beta0_il6_stat3
            - u_il6_stat_egfr_value * beta_u_il6_stat_egfr
            - sil6r_value * beta_sil6r_to_il6_stat3
        ),
        scale=1,
    )

    beta0_cytok = -1.9
    beta_il6_stat3_tocytok = 0.02
    beta_egfr_tocytok = 0.06
    beta_tnf_tocytok = 0.01
    beta_u_adam17_cytok = 0.01
    cytok = generator.normal(
        loc=100
        * _r_exp(
            -beta0_cytok
            - il6_stat3 * beta_il6_stat3_tocytok
            - egfr * beta_egfr_tocytok
            - tnf * beta_tnf_tocytok
            - u_adam17_cytok_value * beta_u_adam17_cytok
        ),
        scale=1,
    )

    data = {
        "ADAM17": adam17,
        "Sil6r": sil6r_value,
        "TNF": tnf,
        "EGFR": egfr,
        "IL6STAT3": il6_stat3,
        "cytok": cytok,
    }
    df = pd.DataFrame(data)
    return df

sars_large_example = Example(
    name="SARS-CoV-2 Graph",
    reference="Mohammad-Taheri, S., Zucker, J., Hoyt, C. T., Sachs, K., Tewari, V., Ness, R., & Vitek, O. 2022."
    "Do-calculus enables estimation of causal effects in partially observed biomolecular pathways."
    "- Bioinformatics, 38 (Supplement_1),i350-i358.",
    graph=NxMixedGraph.from_str_edges(
        nodes=[
            "SARS_COV2",
            "ACE2",
            "Ang",
            "AGTR1",
            "ADAM17",
            "Toci",
            "Sil6r",
            "EGF",
            "TNF",
            "EGFR",
            "PRR",
            "NFKB",
            "IL6STAT3",
            "IL6AMP",
            "cytok",
            "Gefi",
        ],
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
    ),
    example_queries=[
        Query.from_str(treatments="Sil6r", outcomes="cytok"),
        Query.from_str(treatments="EGFR", outcomes="cytok"),
    ],
)