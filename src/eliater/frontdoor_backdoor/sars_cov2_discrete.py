"""This module contains a method to generate testing data for sars_large_example case study."""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Variable
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "sars_large_example",
]


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

    gefi = generator.binomial(n=1, p=0.9, size=num_samples)
    toci = generator.binomial(n=1, p=0.7, size=num_samples)
    u_sars_ang = generator.binomial(n=1, p=0.6, size=num_samples)
    u_prr_nfkb = generator.binomial(n=1, p=0.6, size=num_samples)
    u_egf_egfr = generator.binomial(n=1, p=0.6, size=num_samples)
    u_tnf_egfr = generator.binomial(n=1, p=0.6, size=num_samples)
    u_il6_stat_egfr = generator.binomial(n=1, p=0.6, size=num_samples)
    u_adam17_sil6_ra = generator.binomial(n=1, p=0.6, size=num_samples)

    beta0_u_sars_ang_to_sars_cov2 = -1.8
    beta_u_sars_ang = 0.05  # positive
    loc_sars_cov2 = np.array(
        _r_exp(-beta0_u_sars_ang_to_sars_cov2 - u_sars_ang * beta_u_sars_ang)
    )
    sars_cov2 = generator.binomial(n=1, p=loc_sars_cov2, size=num_samples)

    beta0_sars_cov2_to_ace2 = 1.5
    beta_sars_cov2_to_ace2 = -0.04  # negative
    loc_ace2 = _r_exp(-beta0_sars_cov2_to_ace2 - sars_cov2 * beta_sars_cov2_to_ace2)
    ace2 = generator.binomial(n=1, p=loc_ace2, size=num_samples)

    beta0_ang = 1.1
    beta_ace2_to_ang = -0.06  # negative
    beta_u_sars_ang = 0.05  # positive
    loc_ang = _r_exp(-beta0_ang - ace2 * beta_ace2_to_ang - u_sars_ang * beta_u_sars_ang)
    ang = generator.binomial(n=1, p=loc_ang, size=num_samples)

    beta0_agtr1 = -1.5
    beta_ang_to_agtr1 = 0.08
    loc_agtr1 = _r_exp(-beta0_agtr1 - ang * beta_ang_to_agtr1)
    agtr1 = generator.binomial(n=1, p=loc_agtr1, size=num_samples)

    beta0_adam17 = -1
    beta_agtr1_to_adam17 = 0.04
    beta_u_adam17_sil6r_to_adam17 = 0.04
    loc_adam17 = _r_exp(
            -beta0_adam17
            - agtr1 * beta_agtr1_to_adam17
            - u_adam17_sil6_ra * beta_u_adam17_sil6r_to_adam17
        )
    adam17 = generator.binomial(n=1, p=loc_adam17, size=num_samples)

    beta0_sil6r = -1.9
    beta_adam17_to_sil6r = 0.03
    beta_u_to_sil6r = 0.05
    beta_toci_to_sil6r = -0.04  # negative
    loc_sil6r = _r_exp(
            -beta0_sil6r
            - adam17 * beta_adam17_to_sil6r
            - u_adam17_sil6_ra * beta_u_to_sil6r
            - toci * beta_toci_to_sil6r
        )
    sil6r = generator.binomial(n=1, p=loc_sil6r, size=num_samples)

    beta0_egf = -1.6
    beta_adam17_to_egf = 0.03
    beta_u_to_egf = 0.05
    loc_egf = _r_exp(-beta0_egf - adam17 * beta_adam17_to_egf - u_egf_egfr * beta_u_to_egf)
    egf = generator.binomial(n=1, p=loc_egf, size=num_samples)

    beta0_tnf = -1.8
    beta_adam17_to_tnf = 0.05
    beta_u_to_tnf = 0.06
    loc_tnf = _r_exp(-beta0_tnf - adam17 * beta_adam17_to_tnf - u_tnf_egfr * beta_u_to_tnf)
    tnf = generator.binomial(n=1, p=loc_tnf, size=num_samples)

    beta0_egfr = -1.9
    beta_egf_to_egfr = 0.03
    beta_u1_to_egfr = 0.05
    beta_u2_to_egfr = 0.02
    beta_u3_to_egfr = 0.04
    beta_gefi_to_egfr = -0.08  # negative
    if Variable("EGFR") in treatments:
        egfr = np.full(num_samples, treatments[Variable("EGFR")])
    else:
        p = _r_exp(
            -beta0_egfr
            - egf * beta_egf_to_egfr
            - u_il6_stat_egfr * beta_u1_to_egfr
            - u_tnf_egfr * beta_u2_to_egfr
            - u_egf_egfr * beta_u3_to_egfr
            - gefi * beta_gefi_to_egfr
        )
        egfr = generator.binomial(1, p, size=num_samples)

    beta0_prr = -1.4
    beta_sars_cov2_to_prr = 0.05
    beta_u_to_prr = 0.02
    loc_prr = _r_exp(-beta0_prr - sars_cov2 * beta_sars_cov2_to_prr - u_prr_nfkb * beta_u_to_prr)
    prr = generator.binomial(n=1, p=loc_prr, size=num_samples)

    beta0_nfkb = -1.8
    beta_prr_to_nfkb = 0.01
    beta_u_to_nfkb = 0.02
    beta_egfr_to_nfkb = 0.7
    beta_tnf_to_nfkb = 0.01
    loc_nfkb = _r_exp(
            -beta0_nfkb
            - prr * beta_prr_to_nfkb
            - u_prr_nfkb * beta_u_to_nfkb
            - egfr * beta_egfr_to_nfkb
            - tnf * beta_tnf_to_nfkb
        )
    nfkb = generator.binomial(n=1, p=loc_nfkb, size=num_samples)

    beta0_il6_stat3 = -1.6
    beta_u_to_il6_stat3 = -0.05
    beta_sil6r_to_il6_stat3 = 0.04
    loc_il6_stat3 = _r_exp(
            -beta0_il6_stat3
            - u_il6_stat_egfr * beta_u_to_il6_stat3
            - sil6r * beta_sil6r_to_il6_stat3
        )
    il6_stat3 = generator.binomial(n=1, p=loc_il6_stat3, size=num_samples)

    beta0_il6_amp = -1.1
    beta_nfkb_to_il6_amp = -5
    beta_il6_stat3_to_il6_amp = 0.03
    loc_il6_amp = _r_exp(
            -beta0_il6_amp - nfkb * beta_nfkb_to_il6_amp - il6_stat3 * beta_il6_stat3_to_il6_amp
        )
    il6_amp = generator.binomial(n=1, p=loc_il6_amp, size=num_samples)

    beta0_cytok = -1.1
    beta_il6_amp_tocytok = 4
    loc_cytok = _r_exp(-beta0_cytok - il6_amp * beta_il6_amp_tocytok)
    cytok = generator.binomial(n=1, p=loc_cytok, size=num_samples)

    data = {
        "SARS_COV2": sars_cov2,
        "ACE2": ace2,
        "Ang": ang,
        "AGTR1": agtr1,
        "ADAM17": adam17,
        "Toci": toci,
        "Sil6R": sil6r,
        "EGF": egf,
        "TNF": tnf,
        "Gefi": gefi,
        "EGFR": egfr,
        "PRR": prr,
        "NFKB": nfkb,
        "IL6STAT3": il6_stat3,
        "IL6AMP": il6_amp,
        "cytok": cytok,
    }
    df = pd.DataFrame(data)
    return df


sars_large_example = Example(
    name="SARS-CoV-2 large Graph",
    reference="Mohammad-Taheri, S., Zucker, J., Hoyt, C. T., Sachs, K., Tewari, V., Ness, R., & Vitek, O. 2022."
    "Do-calculus enables estimation of causal effects in partially observed biomolecular pathways."
    "- Bioinformatics, 38 (Supplement_1),i350-i358.",
    description="In this example EGFR is generated as a binary value. Hence, if you want to intervene on it, please"
                "choose either o or 1",
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
        Query.from_str(treatments="EGFR", outcomes="cytok"),
    ],
)


#obs_data = generate(num_samples=100, seed=1)
#obs_data.to_csv("~/Github/Causal_workflow_in_R/Covid_case_study/covid_data_discrete.csv")
#intv_data_1 = generate(num_samples=40, seed=1, treatments = {Variable('EGFR'): 1})
#intv_data_0 = generate(num_samples=40, seed=1, treatments = {Variable('EGFR'): 0})
#print(np.mean(intv_data_1['cytok']) - np.mean(intv_data_0['cytok'])) #ATE
