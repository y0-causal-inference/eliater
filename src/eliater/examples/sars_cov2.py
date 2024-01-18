"""This module contains a method to generate testing data for sars_large_example case study."""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Variable
from y0.examples import Example
from y0.graph import NxMixedGraph

from .sars import graph

__all__ = [
    "sars_large_example",
]


def _r_exp(x):
    return 1 / (1 + np.exp(x))


def generate_continuous(
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

    gefi = generator.normal(loc=45, scale=10, size=num_samples)
    toci = generator.normal(loc=45, scale=10, size=num_samples)
    u_sars_ang = generator.normal(loc=44.0, scale=10.0, size=num_samples)
    u_prr_nfkb = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    u_egf_egfr = generator.normal(loc=35.0, scale=10.0, size=num_samples)
    u_tnf_egfr = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    u_il6_stat_egfr = generator.normal(loc=44.0, scale=10.0, size=num_samples)
    u_adam17_sil6_ra = generator.normal(loc=40.0, scale=10.0, size=num_samples)

    beta0_u_sars_ang_to_sars_cov2 = -1.8
    beta_u_sars_ang = 0.05  # positive
    loc_sars_cov2 = np.array(
        100 / (1 + np.exp(-beta0_u_sars_ang_to_sars_cov2 - u_sars_ang * beta_u_sars_ang))
    )
    sars_cov2 = generator.normal(loc=loc_sars_cov2, scale=1)

    beta0_sars_cov2_to_ace2 = 1.5
    beta_sars_cov2_to_ace2 = -0.04  # negative
    loc_ace2 = 100 / (1 + np.exp(-beta0_sars_cov2_to_ace2 - sars_cov2 * beta_sars_cov2_to_ace2))
    ace2 = generator.normal(loc=loc_ace2, scale=1)

    beta0_ang = 1.1
    beta_ace2_to_ang = -0.06  # negative
    beta_u_sars_ang = 0.05  # positive
    loc_ang = 100 / (
        1 + np.exp(-beta0_ang - ace2 * beta_ace2_to_ang - u_sars_ang * beta_u_sars_ang)
    )
    ang = generator.normal(loc=loc_ang, scale=1)

    beta0_agtr1 = -1.5
    beta_ang_to_agtr1 = 0.08
    loc_agtr1 = 100 / (1 + np.exp(-beta0_agtr1 - ang * beta_ang_to_agtr1))
    agtr1 = generator.normal(loc=loc_agtr1, scale=1)

    beta0_adam17 = -1
    beta_agtr1_to_adam17 = 0.04
    beta_u_adam17_sil6r_to_adam17 = 0.04
    loc_adam17 = 100 / (
        1
        + np.exp(
            -beta0_adam17
            - agtr1 * beta_agtr1_to_adam17
            - u_adam17_sil6_ra * beta_u_adam17_sil6r_to_adam17
        )
    )
    adam17 = generator.normal(loc=loc_adam17, scale=1)

    beta0_sil6r = -1.9
    beta_adam17_to_sil6r = 0.03
    beta_u_to_sil6r = 0.05
    beta_toci_to_sil6r = -0.04  # negative
    loc_sil6r = 100 / (
        1
        + np.exp(
            -beta0_sil6r
            - adam17 * beta_adam17_to_sil6r
            - u_adam17_sil6_ra * beta_u_to_sil6r
            - toci * beta_toci_to_sil6r
        )
    )
    sil6r = generator.normal(loc=loc_sil6r, scale=1)

    beta0_egf = -1.6
    beta_adam17_to_egf = 0.03
    beta_u_to_egf = 0.05
    loc_egf = 100 / (
        1 + np.exp(-beta0_egf - adam17 * beta_adam17_to_egf - u_egf_egfr * beta_u_to_egf)
    )
    egf = generator.normal(loc=loc_egf, scale=1)

    beta0_tnf = -1.8
    beta_adam17_to_tnf = 0.05
    beta_u_to_tnf = 0.06
    loc_tnf = 100 / (
        1 + np.exp(-beta0_tnf - adam17 * beta_adam17_to_tnf - u_tnf_egfr * beta_u_to_tnf)
    )
    tnf = generator.normal(loc=loc_tnf, scale=1)

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
    loc_prr = 100 / (
        1 + np.exp(-beta0_prr - sars_cov2 * beta_sars_cov2_to_prr - u_prr_nfkb * beta_u_to_prr)
    )
    prr = generator.normal(loc=loc_prr, scale=1)

    beta0_nfkb = -1.8
    beta_prr_to_nfkb = 0.01
    beta_u_to_nfkb = -0.02
    beta_egfr_to_nfkb = 0.06
    beta_tnf_to_nfkb = 0.01
    loc_nfkb = 100 / (
        1
        + np.exp(
            -beta0_nfkb
            - prr * beta_prr_to_nfkb
            - u_prr_nfkb * beta_u_to_nfkb
            - egfr * beta_egfr_to_nfkb
            - tnf * beta_tnf_to_nfkb
        )
    )
    nfkb = generator.normal(loc=loc_nfkb, scale=1)

    beta0_il6_stat3 = -1.6
    beta_u_to_il6_stat3 = -0.05
    beta_sil6r_to_il6_stat3 = 0.04
    loc_il6_stat3 = 100 / (
        1
        + np.exp(
            -beta0_il6_stat3
            - u_il6_stat_egfr * beta_u_to_il6_stat3
            - sil6r * beta_sil6r_to_il6_stat3
        )
    )
    il6_stat3 = generator.normal(loc=loc_il6_stat3, scale=1)

    beta0_il6_amp = -1.98
    beta_nfkb_to_il6_amp = 0.02
    beta_il6_stat3_to_il6_amp = 0.03
    loc_il6_amp = 100 / (
        1
        + np.exp(
            -beta0_il6_amp - nfkb * beta_nfkb_to_il6_amp - il6_stat3 * beta_il6_stat3_to_il6_amp
        )
    )
    il6_amp = generator.normal(loc=loc_il6_amp, scale=1)

    beta0_cytok = -1.9
    beta_il6_amp_tocytok = 0.06
    loc_cytok = 100 / (1 + np.exp(-beta0_cytok - il6_amp * beta_il6_amp_tocytok))
    cytok = generator.normal(loc=loc_cytok, scale=1)

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
    graph=graph,
    example_queries=[
        Query.from_str(treatments="EGFR", outcomes="cytok"),
    ],
)


# obs_data = generate(num_samples=260, seed=1)
# print(np.mean(obs_data['cytok']))
# intv_data_1 = generate(num_samples=1000, seed=1, treatments = {Variable('EGFR'): 1})
# intv_data_0 = generate(num_samples=1000, seed=1, treatments = {Variable('EGFR'): 0})
# print(np.mean(intv_data_1['cytok']) - np.mean(intv_data_0['cytok'])) #ATE
