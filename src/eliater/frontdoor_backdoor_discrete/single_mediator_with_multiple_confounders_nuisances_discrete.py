"""This module generates discrete data for the single_mediator_with_multiple_confounders_nuisances_discrete_example case study.

The single_mediator_with_multiple_confounders_nuisances_discrete_example case study is a variation of the "frontdoor-backdoor" graph
where it contains an exposure variable (X),
and an outcome (Y), one mediator (M1) on the directed path connecting exposure to the outcome, and three observed confounders (Z1, Z2, Z3). In addition,
it contains three variables that are descendants of the mediator (R1, R2, R3).

A frontdoor graph is a network structure where there is an exposure variable, and an
outcome, and one or more variables on the directed path connecting exposure to the
outcome. In addition, it contains one or more latent confounders between an exposure and the
outcome. As the confounders are latent, the effect of exposure on the outcome can be estimated
using Pearl's frontdoor criterion.

A backdoor graph is a network structure where there is an exposure variable, and an
outcome, and one or more observed confounders between an exposure and the
outcome. As the confounders are observed, the effect of exposure on the outcome can be estimated
using Pearl's backdoor criterion.

A frontdoor-backdoor graph is designed to have the properties from both graph. It is a network that
includes an exposure variable, and an outcome, and one or more mediator variables on the directed path connecting
exposure to the outcome. In addition, it contains one or more observed confounders between an exposure and the
outcome. As the confounders are observed and mediators are present, the effect of exposure on the outcome can be
estimated using Pearl's frontdoor or backdoor criterion.
"""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Z1, Z2, Z3, Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

M1 = Variable("M1")
R1 = Variable("R1")
R2 = Variable("R2")
R3 = Variable("R3")

__all__ = [
    "single_mediator_with_multiple_confounders_nuisances_discrete_example",
]

graph = NxMixedGraph.from_edges(
    directed=[
        (Z1, X),
        (X, M1),
        (M1, Y),
        (Z1, Z2),
        (Z2, Z3),
        (Z3, Y),
        (M1, R1),
        (R1, R2),
        (R2, R3),
        (Y, R3),
    ],
)


def _r_exp(x):
    return 1 / (1 + np.exp(x))


def generate(
    num_samples: int = 1000,
    treatments: dict[Variable, float] | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate discrete testing data for the multiple_mediators_with_multiple_confounders_nuisances_discrete case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the multiple_mediators_with_multiple_confounders_nuisances_discrete example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    values_z1 = [0, 1]
    probs_z1 = [0.4, 0.6]

    if Z1 in treatments:
        z1 = np.full(num_samples, treatments[Z1])
    else:
        z1 = generator.choice(values_z1, num_samples, p=probs_z1)

    beta0_z2 = 1
    beta_z1_to_z2 = 0.3

    if Z2 in treatments:
        z2 = np.full(num_samples, treatments[Z2])
    else:
        probs_z2 = _r_exp(-beta0_z2 - z1 * beta_z1_to_z2)
        z2 = generator.binomial(n=1, p=probs_z2, size=num_samples)

    beta0_z3 = 1.2
    beta_z2_to_z3 = 0.6

    if Z3 in treatments:
        z3 = np.full(num_samples, treatments[Z3])
    else:
        probs_z3 = _r_exp(-beta0_z3 - z2 * beta_z2_to_z3)
        z3 = generator.binomial(n=1, p=probs_z3, size=num_samples)

    beta0_x = 1
    beta_z1_to_x = 0.6

    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        probs_x = _r_exp(-beta0_x - z1 * beta_z1_to_x)
        x = generator.binomial(n=1, p=probs_x, size=num_samples)

    beta0_m1 = 1
    beta_x_to_m1 = 0.7

    if M1 in treatments:
        m1 = np.full(num_samples, treatments[M1])
    else:
        probs_m1 = _r_exp(-beta0_m1 - x * beta_x_to_m1)
        m1 = generator.binomial(n=1, p=probs_m1, size=num_samples)

    beta0_y = 1.8
    beta_z3_to_y = 0.5
    beta_m1_to_y = 0.7
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        probs_y = _r_exp(-beta0_y - z3 * beta_z3_to_y - m1 * beta_m1_to_y)
        y = generator.binomial(n=1, p=probs_y, size=num_samples)

    beta0_r1 = 1.5
    beta_m1_to_r1 = 0.7

    if R1 in treatments:
        r1 = np.full(num_samples, treatments[R1])
    else:
        probs_r1 = _r_exp(-beta0_r1 - m1 * beta_m1_to_r1)
        r1 = generator.binomial(n=1, p=probs_r1, size=num_samples)

    beta0_r2 = 1.4
    beta_r1_to_r2 = 0.4

    if R2 in treatments:
        r2 = np.full(num_samples, treatments[R2])
    else:
        probs_r2 = _r_exp(-beta0_r2 - r1 * beta_r1_to_r2)
        r2 = generator.binomial(n=1, p=probs_r2, size=num_samples)

    beta0_r3 = 1.1
    beta_r2_to_r3 = 0.3
    beta_y_to_r3 = 0.3

    if R3 in treatments:
        r3 = np.full(num_samples, treatments[R3])
    else:
        probs_r3 = _r_exp(-beta0_r3 - r2 * beta_r2_to_r3 - y * beta_y_to_r3)
        r3 = generator.binomial(n=1, p=probs_r3, size=num_samples)

    return pd.DataFrame(
        {
            X.name: x,
            M1.name: m1,
            Z1.name: z1,
            Z2.name: z2,
            Z3.name: z3,
            R1.name: r1,
            R2.name: r2,
            R3.name: r3,
            Y.name: y,
        }
    )


single_mediator_with_multiple_confounders_nuisances_discrete_example = Example(
    name="frontdoor with multiple mediators, confounders and nuisance variables",
    reference="Causal workflow paper, figure 4 (a).",
    description="This is an extension of the frontdoor_backdoor example from y0 module"
    " but with more variables directly connecting the treatment to outcome (mediators)"
    "and several additional variables that are a direct cause of both the treatment and outcome"
    "(confounders), and several nuisance variables. The nuisance variables are R1, R2, R3. "
    "They should not be part of query estimation because they are downstream of the outcome."
    " In the data generation process, all the variables are discrete. This "
    "example is designed to check if the conditional independencies implied by the graph are"
    " aligned with the ones implied by the data via the X-square test.",
    graph=graph,
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
