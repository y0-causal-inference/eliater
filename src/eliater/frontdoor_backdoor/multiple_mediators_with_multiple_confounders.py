"""This module contains a method to generate continuous testing data for the multiple_mediators_with_multiple_confounders case study.

The multiple_mediators_with_multiple_confounders case study is a variation of the "frontdoor-backdoor" graph, where it contains an exposure variable (X),
and an outcome (Y), two variables (M1, M2) on the directed path connecting exposure to the outcome, and three observed confounders (Z1, Z2, Z3).
As the confounder(s) are observed, the effect of exposure on the outcome can be estimated using Pearl's frontdoor or backdoor criterion.
"""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Z1, Z2, Z3, Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "multiple_mediators_with_multiple_confounders_example",
]

M1 = Variable("M1")
M2 = Variable("M2")
R1 = Variable("R1")
R2 = Variable("R2")
R3 = Variable("R3")


graph = NxMixedGraph.from_edges(
    directed=[
        (Z1, X),
        (X, M1),
        (M1, M2),
        (M2, Y),
        (Z1, Z2),
        (Z2, Z3),
        (Z3, Y),
    ],
)


def generate(
    num_samples: int = 1000,
    treatments: dict[Variable, float] | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate testing data for the multi_mediators_confounder case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the multi_mediators_confounder example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    # latent node between X and Z1
    # u1 = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    # latent node between Y and Z2
    u2 = generator.normal(loc=50.0, scale=10.0, size=num_samples)

    beta0_z1 = 50  # 1.5
    # beta_u1_to_z1 = 0.3

    if Z1 in treatments:
        z1 = np.full(num_samples, treatments[Z1])
    else:
        loc_z1 = beta0_z1  # + u1 * beta_u1_to_z1
        z1 = generator.normal(loc=loc_z1, scale=10.0, size=num_samples)

    beta0_z2 = 3
    beta_z1_to_z2 = 0.3
    beta_u2_to_z2 = 0.7

    if Z2 in treatments:
        z2 = np.full(num_samples, treatments[Z2])
    else:
        loc_z2 = beta0_z2 + z1 * beta_z1_to_z2 + u2 * beta_u2_to_z2
        z2 = generator.normal(loc=loc_z2, scale=4.0, size=num_samples)

    beta0_z3 = 4
    beta_z2_to_z3 = 0.6

    if Z3 in treatments:
        z3 = np.full(num_samples, treatments[Z3])
    else:
        loc_z3 = beta0_z3 + z2 * beta_z2_to_z3
        z3 = generator.normal(loc=loc_z3, scale=4.0, size=num_samples)

    beta0_x = 1
    beta_z1_to_x = 0.6
    # beta_u1_to_x = 0.3

    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        loc_x = beta0_x + z1 * beta_z1_to_x  # + u1 * beta_u1_to_x
        x = generator.normal(loc=loc_x, scale=4.0, size=num_samples)

    beta0_m1 = 2
    beta_x_to_m1 = 0.7

    if M1 in treatments:
        m1 = np.full(num_samples, treatments[M1])
    else:
        loc_m1 = beta0_m1 + x * beta_x_to_m1
        m1 = generator.normal(loc=loc_m1, scale=4.0, size=num_samples)

    beta0_m2 = 2
    beta_m1_to_m2 = 0.7

    if M2 in treatments:
        m2 = np.full(num_samples, treatments[M2])
    else:
        loc_m2 = beta0_m2 + m1 * beta_m1_to_m2
        m2 = generator.normal(loc=loc_m2, scale=7.0, size=num_samples)

    beta0_y = 1.8
    beta_z3_to_y = 0.5
    beta_m2_to_y = 0.7
    beta_u2_to_y = 0.8
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        loc_y = beta0_y + z3 * beta_z3_to_y + m2 * beta_m2_to_y + u2 * beta_u2_to_y
        y = generator.normal(
            loc=loc_y,
            scale=10.0,
            size=num_samples,
        )

    return pd.DataFrame(
        {
            X.name: x,
            M1.name: m1,
            M2.name: m2,
            Z1.name: z1,
            Z2.name: z2,
            Z3.name: z3,
            Y.name: y,
        }
    )


multiple_mediators_with_multiple_confounders_example = Example(
    name="front door with multiple mediators and multiple confounders example",
    reference="Sara Taheri",
    description="This is an extension of the frontdoor_backdoor example from y0 module"
    " but with more variables directly connecting the treatment to outcome (mediators)"
    "and several additional variables that are a direct cause of both the treatment and outcome"
    "(confounders). In the data generation process, the data was generated  with the assumption"
    " that there exist a bi-directed edge between Z2 and Y. However, the graph does not include"
    " this confounder. In this example all the variables are continuous. It is designed to check"
    " if the conditional independencies implied by the graph are aligned with the ones implied by"
    " data via the Pearson test.",
    graph=graph,
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
