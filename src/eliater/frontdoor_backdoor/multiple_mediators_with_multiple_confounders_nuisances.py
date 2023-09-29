"""This module contains a method to generate testing data for the multi_mediators_confounder_nuisance_var case study."""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Z1, Z2, Z3, Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

M1 = Variable("M1")
M2 = Variable("M2")
R1 = Variable("R1")
R2 = Variable("R2")
R3 = Variable("R3")

__all__ = [
    "multi_mediators_confounders_nuisance_vars_example",
]

graph = NxMixedGraph.from_edges(
    directed=[
        (Z1, X),
        (X, M1),
        (M1, M2),
        (M2, Y),
        (Z1, Z2),
        (Z2, Z3),
        (Z3, Y),
        (M1, R1),
        (R1, R2),
        (R2, R3),
        (Y, R3),
    ],
    undirected=[
        (Z1, X),
    ],
)


def generate(
    num_samples: int = 1000,
    treatments: dict[Variable, float] | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate testing data for the multi_mediators_confounder_nuisance_var case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the multi_mediators_confounder_nuisance_var example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    # latent node between X and Z1
    u1 = generator.normal(loc=40.0, scale=10.0, size=num_samples)
    # latent node between Y and Z2
    u2 = generator.normal(loc=50.0, scale=10.0, size=num_samples)

    beta0_z1 = 1.5
    beta_u1_to_z1 = 0.7

    if Z1 in treatments:
        z1 = np.full(num_samples, treatments[Z1])
    else:
        loc_z1 = beta0_z1 + u1 * beta_u1_to_z1
        z1 = generator.normal(loc=loc_z1, scale=10.0, size=num_samples)

    beta0_z2 = 3
    beta_z1_to_z2 = 0.3
    beta_u2_to_z2 = 0.6

    if Z2 in treatments:
        z2 = np.full(num_samples, treatments[Z2])
    else:
        loc_z2 = beta0_z2 + z1 * beta_z1_to_z2 + u2 * beta_u2_to_z2
        z2 = generator.normal(loc=loc_z2, scale=10.0, size=num_samples)

    beta0_z3 = 4
    beta_z2_to_z3 = 0.6

    if Z3 in treatments:
        z3 = np.full(num_samples, treatments[Z3])
    else:
        loc_z3 = beta0_z3 + z2 * beta_z2_to_z3
        z3 = generator.normal(loc=loc_z3, scale=10.0, size=num_samples)

    beta0_x = 1
    beta_z1_to_x = 0.6
    beta_u1_to_x = 0.7

    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        loc_x = beta0_x + z1 * beta_z1_to_x + u1 * beta_u1_to_x
        x = generator.normal(loc=loc_x, scale=10.0, size=num_samples)

    beta0_m1 = 2
    beta_x_to_m1 = 0.7

    if M1 in treatments:
        m1 = np.full(num_samples, treatments[M1])
    else:
        loc_m1 = beta0_m1 + x * beta_x_to_m1
        m1 = generator.normal(loc=loc_m1, scale=10.0, size=num_samples)

    beta0_m2 = 2
    beta_m1_to_m2 = 0.7

    if M2 in treatments:
        m2 = np.full(num_samples, treatments[M2])
    else:
        loc_m2 = beta0_m2 + m1 * beta_m1_to_m2
        m2 = generator.normal(loc=loc_m2, scale=10.0, size=num_samples)

    beta0_y = 1.8
    beta_z3_to_y = 0.5
    beta_m2_to_y = 0.7
    beta_u2_to_y = 0.8
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(
            loc=beta0_y + z3 * beta_z3_to_y + m2 * beta_m2_to_y + u2 * beta_u2_to_y,
            scale=10.0,
            size=num_samples,
        )

    beta0_r1 = -3
    beta_m1_to_r1 = 0.7

    if R1 in treatments:
        r1 = np.full(num_samples, treatments[R1])
    else:
        loc_r1 = beta0_r1 + m1 * beta_m1_to_r1
        r1 = generator.normal(loc=loc_r1, scale=10.0, size=num_samples)

    beta0_r2 = -3
    beta_r1_to_r2 = 0.7

    if R2 in treatments:
        r2 = np.full(num_samples, treatments[R2])
    else:
        loc_r2 = beta0_r2 + r1 * beta_r1_to_r2
        r2 = generator.normal(loc=loc_r2, scale=10.0, size=num_samples)

    beta0_r3 = -3
    beta_r2_to_r3 = 0.7
    beta_y_to_r3 = -0.4

    if R3 in treatments:
        r3 = np.full(num_samples, treatments[R3])
    else:
        loc_r3 = beta0_r3 + r2 * beta_r2_to_r3 + y * beta_y_to_r3
        r3 = generator.normal(loc=loc_r3, scale=10.0, size=num_samples)

    return pd.DataFrame(
        {
            X.name: x,
            M1.name: m1,
            M2.name: m2,
            Z1.name: z1,
            Z2.name: z2,
            Z3.name: z3,
            R1.name: r1,
            R2.name: r2,
            R3.name: r3,
            Y.name: y,
        }
    )


multi_mediators_confounders_nuisance_vars_example = Example(
    name="frontdoor with multiple mediators, confounders and nuisance variables",
    reference="Causal workflow paper, figure 4 (a).",
    description="This is an extension of the frontdoor_backdoor example from y0 module"
    " but with more variables directly connecting the treatment to outcome (mediators)"
    "and several additional variables that are a direct cause of both the treatment and outcome"
    "(confounders), and several nuisance variables. The nuisance variables are R1, R2, R3. "
    "They should not be part of query estimation because they are downstream of the outcome."
    " In the data generation process, all the variables are continuous, and the data was generated"
    " with the assumption that there exist a bi-directed edge between Z2 and Y. However, the graph does not include"
    " this confounder. This example is designed to check if the conditional independencies implied by the graph are"
    " aligned with the ones implied by the data via the Pearson test.",
    graph=graph,
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
