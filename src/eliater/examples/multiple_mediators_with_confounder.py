"""This module contains a method to generate testing data for the multi_mediators_confounder case study."""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Z1, Z2, Z3, Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "multiple_mediators_confounder_example",
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
    undirected=[
        (Z1, X),
        # (Y, Z2)
        # We are generating data with the assumption that there is a bi-directed edge between
        # Y and Z2, but that bi-directed edge is missed from this prior knowledge graph.
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


multiple_mediators_confounder_example = Example(
    name="front door with multiple mediators and multiple confounders example",
    reference="Causal workflow paper, figure 4 (b). The query can be estimated with both front-door and "
    "back-door approaches",
    description="This is an extension of front door example but with multiple mediators and multiple confounders",
    graph=graph,
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
