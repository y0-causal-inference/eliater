"""This module contains a method to generate testing data for the multi_mediators case study."""

import numpy as np
import pandas as pd
from y0.algorithm.identify import Query
from y0.dsl import Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "multiple_mediators_example",
]

M1 = Variable("M1")
M2 = Variable("M2")
R1 = Variable("R1")
R2 = Variable("R2")
R3 = Variable("R3")


graph = NxMixedGraph.from_edges(
    directed=[
        (X, M1),
        (M1, M2),
        (M2, Y),
    ],
    undirected=[
        (X, Y),
        # (M1, Y) We generated data for this graph with the assumption that the bi-directed edge between
        # M1 and Y is present. However, we assume that the prior knowledge graph does not have this information.
    ],
)


def generate(
    num_samples: int = 1000,
    treatments: dict[Variable, float] | None = None,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate testing data for the multi_mediators case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the multi_mediators example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    u = generator.normal(loc=50.0, scale=10.0, size=num_samples)
    u2 = generator.normal(loc=40.0, scale=10.0, size=num_samples)

    beta0_x = 1
    beta_u_to_x = 0.7

    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        loc_x = beta0_x + u * beta_u_to_x
        x = generator.normal(loc=loc_x, scale=10.0, size=num_samples)

    beta0_m1 = 2
    beta_x_to_m1 = 0.7
    beta_u2_to_m1 = 0.8

    if M1 in treatments:
        m1 = np.full(num_samples, treatments[M1])
    else:
        loc_m1 = beta0_m1 + x * beta_x_to_m1 + u2 * beta_u2_to_m1
        m1 = generator.normal(loc=loc_m1, scale=10.0, size=num_samples)

    beta0_m2 = 2
    beta_m1_to_m2 = 0.7

    if M2 in treatments:
        m2 = np.full(num_samples, treatments[M2])
    else:
        loc_m2 = beta0_m2 + m1 * beta_m1_to_m2
        m2 = generator.normal(loc=loc_m2, scale=10.0, size=num_samples)

    beta0_y = 1.8
    beta_u_to_y = 0.5
    beta_u2_to_y = 0.7
    beta_m2_to_y = 0.7
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(
            loc=beta0_y + u * beta_u_to_y + m2 * beta_m2_to_y + u2 * beta_u2_to_y,
            scale=10.0,
            size=num_samples,
        )

    return pd.DataFrame({X.name: x, M1.name: m1, M2.name: m2, Y.name: y})


multiple_mediators_example = Example(
    name="Multiple mediators example",
    reference="Inspired by the frontdoor example, but with multiple mediators.",  # TODO that is not a reference to a paer. Put that in the description
    graph=graph,
    description=...,
    # TODO write a good description
    #  - What phenomena does the graph model here. Give a real-world example if possible
    #  - What is this example graph used to demonstrate?
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
