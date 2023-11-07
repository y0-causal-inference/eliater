"""This module contains a method to generate continuous testing data for the multiple_mediators_single_confounder case study.

The multiple_mediators_single_confounder case study is a variation of the "frontdoor" graph, where it contains an
exposure variable (X), and an outcome (Y), and two variables (M1, M2) on the directed path connecting exposure
to the outcome. In addition, it contains a bi-directed edge between X and Y, indicating the existence of one
or more latent confounders between the exposure and the outcome. As the confounder(s) are latent, the effect
of exposure on the outcome can be estimated using Pearl's frontdoor criterion.
"""

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Variable, X, Y
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "multiple_mediators_single_confounder_example",
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

    # latent confounder between x and y
    u = generator.normal(loc=50.0, scale=10.0, size=num_samples)

    # latent confounder between m1 and y
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


multiple_mediators_single_confounder_example = Example(
    name="Multiple mediators and a single confounder example",
    reference="Sara Taheri",
    graph=graph,
    description="This is an extension of the frontdoor_backdoor example from y0 module"
    " but with more variables directly connecting the treatment to outcome (mediators)"
    "and an additional variable that is a direct cause of both the treatment and outcome"
    "(confounder). In the data generation process, the data was generated with the assumption"
    " that there exist a bi-directed edge between M1 and Y. However, the graph does not include"
    " this confounder. In this example all the variables are continuous. It is designed to check"
    " if the conditional independencies implied by the graph are aligned with the ones implied by"
    " data via the Pearson test.",
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
