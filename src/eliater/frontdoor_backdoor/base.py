"""This module contains methods to generate random continuous data for the frontdoor backdoor example.

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
includes an exposure variable, and an outcome, and one or more variables on the directed path connecting
exposure to the outcome. In addition, it contains one or more observed confounders between an exposure and the
outcome. As the confounders are observed and mediators are present, the effect of exposure on the outcome can be
estimated using Pearl's frontdoor or backdoor criterion.
"""

from typing import Optional

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import M, Variable, X, Y, Z
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "base_example",
]

graph = NxMixedGraph.from_edges(directed=[(Z, X), (X, M), (M, Y), (Z, Y)])


def generate(
    num_samples: int = 1000,
    treatments: dict[Variable, float] | None = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate random continuous testing data.

    :param num_samples: The number of samples to generate. Defaults to 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with random continuous data.
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)
    if Z in treatments:
        z = np.full(num_samples, treatments[Z])
    else:
        z = generator.normal(loc=10, scale=1, size=num_samples)
    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        x = generator.normal(loc=z * 0.7, scale=3, size=num_samples)
    if M in treatments:
        m = np.full(num_samples, treatments[M])
    else:
        m = generator.normal(loc=x * 0.4, scale=2, size=num_samples)
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(loc=m * 0.5 + z * 0.3, scale=6)
    data = pd.DataFrame({Z.name: z, M.name: m, X.name: x, Y.name: y})
    return data


base_example = Example(
    name="Frontdoor/Backdoor Example",
    reference="frontdoor_backdoor example from y0 module",
    graph=graph,
    description="In this example all the variables are continuous. This "
    "example is designed to check if the conditional independencies implied by"
    "the graph align with the ones implied by data via the Pearson test.",
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)

base_example.__doc__ = base_example.description
