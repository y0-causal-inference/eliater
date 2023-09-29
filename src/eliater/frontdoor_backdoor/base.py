"""This module contains methods to generate random continuous data for the frontdoor backdoor example."""

from typing import Optional

import numpy as np
import pandas as pd

from y0.algorithm.identify import Query
from y0.dsl import Variable, W, X, Y, Z
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "base_example",
]

graph = NxMixedGraph.from_edges(directed=[(W, X), (X, Z), (Z, Y), (W, Y)])


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
    if W in treatments:
        w = np.full(num_samples, treatments[W])
    else:
        w = generator.normal(loc=10, scale=1, size=num_samples)
    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        x = generator.normal(loc=w * 0.7, scale=3, size=num_samples)
    if Z in treatments:
        z = np.full(num_samples, treatments[Z])
    else:
        z = generator.normal(loc=x * 0.4, scale=2, size=num_samples)
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(loc=z * 0.5 + w * 0.3, scale=6)
    data = pd.DataFrame({W.name: w, Z.name: z, X.name: x, Y.name: y})
    return data


base_example = Example(
    name="frontdoor_backdoor example",
    reference="frontdoor_backdoor example from y0 module",
    graph=graph,
    description="In this example all the variables are continuous. This "
    "example is designed to check if the conditional independencies implied by"
    "the graph align with the ones implied by data via the Pearson test.",
    generate_data=generate,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
