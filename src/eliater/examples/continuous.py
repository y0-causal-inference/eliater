"""This module contains methods to generate random continuous data."""

from typing import Optional

import numpy as np
import pandas as pd
from y0.algorithm.identify import Query
from y0.examples import Example
from y0.graph import NxMixedGraph

__all__ = [
    "continuous_example",
]

continuous = NxMixedGraph.from_str_edges(directed=[("W", "X"), ("X", "Z"), ("Z", "Y"), ("W", "Y")])


def generate_random_continuous_data(num_samples: int, seed: Optional[int] = 1) -> pd.DataFrame:
    """Generate random continuous testing data.

    :param num_samples: The number of samples to generate. Try 1000.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with random continuous data.
    """
    np.random.seed(seed)
    w = np.random.normal(loc=10, scale=1, size=num_samples)
    x = np.random.normal(loc=w * 0.7, scale=3, size=num_samples)
    z = np.random.normal(loc=x * 0.4, scale=2, size=num_samples)
    y = np.random.normal(loc=z * 0.5 + w * 0.3, scale=6)
    data = pd.DataFrame({"W": w, "Z": z, "X": x, "Y": y})
    return data


continuous_example = Example(
    name=...,
    reference=...,
    graph=continuous,
    generate_data=generate_random_continuous_data,
    example_queries=[Query.from_str(treatments="X", outcomes="Y")],
)
