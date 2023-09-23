from typing import Optional

import pandas as pd
import numpy as np


def generate_random_continuous_data(num_samples: int, seed: Optional[int] = 1) -> pd.DataFrame:
    """Generate random continuous testing data.

    :param num_samples: The number of samples to generate. Try 1000.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with random continuous data.
    """
    np.random.seed(seed)
    W = np.random.normal(loc=10, scale=1, size=num_samples)
    X = np.random.normal(loc=W * 0.7, scale=3, size=num_samples)
    Z = np.random.normal(loc=X * 0.4, scale=2, size=num_samples)
    Y = np.random.normal(loc=Z * 0.5 + W * 0.3, scale=6)
    data = pd.DataFrame({"W": W, "Z": Z, "X": X, "Y": Y})
    return data
