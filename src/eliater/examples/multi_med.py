import numpy as np
import pandas as pd

import y0
from y0.dsl import Z1, Z2, Z3, Variable, X, Y

M1 = y0.dsl.Variable("M1")
M2 = y0.dsl.Variable("M2")
R1 = y0.dsl.Variable("R1")
R2 = y0.dsl.Variable("R2")
R3 = y0.dsl.Variable("R3")


def generate_data_for_multi_med(
    num_samples: int, treatments: dict[Variable, float] | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Generate testing data for the multi_med case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the multi_med example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    u = generator.normal(loc=50.0, scale=10.0, size=num_samples)

    beta0_x = 1
    beta_u_to_x = 0.7

    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        loc_x = beta0_x + u * beta_u_to_x
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
    beta_u_to_y = 0.5
    beta_m2_to_y = 0.7
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(
            loc=beta0_y + u * beta_u_to_y + m2 * beta_m2_to_y,
            scale=10.0,
            size=num_samples,
        )

    return pd.DataFrame(
        {
            X.name: x,
            M1.name: m1,
            M2.name: m2,
            Y.name: y,
        }
    )
