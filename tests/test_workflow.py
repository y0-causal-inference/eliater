import unittest
import pandas as pd
import numpy as np
import y0
from y0.dsl import Variable, X, Y, Z
multi_med_confounder = NxMixedGraph.from_edges(
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

def generate_data_for_multi_med_confounder(
        num_samples: int, treatments: dict[Variable, float] | None = None, *, seed: int | None = None
) -> pd.DataFrame:
    """Generate testing data for the multi_med_confounder case study.

    :param num_samples: The number of samples to generate. Try 1000.
    :param treatments: An optional dictionary of the values to fix each variable to.
    :param seed: An optional random seed for reproducibility purposes
    :returns: A pandas Dataframe with columns corresponding
        to the variable names in the multi_med_confounder example
    """
    if treatments is None:
        treatments = {}
    generator = np.random.default_rng(seed)

    u = generator.normal(loc=40.0, scale=10.0, size=num_samples)

    beta0_z1 = 1.5
    beta_u_to_z1 = 0.7

    if Z1 in treatments:
        z1 = np.full(num_samples, treatments[Z1])
    else:
        loc_z1 = beta0_z1 + u * beta_u_to_z1
        z1 = generator.normal(loc = loc_z1, scale = 10.0, size=num_samples)


    beta0_z2 = 3
    beta_z1_to_z2 = 0.3

    if Z2 in treatments:
        z2 = np.full(num_samples, treatments[Z2])
    else:
        loc_z2 = beta0_z2 + z1 * beta_z1_to_z2
        x = generator.normal(loc = loc_z2, scale = 10.0, size=num_samples)

    beta0_z3 = 4
    beta_z2_to_z3 = 0.6

    if Z3 in treatments:
        z3 = np.full(num_samples, treatments[Z3])
    else:
        loc_z3 = beta0_z3 + z2 * beta_z2_to_z3
        z3 = generator.normal(loc = loc_z3, scale = 10.0, size=num_samples)

    beta0_x = 1
    beta_z1_to_x = 0.6
    beta_u_to_x = 0.7

    if X in treatments:
        x = np.full(num_samples, treatments[X])
    else:
        loc_x = beta0_x + z1 * beta_z1_to_x + u * beta_u_to_x
        x = generator.normal(loc = loc_x, scale = 10.0, size=num_samples)

    beta0_m1 = 2
    beta_x_to_m1 = 0.7

    if M1 in treatments:
        m1 = np.full(num_samples, treatments[M1])
    else:
        loc_m1 = beta0_m1 + x * beta_x_to_m1
        m1 = generator.normal(loc = loc_m1, scale = 10.0, size=num_samples)

    beta0_m2 = 2
    beta_m1_to_m2 = 0.7

    if M2 in treatments:
        m2 = np.full(num_samples, treatments[M2])
    else:
        loc_m2 = beta0_m2 + m1 * beta_m1_to_m2
        m2 = generator.normal(loc = loc_m2, scale = 10.0, size=num_samples)

    beta0_y = 1.8
    beta_z3_to_y = 0.5
    beta_m2_to_y = 0.7
    if Y in treatments:
        y = np.full(num_samples, treatments[Y])
    else:
        y = generator.normal(
            loc=beta0_y + z3 * beta_z3_to_y + m2 * beta_m2_to_y,
            scale=10.0,
            size=num_samples,
        )

    beta0_r1 = -3
    beta_m1_to_r1 = 0.7

    if R1 in treatments:
        r1 = np.full(num_samples, treatments[R1])
    else:
        loc_r1 = beta0_r1 + x * beta_m1_to_r1
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
    bea_y_to_r3 = -0.4

    if R3 in treatments:
        r3 = np.full(num_samples, treatments[R3])
    else:
        loc_r3 = beta0_r3 + r1 * beta_r2_to_r3 + y * bea_y_to_r3
        r3 = generator.normal(loc=loc_r3, scale=10.0, size=num_samples)

class TestWorkflow(unittest.TestCase):


    def test_fix_graph(self):
        # TODO: Add graphs and examples

