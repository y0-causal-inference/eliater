from pathlib import Path

import pandas as pd

__all__ = [
    "load_sachs_df",
    "load_ecoli_obs",
    "load_sachs_test1",
    "load_sachs_test2"
]

HERE = Path(__file__).parent.resolve()
SACHS_PATH = HERE.joinpath("sachs_discretized_2bin.csv")
ECOLI_OBS_PATH = HERE.joinpath("EColi_obs_data.csv")
SACHS_TEST_1 = HERE.joinpath("cd3cd28+g0076.csv")
SACHS_TEST_2 = HERE.joinpath("cd3cd28.csv")


def load_sachs_df() -> pd.DataFrame:
    return pd.read_csv(SACHS_PATH, index_col=0)


def load_sachs_test1() -> pd.DataFrame:
    return pd.read_csv(SACHS_TEST_1)


def load_sachs_test2() -> pd.DataFrame:
    return pd.read_csv(SACHS_TEST_2)


def load_ecoli_obs() -> pd.DataFrame:
    return pd.read_csv(ECOLI_OBS_PATH)


