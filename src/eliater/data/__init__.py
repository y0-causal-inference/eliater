from pathlib import Path

import pandas as pd

__all__ = [
    "load_sachs_df",
    "load_ecoli_obs",
]

HERE = Path(__file__).parent.resolve()
SACHS_PATH = HERE.joinpath("sachs_discretized_2bin.csv")
ECOLI_OBS_PATH = HERE.joinpath("EColi_obs_data.csv")


def load_sachs_df() -> pd.DataFrame:
    return pd.read_csv(SACHS_PATH, index_col=0)


def load_ecoli_obs() -> pd.DataFrame:
    return pd.read_csv(ECOLI_OBS_PATH)
