"""Example data."""

from pathlib import Path

import pandas as pd

__all__ = [
    "SACHS_PATH",
]

HERE = Path(__file__)
SACHS_PATH = HERE.joinpath("sachs_discretized_2bin.csv")


def load_sachs_df() -> pd.DataFrame:
    return pd.read_csv(SACHS_PATH, index_col=False)
