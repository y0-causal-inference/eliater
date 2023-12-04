"""Example data."""

from pathlib import Path

import pandas as pd

__all__ = [
    "SACHS_PATH",
]

HERE = Path(__file__).parent.resolve()
SACHS_PATH = HERE.joinpath("sachs_discretized_2bin.csv")


def load_sachs_df() -> pd.DataFrame:
    """Loads the sachs discrete data into a pandas dataframe.

    .. todo::

        detailed explanation of:

        1. the biological explanation of what this data is (i.e., what kinds of experiments were done, what model system, what equipment was used?)
        2. how it ended up in this repo (what preprocessing steps were done, etc.)
        3. Why are columns seemingly ranges? How does pandas or downstream code deal with this?
    """
    return pd.read_csv(SACHS_PATH, index_col=False)
