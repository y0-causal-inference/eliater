# -*- coding: utf-8 -*-

"""A high level, end-to-end causal inference workflow."""

from .api import workflow
from .discover_latent_nodes import remove_nuisance_variables
from .network_validation import (
    add_ci_undirected_edges,
    discretize_binary,
    plot_ci_size_dependence,
    plot_treatment_and_outcome,
)
from .notebook_utils import (
    step_1_notebook,
    step_2_notebook,
    step_3_notebook,
    step_5_notebook_synthetic,
)
from .version import get_version

__all__ = [
    "workflow",
    "remove_nuisance_variables",
    "add_ci_undirected_edges",
    "plot_ci_size_dependence",
]


def version_df():
    """Get provenance information for how the code was run."""
    import datetime

    import pandas

    import y0.version

    rows = [
        ("eliater", get_version(with_git_hash=True)),
        ("y0", y0.version.get_version(with_git_hash=True)),
        ("Run at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    return pandas.DataFrame(rows, columns=["key", "value"])
