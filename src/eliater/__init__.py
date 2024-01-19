# -*- coding: utf-8 -*-

"""A high level, end-to-end causal inference workflow."""

from .api import workflow
from .discover_latent_nodes import remove_nuisance_variables
from .network_validation import add_ci_undirected_edges, plot_ci_size_dependence
from .version import get_version

__all__ = [
    "workflow",
    "remove_nuisance_variables",
    "add_ci_undirected_edges",
    "plot_ci_size_dependence",
]


def version_df():
    import y0.version
    import datetime
    import pandas

    rows = [
        ("eliater", get_version(with_git_hash=True)),
        ("y0", y0.version.get_version(with_git_hash=True)),
        ("Run at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]
    return pandas.DataFrame(rows, columns=["key", "value"])
