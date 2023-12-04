# -*- coding: utf-8 -*-

"""A high level, end-to-end causal inference workflow."""

from .discover_latent_nodes import remove_nuisance_variables
from .api import workflow

__all__ = [
    "workflow",
    "remove_nuisance_variables",
]
