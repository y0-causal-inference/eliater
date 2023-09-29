"""This module contains multiple variations of the "frontdoor-backdoor" graph."""

from .base import base_example
from .multiple_mediators_single_confounder import multiple_mediators_example
from .multiple_mediators_with_multiple_confounders import multiple_mediators_confounder_example
from .multiple_mediators_with_multiple_confounders_nuisances import (
    multi_mediators_confounder_nuisance_var_example,
)

__all__ = [
    "base_example",
    "multiple_mediators_example",
    "multiple_mediators_confounder_example",
    "multi_mediators_confounder_nuisance_var_example",
]
