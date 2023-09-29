"""This module contains multiple variations of the "frontdoor-backdoor" graph."""

from .base import base_example
from .multiple_mediators import multiple_mediators_example
from .multiple_mediators_with_confounder import multiple_mediators_confounder_example
from .multiple_mediators_with_nuisance_confounder import (
    multi_mediators_confounder_nuisance_var_example,
)

__all__ = [
    "base_example",
    "multiple_mediators_example",
    "multiple_mediators_confounder_example",
    "multi_mediators_confounder_nuisance_var_example",
]
