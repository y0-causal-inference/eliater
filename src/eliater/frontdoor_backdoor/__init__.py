"""This module contains multiple variations of the "frontdoor-backdoor" graph."""

from y0.examples import Example

from .base import frontdoor_backdoor_example
from .multiple_mediators_single_confounder import multiple_mediators_single_confounder_example
from .multiple_mediators_with_multiple_confounders import multiple_mediators_confounders_example
from .multiple_mediators_with_multiple_confounders_nuisances import (
    multi_mediators_confounders_nuisance_vars_example,
)

__all__ = [
    "frontdoor_backdoor_example",
    "multiple_mediators_single_confounder_example",
    "multiple_mediators_confounders_example",
    "multi_mediators_confounders_nuisance_vars_example",
]

for x in __all__:
    if isinstance(v := locals()[x], Example):
        v.__doc__ = v.description
