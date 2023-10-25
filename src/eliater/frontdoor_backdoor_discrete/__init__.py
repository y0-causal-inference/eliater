"""This module contains multiple variations of the "frontdoor-backdoor" graph."""

from y0.examples import Example

from .single_mediator_with_multiple_confounders_nuisances_discrete import (
    single_mediator_with_multiple_confounders_nuisances_discrete_example,
)

__all__ = [
    "single_mediator_with_multiple_confounders_nuisances_discrete_example",
]

for x in __all__:
    if isinstance(v := locals()[x], Example):
        v.__doc__ = v.description
