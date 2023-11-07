"""This module contains multiple variations of the "frontdoor-backdoor" graph.

A frontdoor graph is a network structure where there is an exposure variable, and an
outcome, and one or more variables on the directed path connecting exposure to the
outcome. In addition, it contains one or more latent confounders between an exposure and the
outcome. As the confounders are latent, the effect of exposure on the outcome can be estimated
using Pearl's frontdoor criterion.

A backdoor graph is a network structure where there is an exposure variable, and an
outcome, and one or more observed confounders between an exposure and the
outcome. As the confounders are observed, the effect of exposure on the outcome can be estimated
using Pearl's backdoor criterion.

A frontdoor-backdoor graph is designed to have the properties from both graph. It is a network that
includes an exposure variable, and an outcome, and one or more variables on the directed path connecting
exposure to the outcome. In addition, it contains one or more observed confounders between an exposure and the
outcome. As the confounders are observed and mediators are present, the effect of exposure on the outcome can be
estimated using Pearl's frontdoor or backdoor criterion.

"""

from y0.examples import Example

from .base import base_example
from .example1 import multiple_mediators_single_confounder_example
from .example2 import multiple_mediators_with_multiple_confounders_example
from .example3 import multiple_mediators_confounders_nuisance_vars_example

__all__ = [
    "base_example",
    "example1",
    "example2",
    "example3",
]

for x in __all__:
    if isinstance(v := locals()[x], Example):
        v.__doc__ = v.description
