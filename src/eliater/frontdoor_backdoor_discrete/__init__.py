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

from .sars_cov2_discrete import sars_large_example

__all__ = [
    "sars_large_example",
]

for x in __all__:
    if isinstance(v := locals()[x], Example):
        v.__doc__ = v.description
