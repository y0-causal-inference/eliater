"""Examples for eliater."""

from .ecoli import ecoli_transcription_example
from .frontdoor_backdoor_discrete import (
    single_mediator_with_multiple_confounders_nuisances_discrete_example as example_4,
)
from .sars import sars_cov_2_example
from .t_cell_signaling_pathway import t_cell_signaling_example
from ..frontdoor_backdoor.base import frontdoor_backdoor_example
from ..frontdoor_backdoor.example1 import multiple_mediators_single_confounder_example as example_1
from ..frontdoor_backdoor.example2 import example_2
from ..frontdoor_backdoor.example3 import (
    multiple_mediators_confounders_nuisance_vars_example as example_3,
)

__all__ = [
    "examples",
    # actual examples
    "ecoli_transcription_example",
    "sars_cov_2_example",
    "t_cell_signaling_example",
    "frontdoor_backdoor_example",
    "example_1",
    "example_2",
    "example_3",
    "example_4",
]

examples = [
    ecoli_transcription_example,
    sars_cov_2_example,
    t_cell_signaling_example,
    frontdoor_backdoor_example,
    example_1,
    example_2,
    example_3,
    example_4,
]
