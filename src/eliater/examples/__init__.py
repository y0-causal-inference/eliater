"""Examples for eliater."""

from .ecoli import ecoli_transcription_example
from .frontdoor_backdoor_discrete import (
    single_mediator_with_multiple_confounders_nuisances_discrete_example,
)
from .sars import sars_cov_2_example
from .t_cell_signaling_pathway import t_cell_signaling_example
from ..frontdoor_backdoor.example2 import example_2

__all__ = [
    "examples",
    # actual examples
    "ecoli_transcription_example",
    "sars_cov_2_example",
    "t_cell_signaling_example",
    "single_mediator_with_multiple_confounders_nuisances_discrete_example",
    "example_2",
]

examples = [
    ecoli_transcription_example,
    sars_cov_2_example,
    t_cell_signaling_example,
    example_2,
]
