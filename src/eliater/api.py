"""Implementation of Eliater workflow.

To run the workflow and reproduce results on all examples in the
package, use ``python -m eliater.api``.
"""

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import click
import pandas as pd

from eliater.discover_latent_nodes import remove_nuisance_variables
from eliater.examples import examples
from eliater.network_validation import add_ci_undirected_edges
from eliater.regression import get_eliater_regression
from y0.algorithm.estimation import estimate_ace
from y0.algorithm.identify import identify_outcomes
from y0.dsl import Expression, Variable
from y0.graph import NxMixedGraph, _ensure_set
from y0.struct import CITest

__all__ = [
    "workflow",
    "reproduce",
]

HERE = Path(__file__).parent.resolve()
RESULTS_PATH = HERE.joinpath("case_studies.tsv")

# Ignore all warnings
warnings.filterwarnings("ignore")


@dataclass
class Step:
    """Represents the state after a step in the workflow."""

    graph: NxMixedGraph
    estimand: Expression
    ace: float
    ace_delta: float
    direct_effect: float
    direct_effect_delta: float


def workflow(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Union[Variable, set[Variable]],
    outcomes: Union[Variable, set[Variable]],
    *,
    conditions: Union[None, Variable, set[Variable]] = None,
    ci_method: Optional[CITest] = None,
    ci_significance_level: Optional[float] = None,
    ace_bootstraps: int | None = None,
    ace_significance_level: float | None = None,
) -> list[Step]:
    """Run the Eliater workflow.

    This workflow has three parts:

    1. Add undirected edges between d-separated nodes for which a data-driven conditional independency test fails
    2. Remove nuisance variables.
    3. Estimates the average causal effect (ACE) of the treatments on outcomes

    :param graph: An acyclic directed mixed graph
    :param data: Data associated with nodes in the graph
    :param treatments: The node or nodes that are treated
    :param outcomes: The node or nodes that are outcomes
    :param conditions: Conditions on the query (currently not implemented for all parts)
    :param ci_method:
        The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data
        or :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param ci_significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :param ace_bootstraps: The number of bootstraps for calculating the ACE. Defaults to 0 (i.e., not used by default)
    :param ace_significance_level: The significance level for the ACE. Defaults to 0.05.
    :returns: A set of states after each step
    :raises ValueError: If the graph becomes unidentifiable throughout the workflow
    """
    treatments = _ensure_set(treatments)
    outcomes = _ensure_set(outcomes)

    def _estimate_ace(_graph: NxMixedGraph) -> float:
        return estimate_ace(
            graph=_graph,
            treatments=list(treatments),
            outcomes=list(outcomes),
            conditions=conditions,
            data=data,
            bootstraps=ace_bootstraps,
            alpha=ace_significance_level,
        )

    def _identify(_graph: NxMixedGraph) -> Expression:
        return identify_outcomes(
            _graph, treatments=treatments, outcomes=outcomes, conditions=conditions
        )

    def _get_direct_effect(_graph: NxMixedGraph) -> float:
        return get_eliater_regression(
            graph, treatment=list(treatments)[0], outcome=list(outcomes)[0], data=data
        )

    input_estimand = _identify(graph)
    if input_estimand is None:
        raise ValueError("input graph is not identifiable")
    input_ace = _estimate_ace(graph)
    input_direct_effect = _get_direct_effect(graph)
    initial = Step(
        graph=graph,
        estimand=input_estimand,
        ace=input_ace,
        ace_delta=0.0,
        direct_effect=input_direct_effect,
        direct_effect_delta=0.0,
    )

    graph_1 = add_ci_undirected_edges(
        graph, data, method=ci_method, significance_level=ci_significance_level
    )
    graph_1_estimand = _identify(graph_1)
    if graph_1_estimand is None:
        raise ValueError("not identifiable after adding CI edges")
    graph_1_ace = _estimate_ace(graph_1)
    graph_1_direct_effect = _get_direct_effect(graph_1)
    step_1 = Step(
        graph=graph_1,
        estimand=graph_1_estimand,
        ace=graph_1_ace,
        ace_delta=graph_1_ace - input_ace,
        direct_effect=graph_1_direct_effect,
        direct_effect_delta=graph_1_direct_effect - input_direct_effect,
    )

    graph_2 = remove_nuisance_variables(graph_1, treatments=treatments, outcomes=outcomes)
    graph_2_estimand = _identify(graph_2)
    if not graph_2_estimand:
        raise ValueError("not identifiable after removing nuisance variables")
    graph_2_ace = _estimate_ace(graph_2)
    graph_2_direct_effect = _get_direct_effect(graph_2)
    step_2 = Step(
        graph=graph_2,
        estimand=graph_2_estimand,
        ace=graph_2_ace,
        ace_delta=graph_2_ace - input_ace,
        direct_effect=graph_2_direct_effect,
        direct_effect_delta=graph_2_direct_effect - input_direct_effect,
    )

    return [initial, step_1, step_2]


@click.command()
def reproduce():
    """Run this function to generate the results for the paper."""
    click.echo("Make sure you're on the dev version of y0")
    rows = []
    columns = [
        "name",
        "treatments",
        "outcomes",
        "initial_nodes",
        "initial_estimand",
        "initial_ace",
        "initial_direct_effect",
        "step_1_nodes",
        "step_1_estimand",
        "step_1_ace",
        "step_1_ace_delta",
        "step_1_direct_effect",
        "step_1_direct_effect_delta",
        "step_2_nodes",
        "step_2_estimand",
        "step_2_ace",
        "step_2_ace_delta",
        "step_2_direct_effect",
        "step_2_direct_effect_delta",
    ]
    for example in examples:
        if example.data is not None:
            data = example.data
        elif example.generate_data is not None:
            data = example.generate_data(2000, seed=0)
        else:
            continue

        for query in example.example_queries:
            click.echo(f"\n> {example.name}")
            if len(query.treatments) != 1 or len(query.outcomes) != 1:
                click.echo(f"[{example.name}] skipping query:")
                continue

            try:
                steps = workflow(
                    graph=example.graph,
                    data=data,
                    treatments=query.treatments,
                    outcomes=query.outcomes,
                )
            except (ValueError, RuntimeError) as e:
                click.echo(f"Failed on query: {query.expression}")
                click.secho(f"{type(e).__name__}: {e}", fg="red")
                continue

            parts = []
            for i, step in enumerate(steps):
                parts.append(step.graph.directed.number_of_nodes())
                parts.append(step.estimand.to_y0())
                parts.append(round(step.ace, 4))
                if i > 0:
                    parts.append(round(step.ace_delta, 4))
                parts.append(round(step.direct_effect, 4))
                if i > 0:
                    parts.append(round(step.direct_effect_delta, 4))

            rows.append(
                (
                    example.name,
                    ", ".join(sorted(t.name for t in query.treatments)),
                    ", ".join(sorted(o.name for o in query.outcomes)),
                    *parts,
                )
            )
    if not rows:
        raise ValueError("No examples available!")
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(RESULTS_PATH, sep="\t", index=False)
    click.echo(f"\nOutputting {len(rows)} results to {RESULTS_PATH}")
    return df


if __name__ == "__main__":
    reproduce()
