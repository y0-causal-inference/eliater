"""Implementation of Eliater workflow."""

from pathlib import Path
from typing import Optional, Union

import click
import pandas as pd

from eliater.discover_latent_nodes import remove_nuisance_variables
from eliater.examples import examples
from eliater.network_validation import add_ci_undirected_edges
from y0.algorithm.estimation import estimate_ace
from y0.algorithm.identify import identify_outcomes
from y0.dsl import Variable
from y0.graph import NxMixedGraph, _ensure_set
from y0.struct import CITest

__all__ = [
    "workflow",
    "reproduce",
]

HERE = Path(__file__).parent.resolve()
RESULTS_PATH = HERE.joinpath("case_studies.tsv")


def workflow(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    treatments: Union[Variable, set[Variable]],
    outcomes: Union[Variable, set[Variable]],
    *,
    ci_method: Optional[CITest] = None,
    ci_significance_level: Optional[float] = None,
    ace_bootstraps: int | None = None,
    ace_significance_level: float | None = None,
):
    """Run the Eliater workflow.

    This workflow has three parts:

    1. Add undirected edges between d-separated nodes for which a data-driven conditional independency test fails
    2. Remove nuisance variables.
    3. Estimates the average causal effect (ACE) of the treatments on outcomes

    :param graph: An acyclic directed mixed graph
    :param data: Data associated with nodes in the graph
    :param treatments: The node or nodes that are treated
    :param outcomes: The node or nodes that are outcomes
    :param ci_method:
        The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data
        or :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param ci_significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :param ace_bootstraps: The number of bootstraps for calculating the ACE. Defaults to 0 (i.e., not used by default)
    :param ace_significance_level: The significance level for the ACE. Defaults to 0.05.
    :returns: A triple with a modified graph, the estimand, and the ACE value.
    :raises ValueError: If the graph becomes unidentifiable throughout the workflow
    """
    treatments = _ensure_set(treatments)
    outcomes = _ensure_set(outcomes)

    def _estimate_ace(_graph):
        return estimate_ace(
            graph=_graph,
            treatments=list(treatments),
            outcomes=list(outcomes),
            data=data,
            bootstraps=ace_bootstraps,
            alpha=ace_significance_level,
        )

    input_estimand = identify_outcomes(graph, treatments=treatments, outcomes=outcomes)
    if input_estimand is None:
        raise ValueError("input graph is not identifiable")
    input_ace = _estimate_ace(graph)

    graph_1 = add_ci_undirected_edges(
        graph, data, method=ci_method, significance_level=ci_significance_level
    )
    graph_1_estimand = identify_outcomes(graph_1, treatments=treatments, outcomes=outcomes)
    if graph_1_estimand is None:
        raise ValueError("not identifiable after adding CI edges")
    graph_1_ace = _estimate_ace(graph_1)
    graph_1_ace_delta = graph_1_ace - input_ace

    # TODO extend this to consider condition variables
    graph_2 = remove_nuisance_variables(graph_1, treatments=treatments, outcomes=outcomes)
    graph_2_estimand = identify_outcomes(graph_2, treatments=treatments, outcomes=outcomes)
    if not graph_2_estimand:
        raise ValueError("not identifiable after removing nuisance variables")
    graph_2_ace = _estimate_ace(graph_2)
    graph_2_ace_delta = graph_2_ace - input_ace

    return (
        input_estimand,
        input_ace,
        graph_1,
        graph_1_estimand,
        graph_1_ace,
        graph_1_ace_delta,
        graph_2,
        graph_2_estimand,
        graph_2_ace,
        graph_2_ace_delta,
    )


def reproduce():
    """Run this function to generate the results for the paper."""
    click.echo("Make sure you're on the dev version of y0")
    rows = []
    for example in examples:
        if example.data is None:
            continue
        for query in example.example_queries:
            if len(query.treatments) != 1 or len(query.outcomes) != 1:
                click.echo(f"[{example.name}] skipping query:")
                continue

            try:
                record = workflow(
                    graph=example.graph,
                    data=example.data,
                    treatments=query.treatments,
                    outcomes=query.outcomes,
                )
            except Exception as e:
                click.echo(f"[{example.name}] failed on query: {query.expression}")
                click.secho(str(e), fg="red")
                continue
            rows.append(
                (
                    example.name,
                    ", ".join(sorted(t.name for t in query.treatments)),
                    ", ".join(sorted(o.name for o in query.outcomes)),
                    *record,
                )
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_PATH, sep="\t", index=False)
    return df


if __name__ == "__main__":
    reproduce()
