import time
from operator import attrgetter
from textwrap import dedent
from typing import Optional

import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import trange

from eliater.discover_latent_nodes import find_nuisance_variables, remove_nuisance_variables
from eliater.network_validation import discretize_binary
from eliater.regression import estimate_query_by_linear_regression, get_adjustment_set
from y0.algorithm.estimation import estimate_ace
from y0.algorithm.falsification import get_graph_falsifications
from y0.algorithm.identify import identify_outcomes
from y0.dsl import Variable
from y0.examples import Example
from y0.graph import NxMixedGraph
from y0.struct import DEFAULT_SIGNIFICANCE, CITest, _ensure_method
from scipy.stats import ttest_1samp


def is_notebook() -> bool:
    """Check if currently in a notebook."""
    try:
        shell = get_ipython().__class__.__name__  # type:ignore
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def display_markdown(s: str) -> None:
    IPython.display.display(IPython.display.Markdown(dedent(s)))


def display_df(df: pd.DataFrame) -> None:
    html = df.reset_index(drop=True).to_html(index=False)
    IPython.display.display(IPython.display.HTML(html))


def step_1_notebook(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    *,
    binarize: bool = False,
    method: Optional[CITest] = None,
    max_given: Optional[int] = 5,
    significance_level: Optional[float] = None,
    show_all: bool = False,
    acceptable_percentage: float = 0.3,
    show_progress: bool = False,
):
    """Print the summary of conditional independency test results.

    Prints the summary to the console, which includes the total number of conditional independence tests,
    the number and percentage of failed tests, and statistical information about each test such as p-values,
    and test results.

    :param graph: an NxMixedGraph
    :param data: observational data corresponding to the graph
    :param method: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param max_given: The maximum set size in the power set of the vertices minus the d-separable pairs
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :param verbose: If `False`, only print the details of failed tests.
        If 'True', print the details of all the conditional independency results. Defaults to `False`
    :param tablefmt: The format for the table that gets printed. By default, uses RST, so it can be
        directly copy/pasted into Python documentation
    :param acceptable_percentage: The percentage of tests that need to fail to output an interpretation
        that additional edges should be added. Should be between 0 and 1.
    :param show_progress: If true, shows a progress bar for calculating d-separations
    :returns: If in Jupyter notebook, returns a dataframe. Otherwise, prints the dataframe.
    """
    display_markdown("## Step 1: Checking the ADMG Structure")
    if significance_level is None:
        significance_level = DEFAULT_SIGNIFICANCE
    if binarize:
        data = discretize_binary(data)
        display_markdown(
            "On this try, we're going to discretize the data using K-Bins discretization with K as 2."
            " Here are the first few rows of the transformed dataframe after doing that:"
        )
        display_df(data.head())

    start_time = time.time()
    method = _ensure_method(method, data)
    evidence_df = get_graph_falsifications(
        graph=graph,
        df=data,
        method=method,
        significance_level=significance_level,
        max_given=max_given,
        verbose=show_progress,
        given_sep=";",
    ).evidence
    end_time = time.time() - start_time
    time_text = f"Finished in {end_time:.2f} seconds."
    n_total = len(evidence_df)
    n_failed = evidence_df["p_adj_significant"].sum()
    percent_failed = n_failed / n_total
    if n_failed == 0:
        display_markdown(
            f"All {n_total} d-separations implied by the network's structure are consistent with the data, meaning "
            f"that none of the data-driven conditional independency tests' null hypotheses with the {method} test "
            f"were rejected at p<{significance_level}. {time_text}\n"
        )
    elif percent_failed < acceptable_percentage:
        display_markdown(  # noqa:T201
            f"Of the {n_total} d-separations implied by the network's structure, only {n_failed} "
            f"({percent_failed:.2%}) rejected the null hypothesis for the {method} test at p<{significance_level}."
            f"\n\nSince this is less than {acceptable_percentage:.0%}, Eliater considers this minor and leaves the "
            f"network unmodified. {time_text}\n"
        )
    else:
        display_markdown(
            f"Of the {n_total} d-separations implied by the network's structure, {n_failed} ({percent_failed:.2%}) "
            f"rejected the null hypothesis with the {method} test at p<{significance_level}.\n\nSince this is more "
            f"than {acceptable_percentage:.0%}, Eliater considers this a major inconsistency and therefore suggests "
            f"adding appropriate bidirected edges using the eliater.add_ci_undirected_edges() function. {time_text}\n"
        )
    if show_all:
        dd = evidence_df
    else:
        dd = evidence_df[evidence_df["p_adj_significant"]]

    display_markdown(dd.reset_index(drop=True).to_markdown(index=False))


def step_2_notebook(*, graph: NxMixedGraph, treatment: Variable, outcome: Variable):
    tlatex = treatment.to_latex()
    olatex = outcome.to_latex()
    introduction = dedent(
        f"""       
        ## Step 2: Check Query Identifiability
    
        The causal query of interest is the average treatment effect of ${tlatex}$ on ${olatex}$, defined as: 
        $\\mathbb{{E}}[{olatex} \\mid do({tlatex}=1)] - \\mathbb{{E}}[{olatex} \\mid do({tlatex}=0)]$.
    """
    )

    estimand = identify_outcomes(graph=graph, treatments=treatment, outcomes=outcome)
    if estimand is None:
        analysis = dedent(
            f"""\
        The query was not identifiable, so we can not proceed to Step 3.
        """
        )
    else:
        analysis = dedent(
            f"""\
        
        Running the ID algorithm defined by [Identification of joint interventional distributions in recursive
        semi-Markovian causal models](https://dl.acm.org/doi/10.5555/1597348.1597382) (Shpitser and Pearl, 2006)
        and implemented in the $Y_0$ Causal Reasoning Engine gives the following estimand:
        
        ${estimand.to_latex()}$

        Because the query is identifiable, we can proceed to Step 3.
        """
        )

    display_markdown(introduction + "\n" + analysis)


def step_3_notebook(
    *, graph: NxMixedGraph, treatment: Variable, outcome: Variable
) -> NxMixedGraph | None:
    display_markdown("## Step 3/4: Identify Nuisance Variables and Simplify the ADMG")
    nv = find_nuisance_variables(graph, treatments=treatment, outcomes=outcome)
    if not nv:
        display_markdown(
            f"""\
                No variables were identified as nuisance variables.
                
                Nevertheless, the algorithm proposed in [Graphs for margins of Bayesian
                networks](https://arxiv.org/abs/1408.1809) (Evans, 2016) and implemented in
                the $Y_0$ Causal Reasoning Engine is applied to the ADMG to attempt to
                simplify the graph by reasoning over its bidirected edges (if they exist).
            """
        )
    else:
        nv_text = ", ".join(f"${x.to_latex()}$" for x in sorted(nv, key=attrgetter("name")))

        display_markdown(
            f"""\
            The following {len(nv)} variables were identified as _nuisance_ variables,
            meaning that they appear as descendants of nodes appearing in paths between
            the treatment and outcome, but are not themselves ancestors of the outcome variable:
    
            {nv_text}
    
            These variables are marked as "latent", then
            the algorithm proposed in [Graphs for margins of Bayesian
            networks](https://arxiv.org/abs/1408.1809) (Evans, 2016) and implemented in
            the $Y_0$ Causal Reasoning Engine is applied to the ADMG to
            simplify the graph. This minimally removes the latent variables and makes
            further simplifications if the latent variables are connected by bidirected
            edges to other nodes.
        """
        )

    new_graph = remove_nuisance_variables(graph, treatments=treatment, outcomes=outcome)
    if new_graph == graph:
        display_markdown("The simplification did not modify the graph.")
        return None
    new_graph.draw()
    return new_graph


def step_5_notebook_synthetic(
    *,
    graph: NxMixedGraph,
    reduced_graph: NxMixedGraph | None,
    example: Example,
    treatment: Variable,
    outcome: Variable,
    seed: int = 42,
    samples: int = 10_000,
    n_subsamples: int = 500,
    subsample_size: int = 1_000,
    eps: float = 1e-10,
):
    tlatex = treatment.to_latex()
    olatex = outcome.to_latex()

    data_obs = example.generate_data(samples, seed=seed)
    data_1 = example.generate_data(samples, {treatment: 1.0}, seed=seed)
    data_0 = example.generate_data(samples, {treatment: 0.0}, seed=seed)

    ates = []
    for _ in range(n_subsamples):
        idx = np.random.permutation(samples)[:subsample_size]
        data_1_mean = data_1.loc[idx][outcome.name].mean()
        data_0_mean = data_0.loc[idx][outcome.name].mean()
        diff = data_1_mean - data_0_mean
        ates.append(diff)

    ate = np.mean(ates)
    ate_var = np.var(ates)

    display_markdown(
        f"""\
    ## Step 5: Estimate the Query
    
    ### Calculating the True Average Treatment Effect (ATE)
    
    We first generated synthetic observational data. Now, we generate two interventional datasets:
    one where we set ${treatment.to_latex()}$ to $0.0$ and one where we set ${treatment.to_latex()}$ to $1.0$.
    We can then calculate the "true" average treatment effect (ATE) as the difference of the means
    for the outcome variable ${outcome.to_latex()}$ in each. The ATE is formulated as:
 
    $ATE = \\mathbb{{E}}[{olatex} \\mid do({tlatex} = 1)] - \\mathbb{{E}}[{olatex} \\mid do({tlatex} = 0)]$
    
    After generating {samples:,} samples for each distribution, we took {n_subsamples:,} subsamples of size
    of size {subsample_size:,} and calculated the
    ATE for each. The variance comes to {ate_var:.1e}, which shows that the ATE is very stable with respect
    to random generation. We therefore calculate the _true_ ATE as the average value from these samplings,
    which comes to {ate:.1e}.

    The ATE can be interpreted in the following way:

    1. If the ATE is positive, it suggests that the treatment ${tlatex}$ has a negative effect on the outcome ${olatex}$
    2. If the ATE is negative, it suggests that the treatment ${tlatex}$ has a positive effect on the outcome ${olatex}$

    **Caveat**: Eliater does not yet implement a notion of confidence for the ATE. For example, it's not clear
    where the cutoff for _significance_ is, and whether that is dataset- or ADMG-dependent.
    
    ### Estimating the Average Treatment Effect (ATE)
    
    In practice, we are often unable to get the appropriate interventional data, and therefore want to estimate
    the average treatment effect (ATE) from observational data. Because we're using synthetic data, we generate
    {samples:,} samples, then took {n_subsamples:,} subsamples of size {subsample_size:,} through which we calculated
    the following:

    1. The ATE, using the y0/ananke implementation
    2. The ATE, using the Eliater linear regression implementation
    """
    )

    ananke_ace_reference = []
    ananke_ace_reduced = []
    linreg_ace_reference = []
    linreg_ace_reduced = []
    ev_reference = []
    ev_reduced = []

    reference_adjustment_set, _ = get_adjustment_set(
        graph=graph, treatments=treatment, outcome=outcome
    )
    if reduced_graph is not None:
        reduced_adjustment_set, _ = get_adjustment_set(
            graph=reduced_graph, treatments=treatment, outcome=outcome
        )

    for _ in trange(n_subsamples, leave=False, desc="Analyzing w/ subsampling", unit="sample"):
        data_obs_sample = data_obs.sample(subsample_size)
        ananke_ace_reference.append(
            estimate_ace(graph, treatments=treatment, outcomes=outcome, data=data_obs_sample)
        )

        linreg_ace_reference.append(
            estimate_query_by_linear_regression(
                graph,
                treatments=treatment,
                outcome=outcome,
                data=data_obs_sample,
                query_type="ate",
                _adjustment_set=reference_adjustment_set,
            )
        )
        ev_reference.append(
            estimate_query_by_linear_regression(
                graph,
                data=data_obs_sample,
                treatments=treatment,
                outcome=outcome,
                interventions={treatment: 0},
                query_type="expected_value",
                _adjustment_set=reference_adjustment_set,
            )
        )
        if reduced_graph is not None:
            ananke_ace_reduced.append(
                estimate_ace(
                    reduced_graph, treatments=treatment, outcomes=outcome, data=data_obs_sample
                )
            )
            linreg_ace_reduced.append(
                estimate_query_by_linear_regression(
                    reduced_graph,
                    treatments=treatment,
                    outcome=outcome,
                    data=data_obs_sample,
                    query_type="ate",
                    _adjustment_set=reduced_adjustment_set,
                )
            )
            ev_reduced.append(
                estimate_query_by_linear_regression(
                    reduced_graph,
                    data=data_obs_sample,
                    treatments=treatment,
                    outcome=outcome,
                    interventions={treatment: 0},
                    query_type="expected_value",
                    _adjustment_set=reduced_adjustment_set,
                )
            )

    ananke_ace_reference_var = np.var(ananke_ace_reference)
    linreg_ace_reference_var = np.var(linreg_ace_reference)
    ev_reference_var = np.var(ev_reference)

    # Check that the p-values are significantly different from zero to say
    # if the ATE is significant (then after you can use the sign)
    _, ananke_ace_reference_p_value = ttest_1samp(ananke_ace_reference, 0, alternative="two-sided")
    _, linreg_ace_reference_p_value = ttest_1samp(linreg_ace_reference, 0, alternative="two-sided")

    if reduced_graph is not None:
        ananke_ace_reduced_var = np.var(ananke_ace_reduced)
        ananke_ace_diffs = [a - b for a, b in zip(ananke_ace_reference, ananke_ace_reduced)]

        linreg_ace_reduced_var = np.var(linreg_ace_reduced)
        linreg_ace_diffs = [a - b for a, b in zip(linreg_ace_reference, linreg_ace_reduced)]

        ev_reduced_var = np.var(ev_reduced)
        ev_diffs = [a - b for a, b in zip(ev_reference, ev_reduced)]

        fig, axes = plt.subplots(2, 3, figsize=(14, 6.5))

        sns.histplot(ananke_ace_reference, ax=axes[0][0])
        axes[0][0].set_title(
            f"ATEs on Original ADMG\nVariance: {ananke_ace_reference_var:.1e}, $p={ananke_ace_reference_p_value:.2e}$"
        )
        axes[0][0].set_xlabel("ATE from y0.algorithm.estimation.estimate_ace")
        axes[0][0].axvline(ate, color="red")
        sns.histplot(ananke_ace_reduced, ax=axes[0][1])
        axes[0][1].set_title(f"ATEs on Reduced ADMG\nVariance: {ananke_ace_reduced_var:.1e}")
        axes[0][1].set_xlabel("ATE from y0.algorithm.estimation.estimate_ace")
        axes[0][1].set_ylabel("")
        axes[0][1].axvline(ate, color="red")
        sns.histplot(ananke_ace_diffs, ax=axes[0][2])
        axes[0][2].set_xlabel("Reduced ADMG - Original ADMG")
        axes[0][2].set_ylabel("")
        axes[0][2].set_title(_diff_subtitle(ananke_ace_diffs, eps))

        sns.histplot(linreg_ace_reference, ax=axes[1][0])
        axes[1][0].set_title(
            f"ATEs on Original ADMG\nVariance: {linreg_ace_reference_var:.1e}, $p={linreg_ace_reference_p_value:.2e}$"
        )
        axes[1][0].set_xlabel("ATE from eliater.estimate_query")
        axes[1][0].axvline(ate, color="red")
        sns.histplot(linreg_ace_reduced, ax=axes[1][1])
        axes[1][1].set_title(f"ATEs on Reduced ADMG\nVariance: {linreg_ace_reduced_var:.1e}")
        axes[1][1].set_xlabel("ATE from eliater.estimate_query")
        axes[1][1].set_ylabel("")
        axes[1][1].axvline(ate, color="red")
        sns.histplot(linreg_ace_diffs, ax=axes[1][2])
        axes[1][2].set_xlabel("Reduced ADMG - Original ADMG")
        axes[1][2].set_ylabel("")
        axes[1][2].set_title(_diff_subtitle(linreg_ace_diffs, eps))

    else:
        fig, axes = plt.subplots(1, 2, figsize=(8, 3))

        sns.histplot(ananke_ace_reference, ax=axes[0])
        axes[0].set_title(
            f"ATEs on Original ADMG\nVariance: {ananke_ace_reference_var:.1e}, $p={ananke_ace_reference_p_value:.2e}$"
        )
        axes[0].set_xlabel("ATE from y0.algorithm.estimation.estimate_ace")
        axes[0].axvline(ate, color="red")
        sns.histplot(ananke_ace_reduced, ax=axes[0])

        sns.histplot(linreg_ace_reference, ax=axes[1])
        axes[1].set_title(
            f"ATEs on Original ADMG\nVariance: {linreg_ace_reference_var:.1e}, $p={linreg_ace_reference_p_value:.2e}$"
        )
        axes[1].set_xlabel("ATE from eliater.estimate_query")
        axes[1].axvline(ate, color="red")
        sns.histplot(linreg_ace_reduced, ax=axes[1])

    plt.tight_layout()
    plt.show()

    display_markdown(
        f"""\
    Interpretations:
    
    1. We show the _true_ ATE as a red vertical line
    2. We show both this process done with the original ADMG and the reduced ADMG. This shows that the reduction
       on the ADMG does not affect estimation. However, reduction is still valuable for simplifying visual exploration.
    """
    )

    threshold = 0.01
    _interpret_p(
        p_value=ananke_ace_reference_p_value,
        threshold=threshold,
        treatment=treatment,
        outcome=outcome,
        distribution=ananke_ace_reference,
        label="ananke/y0 estimation of the ACE",
    )
    _interpret_p(
        p_value=linreg_ace_reference_p_value,
        threshold=threshold,
        treatment=treatment,
        outcome=outcome,
        distribution=linreg_ace_reference,
        label="Eliater linear regression estimation of the ACE",
    )

    display_markdown(
        f"""\
    ### Estimating the Expected Value

    We now estimate the query in the form of the expected value:

    $\\mathbb{{E}}[{olatex} \\mid do({tlatex} = 0)]$
    """
    )

    if reduced_graph is not None:
        fig, axes = plt.subplots(1, 3, figsize=(14, 3))

        sns.histplot(ev_reference, ax=axes[0])
        axes[0].set_title(
            f"$E[{olatex} \\mid do({tlatex} = 0)]$ on Original ADMG\nVariance: {ev_reference_var:.1e}"
        )
        axes[0].set_xlabel(f"$E[{olatex} \\mid do({tlatex} = 0)]$ from eliater.estimate_query")
        sns.histplot(ev_reduced, ax=axes[1])
        axes[1].set_title(
            f"$E[{olatex} \\mid do({tlatex} = 0)]$ on Reduced ADMG\nVariance: {ev_reduced_var:.1e}"
        )
        axes[1].set_xlabel(f"$E[{olatex} \\mid do({tlatex} = 0)]$ from eliater.estimate_query")
        axes[1].set_ylabel("")
        sns.histplot(ev_diffs, ax=axes[2])
        axes[2].set_xlabel("Reduced ADMG - Original ADMG")
        axes[2].set_ylabel("")
        axes[2].set_title(_diff_subtitle(ev_diffs, eps))
        plt.tight_layout()
    else:
        fig, axis = plt.subplots(1, 1, figsize=(5, 3))

        sns.histplot(ev_reference, ax=axis)
        axis.set_title(
            f"$E[{olatex} \\mid do({tlatex} = 0)]$ on Original ADMG\nVariance: {ev_reference_var:.1e}"
        )
        axis.set_xlabel(f"$E[{olatex} \\mid do({tlatex} = 0)]$ from eliater.estimate_query")
        # plt.tight_layout()

    plt.show()

    display_markdown(
        """\
        **Caveat**: Eliater does not yet have an automated explanation of what the results of this analysis mean.
    """
    )


def _interpret_p(p_value, threshold, treatment, outcome, distribution, label):
    if p_value < threshold:
        # note that the interpretation is opposite of the sign
        direction = "negative" if np.mean(distribution) > 0 else "positive"
        display_markdown(
            f"""\
            The p-value for the {label} is {p_value:.2e},
            which is below the significance threshold of {threshold}. Therefore, we reject the
            null hypothesis of the 1 Sample T-test and conclude that the distribution is significantly
            different from zero. This means that the treatment ${treatment.to_latex()}$ has 
            *a {direction} effect* on the outcome ${outcome.to_latex()}$.
        """
        )
    else:
        display_markdown(
            f"""\
            The p-value for the {label} is {p_value:.2e},
            which is *not( the significance threshold of {threshold}. Therefore, we do not reject the
            null hypothesis of the 1 Sample T-test and conclude that the distribution is not significantly
            different from zero. This means that the treatment ${treatment.to_latex()}$ has *no significant effect*
            on the outcome ${outcome.to_latex()}$.
            """
        )


def _diff_subtitle(diffs, eps):
    if all(diff < eps for diff in diffs):
        return f"Pairwise Differences\nAll $< {eps:.1e}$ (i.e., artifacts of floating pt. math)"
    else:
        return "Pairwise Differences\nSome are significant"
