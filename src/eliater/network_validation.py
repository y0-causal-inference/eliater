"""This module checks the validity of network structure against observational data.

Given an acyclic directed mixed graph (ADMG) and corresponding observational data,
one can assess whether the conditional independences implied by the structure of the
ADMG are supported by the data with a statistical test for conditional independence.
By default, this workflow uses a chi-square test for discrete data and a Pearson test
for continuous data from :mod:`pgmpy.estimators.CITests`.

This module provides a summary statistics for the total number of tests, percentage of failed
tests, and a list of all (or the failed tests) with their corresponding p-value.

This process allows for checking if the network structure is consistent with the data by checking
the percentage of failed tests. If the percentage of failed tests is lower than 30 percent, the effect
that the inconsistency between the structure and the data may have on causal query estimation is minor.
However, if the percentage of failed tests is larger than 30 percent, we recommend the user to revise
the network structure or the corresponding data.

.. warning:: This functionality is not unit tested! Use at your own risk.

Example
-------
We'll work with :data:`eliater.examples.example_2` and take $X$ as the treatment, $Y$ as the outcome.
The example includes a function for simulating observational data.

.. image:: img/multiple_mediators_with_multiple_confounders.png
  :width: 200px

.. code-block:: python

    from eliater.examples import example_2
    from eliater.network_validation import print_graph_falsifications

    graph = example_2.graph
    observational_df = example_2.generate_data(500, seed=1)

    print_graph_falsifications(graph, data=observational_df, verbose=True)

======  =======  =======  ===========  ===========  =====  ===========  ===================
left    right    given          stats            p  dof          p_adj  p_adj_significant
======  =======  =======  ===========  ===========  =====  ===========  ===================
Y       Z2       M2|Z3     0.392472    7.33647e-20         1.02711e-18  True
M2      Z3       X         0.0887728   0.047259            0.614367     False
M2      Z2       X         0.0874659   0.0506246           0.614367     False
X       Z3       Z1        0.0097293   0.828197            1            False
Z1      Z3       Z2        0.0700012   0.117988            1            False
X       Y        M2|Z3    -0.00485697  0.913731            1            False
X       Z2       Z1       -0.0109544   0.806966            1            False
M1      Y        M2|Z3     0.0124796   0.780733            1            False
M2      Z1       X         0.0697175   0.119489            1            False
Y       Z1       M2|Z3     0.0169804   0.704858            1            False
M2      X        M1        0.0435465   0.331173            1            False
M1      Z3       X         0.0664571   0.137823            1            False
M1      Z1       X        -0.0108138   0.809395            1            False
M1      Z2       X         0.0372645   0.405713            1            False
======  =======  =======  ===========  ===========  =====  ===========  ===================

The results show that out of 14 cases, 1 failed. The failed test is
the conditional independence between $Y$ and $Z_2$, given $M_2$ and $Z_3$.

Finding False Negatives
-----------------------
This module relies on statistical tests, and statistical tests have chances
of producing false negatives, i.e., a pair of variables that are conditionally
independent, be concluded as conditional dependent by the test, or producing false
positives, i.e., a pair of variables that are conditionally dependent be concluded
as conditionally independent by the test. The main reason that the result of the test
may be false negative or false positive is that statistical tests rely on *p*-values.
The p-values are subject to known limitations in statistical analysis [halsey2015fickle]_
and [wang2022addressing]_. In particular, the p-value of a data-driven conditional
independency test decrease as the number of data points increases [lucas2013too]_.

This is described below using the following example graph, and observational data
(simulated specifically for this graph using
:func:`eliater.frontdoor_backdoor.example2.generate`),
and the application of subsampling.

.. image:: img/multiple_mediators_with_multiple_confounders.png
  :width: 200px

.. warning::

   This part is implemented based on Chapter 4 from
   https://livebook.manning.com/book/causal-ai/welcome/v-4/, however
   this resource is paywalled.

In this graph, $Y$ is conditionally independent (i.e., D-separated) of $M_1$ given $M_2$, $Z_2$.
The data has been generated based on this assumption. Hence, we expect the $p$-value to be above
0.05, i.e., not rejecting the null hypothesis of conditional independence. We use the following
workflow to graphically assess how this compares to a data-driven approach.

.. code-block:: python

   import matplotlib.pyplot as plt
   from matplotlib_inline.backend_inline import set_matplotlib_formats

   from y0.graph import NxMixedGraph
   from eliater.examples import example_2
   from eliater import plot_ci_size_dependence

   graph = example_2.graph

   stop = 2_000
   # Generate observational data for this graph (this is a special example)
   observational_df = example_2.generate_data(stop, seed=1)

   plot_ci_size_dependence(
       observational_df,
       start=50,
       stop=stop,
       step=100,
       left="Y",
       right="M1",
       conditions=["M2", "Z2"],
   )
   plt.savefig("pvalue_vs_sample_size.svg")


.. image:: img/pvalue_vs_sample_size.svg
  :width: 350px
  :height: 250px
  :scale: 200 %
  :alt: alternate text
  :align: right

This plot shows that the expected $p$-value will decrease as number of data points increases. The error bars are 90%
bootstrap confidence intervals. The horizontal dashed line is a 0.5 significance level. The p-values above this
threshold show that the test favors the null hypothesis of conditional independence. For number of data points
greater than 1,000, the test is more likely to reject the null hypothesis, and for number of data points greater
than 1,250, the test always rejects the null hypothesis, i.e., the data will no longer support that $Y$ is
independent of $M_1$ given $M_2$, and $Z_2$ where it should be. Hence, the result of network validation depends
on the size of the data. This result may seem disappointing because more data can lead to inaccurate results,
however, regardless of the data size and the significance thresholds, the relative differences between $p$-values
when there is no conditional independence and whe there is will be large and easy to detect.

As a result, the results obtained from this module should be regarded more as heuristics approach
and as an indication of patterns in data as opposed to statement of ground truth and should be interpreted with caution.
However, we recommend that if the percentage of failed tests is small (e.g., smaller than 30 percent), then that impact
of inconsistency between network structure and data is minor in that causal query estimation. Hence, it is safe to
proceed with the estimation procedure.
If the percentage of failed tests is large (greater than 30-40 percent), it indicates that the input network does not
reflect the underlying data generation process, and the network or the data should be revised. Causal structure learning
algorithms, for examples the ones implemented in :mod:`pgmpy`
(see `here <https://pgmpy.org/examples/Structure%20Learning%20in%20Bayesian%20Networks.html>`_)
can be used to revise the network structure and align it with data. This module currently does not repair the
structure of the network if the network is not aligned with data according to conditional independence tests.

For more reference on this topic, please see
chapter 4 of https://livebook.manning.com/book/causal-ai/welcome/v-4/.

.. [halsey2015fickle] Halsey, Lewis G., et al. "The fickle P value generates irreproducible results.
   " Nature methods 12.3 (2015): 179-185.

.. [wang2022addressing] Wang, Ming, and Qi Long. "Addressing Common Misuses and Pitfalls of p values
   in Biomedical Research." Cancer research 82.15 (2022): 2674-2677.

.. [lucas2013too] Lucas, H., and G. Shmueli. "Too big to fail: large samples and the p-value problem."
   Inf. Syst. Res. 24.4 (2013): 906-917.
"""

import time
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import mean, quantile
from sklearn.preprocessing import KBinsDiscretizer
from tabulate import tabulate
from tqdm.auto import trange

import y0.algorithm.conditional_independencies
from y0.algorithm.falsification import get_graph_falsifications
from y0.graph import NxMixedGraph
from y0.struct import (
    DEFAULT_SIGNIFICANCE,
    CITest,
    _ensure_method,
    get_conditional_independence_tests,
)

__all__ = [
    "discretize_binary",
    "plot_treatment_and_outcome",
    "add_ci_undirected_edges",
    "p_value_of_bootstrap_data",
    "p_value_statistics",
    "plot_ci_size_dependence",
]

TESTS = get_conditional_independence_tests()


def plot_treatment_and_outcome(data, treatment, outcome, figsize=(8, 2.5)) -> None:
    """Plot the treatment and outcome histograms."""
    fig, (lax, rax) = plt.subplots(1, 2, figsize=figsize)
    sns.histplot(data=data, x=treatment.name, ax=lax)
    lax.axvline(data[treatment.name].mean(), color="red")
    lax.set_title("Treatment")

    sns.histplot(data=data, x=outcome.name, ax=rax)
    rax.axvline(data[outcome.name].mean(), color="red")
    rax.set_ylabel("")
    rax.set_title("Outcome")


def discretize_binary(data: pd.DataFrame) -> pd.DataFrame:
    """Discretize continuous data into binary data using K-Bins Discretization."""
    kbins = KBinsDiscretizer(n_bins=2, encode="ordinal", strategy="uniform")
    return pd.DataFrame(kbins.fit_transform(data), columns=data.columns)


def add_ci_undirected_edges(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    method: Optional[CITest] = None,
    significance_level: Optional[float] = None,
) -> NxMixedGraph:
    """Add undirected edges between d-separated nodes that fail a data-driven conditional independency test.

    :param graph: An acyclic directed mixed graph
    :param data: observational data corresponding to the graph
    :param method:
        The conditional independency test to use. If None, defaults to
        :data:`y0.struct.DEFAULT_CONTINUOUS_CI_TEST` for continuous data
        or :data:`y0.struct.DEFAULT_DISCRETE_CI_TEST` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.05.
    :returns: A copy of the input graph potentially with new undirected edges added
    """
    warnings.warn(
        "This method has been replaced by a refactored implementation in "
        "y0.algorithm.conditional_independencies.add_ci_undirected_edges",
        DeprecationWarning,
        stacklevel=1,
    )
    return y0.algorithm.conditional_independencies.add_ci_undirected_edges(
        graph=graph, data=data, method=method, significance_level=significance_level
    )


def print_graph_falsifications(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    method: Optional[CITest] = None,
    max_given: Optional[int] = 5,
    significance_level: Optional[float] = None,
    verbose: Optional[bool] = False,
    tablefmt: str = "rst",
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
    if significance_level is None:
        significance_level = DEFAULT_SIGNIFICANCE
    start_time = time.time()
    evidence_df = get_graph_falsifications(
        graph=graph,
        df=data,
        method=method,
        significance_level=significance_level,
        max_given=max_given,
        verbose=show_progress,
    ).evidence
    end_time = time.time() - start_time
    time_text = f"Finished in {end_time:.2f} seconds."
    n_total = len(evidence_df)
    n_failed = evidence_df["p_adj_significant"].sum()
    percent_failed = n_failed / n_total
    if n_failed == 0:
        print(  # noqa:T201
            f"All {n_total} d-separations implied by the network's structure are consistent with the data, meaning "
            f"that none of the data-driven conditional independency tests' null hypotheses were rejected "
            f"at p<{significance_level}.\n\n{time_text}\n"
        )
    elif percent_failed < acceptable_percentage:
        print(  # noqa:T201
            f"Of the {n_total} d-separations implied by the network's structure, only {n_failed}"
            f"({percent_failed:.2%}) rejected the null hypothesis at p<{significance_level}.\n\nSince this is less "
            f"than {acceptable_percentage:.0%}, Eliater considers this minor and leaves the network unmodified.]"
            f"\n\n{time_text}\n"
        )
    else:
        print(  # noqa:T201
            f"Of the {n_total} d-separations implied by the network's structure, {n_failed} ({percent_failed:.2%}) "
            f"rejected the null hypothesis at p<{significance_level}.\n\nSince this is more than "
            f"{acceptable_percentage:.0%}, Eliater considers this a major inconsistency and therefore suggests adding "
            f"appropriate bidirected edges using the eliater.add_ci_undirected_edges() function.\n\n{time_text}\n"
        )
    if verbose:
        dd = evidence_df
    else:
        dd = evidence_df[evidence_df["p_adj_significant"]]
    if _is_notebook():
        return dd.reset_index(drop=True)
    else:
        print(  # noqa:T201
            tabulate(dd, headers=list(dd.columns), tablefmt=tablefmt, showindex=False)
        )


def _is_notebook() -> bool:
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


def p_value_of_bootstrap_data(
    df: pd.DataFrame,
    sample_size: int,
    left: str,
    right: str,
    conditions: list[str],
    *,
    test: Optional[CITest] = None,
    significance_level: Optional[float] = None,
) -> int:
    """Calculate the p-value for a bootstrap data.

    :param df: observational data
    :param sample_size: number of data points to sample a bootstrap data from full_data
    :param left: first variable name positioned on the left side of a conditional independence test
    :param right: second variable name positioned on the right side of a conditional independence test
    :param conditions: variables names to condition on in the conditional independence test
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.05.
    :return: the $p$-value for the given bootstrap data
    """
    if significance_level is None:
        significance_level = 0.05
    test = _ensure_method(method=test, df=df)
    bootstrap_data = df.sample(n=sample_size, replace=True)
    result = TESTS[test](
        X=left,
        Y=right,
        Z=conditions,
        data=bootstrap_data,
        boolean=False,
        significance_level=significance_level,
    )
    p_val = result[1]
    return p_val


def p_value_statistics(
    df: pd.DataFrame,
    sample_size: int,
    left: str,
    right: str,
    conditions: list[str],
    *,
    test: Optional[CITest] = None,
    significance_level: Optional[float] = None,
    boot_size: Optional[int] = None,
):
    """Calculate mean of p-value, the 5th percentile and 95th percentile error, for several bootstrap data.

    :param df: A dataframe containing observational data
    :param sample_size: number of data points to sample a bootstrap data from the dataframe
    :param left: first variable name positioned on the left side of a conditional independence test
    :param right: second variable name positioned on the right side of a conditional independence test
    :param conditions: variables names to condition on in the conditional independence test
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.05.
    :param boot_size: total number of times a bootstrap data is sampled
    :return: the mean of p-value, the 5th percentile and 95 percentile error, for several bootstrap data
    """
    if boot_size is None:
        boot_size = 1_000
    test = _ensure_method(method=test, df=df)
    samples = [
        p_value_of_bootstrap_data(
            df,
            sample_size,
            left,
            right,
            conditions,
            test=test,
            significance_level=significance_level,
        )
        for _ in trange(
            boot_size, desc="Bootstrapping", leave=False, unit_scale=True, unit="bootstrap"
        )
    ]
    p_val = mean(samples)  # Calculate the mean of the p-values to get the bootstrap mean.
    quantile_05, quantile_95 = quantile(samples, q=[0.05, 0.95])
    lower_error = np.absolute(p_val - quantile_05)  # Calculate the 5th percentile
    higher_error = np.absolute(quantile_95 - p_val)  # Calculate the 95th percentile
    return p_val, lower_error, higher_error


def plot_ci_size_dependence(
    df: pd.DataFrame,
    start: int,
    stop: int,
    step: int,
    left: str,
    right: str,
    conditions: list[str],
    *,
    test: Optional[CITest] = None,
    significance_level: Optional[float] = None,
    boot_size: Optional[int] = None,
):
    """Generate the plot of expected p-value versus number of data points.

    :param df: observational data
    :param start: minimum number of data points to sample from df
    :param stop: maximum number of data points to sample from df
    :param step: minimum number of sampled data points increments by step number, and stops
        before maximum number of sampled data points
    :param left: first variable name positioned on the left side of a conditional independence test
    :param right: second variable name positioned on the right side of a conditional independence test
    :param conditions: variables names to condition on in the conditional independence test
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param significance_level: The statistical tests employ this value for comparison with the p-value of the test
        to determine the independence of the tested variables. If none, defaults to 0.05.
    :param boot_size: total number of times a bootstrap data is sampled
    """
    if significance_level is None:
        significance_level = 0.05
    p_vals, lower_errors, higher_errors = zip(
        *[
            p_value_statistics(
                df=df,
                sample_size=size,
                left=left,
                right=right,
                conditions=conditions,
                test=test,
                significance_level=significance_level,
                boot_size=boot_size,
            )
            for size in trange(start, stop, step, desc="Sampling")
        ]
    )

    if len(conditions) < 1:
        plt.title(f"Independence of {left} and {right}")
    else:
        conditions_string = ", ".join(conditions)
        plt.title(f"Independence of {left} and {right} given {conditions_string}")

    plt.xlabel("Data Points")
    plt.ylabel("Expected p-Value")

    sns.lineplot(x=list(range(start, stop, step)), y=p_vals)

    plt.errorbar(
        list(range(start, stop, step)),
        p_vals,
        yerr=np.array([lower_errors, higher_errors]),
        ecolor="grey",
        elinewidth=0.5,
        fmt="none",
    )
    plt.hlines(significance_level, 0, stop, linestyles="dashed")
