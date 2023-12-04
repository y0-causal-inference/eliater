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

T Cell Signalling Example
-------------------------
Here is an example of a protein signalling network of the T cell signaling pathway presented
in [Sachs2005]_. It models the molecular mechanisms and regulatory processes involved
in T cell activation, proliferation, and function.

The data consist of simultaneous measurements of 11 phosphorylated proteins and phospholipids
derived from thousands of individual primary human immune system cells.

.. figure:: img/signaling.png
   :width: 200px
   :height: 200px
   :scale: 150 %

.. code-block:: python

    from y0.graph import NxMixedGraph
    from eliater.data import load_sachs_df
    from eliater.network_validation import conditional_independence_test_summary

    graph = NxMixedGraph.from_str_adj(
        directed={
            "PKA": ["Raf", "Mek", "Erk", "Akt", "Jnk", "P38"],
            "PKC": ["Mek", "Raf", "PKA", "Jnk", "P38"],
            "Raf": ["Mek"],
            "Mek": ["Erk"],
            "Erk": ["Akt"],
            "Plcg": ["PKC", "PIP2", "PIP3"],
            "PIP3": ["PIP2", "Akt"],
            "PIP2": ["PKC"],
        }
    )

    data = load_sachs_df()

    # Prints out the conditional independence test results into the console
    conditional_independence_test_summary(graph, data, verbose=True)

.. image:: img/sachs_table.png
   :width: 200px
   :height: 400px
   :scale: 150 %
   :alt: alternate text
   :align: right

The results show that out of 35 cases, 1 failed. The failed test is
the conditional independence between P38 and PIP2, given PKC, with a p-value of 0.00425.

.. todo:: link all mentions of biological entities to HGNC or UniProt pages

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
   from eliater.frontdoor_backdoor.example2 import generate
   from eliater.network_validation import generate_plot_expected_p_value_vs_num_data_points

   graph = NxMixedGraph.from_str_edges(
       directed=[
           ('Z1', 'X'),
           ('X', 'M1'),
           ('M1', 'M2'),
           ('M2', 'Y'),
           ('Z1', 'Z2'),
           ('Z2', 'Z3'),
           ('Z3', 'Y')
       ]
   )

   # Generate observational data for this graph (this is a special example)
   observational_df = generate(num_samples=2_000, seed=1)

   generate_plot_expected_p_value_vs_num_data_points(
       observational_df,
       start=50,
       stop=2_000,
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
algorithms, for examples the ones implemented in <pgmpy> module
https://pgmpy.org/examples/Structure%20Learning%20in%20Bayesian%20Networks.html
can be used to revise the network structure and align it with data. This module currently does not repair the
structure of the network if the network is not aligned with data according to conditional independence tests.

For more reference on this topic, please see
chapter 4 of https://livebook.manning.com/book/causal-ai/welcome/v-4/.


.. [Sachs2005] Sachs, Karen, et al. "Causal protein-signaling networks derived from multiparameter
   single-cell data." Science 308.5721 (2005): 523-529.

.. [halsey2015fickle] Halsey, Lewis G., et al. "The fickle P value generates irreproducible results.
   " Nature methods 12.3 (2015): 179-185.

.. [wang2022addressing] Wang, Ming, and Qi Long. "Addressing Common Misuses and Pitfalls of p values
   in Biomedical Research." Cancer research 82.15 (2022): 2674-2677.

.. [lucas2013too] Lucas, H., and G. Shmueli. "Too big to fail: large samples and the p-value problem."
   Inf. Syst. Res. 24.4 (2013): 906-917.
"""

import logging
from typing import Dict, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import mean, quantile
from tqdm.auto import trange

from y0.algorithm.falsification import get_graph_falsifications
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import CITest, get_conditional_independence_tests

__all__ = [
    "conditional_independence_test_summary",
    "p_value_of_bootstrap_data",
    "p_value_statistics",
    "generate_plot_expected_p_value_vs_num_data_points",
]

logger = logging.getLogger(__name__)

TESTS = get_conditional_independence_tests()


def _get_state_space_map(
    data: pd.DataFrame, threshold: Optional[int] = 10
) -> Dict[Variable, Literal["discrete", "continuous"]]:
    """Get a dictionary from each variable to its type.

    :param data: the observed data
    :param threshold: The threshold for determining a column as discrete
        based on the number of unique values
    :return: the mapping from column name to its type
    """
    column_values_unique_count = {
        column_name: data[column_name].nunique() for column_name in data.columns
    }
    return {
        Variable(column): "discrete"
        if column_values_unique_count[column] <= threshold
        else "continuous"
        for column in data.columns
    }


def _is_data_discrete(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has discrete data.

    :param data: observational data.
    :return: True, if all the columns have discrete data, False, otherwise
    """
    variable_types = set(_get_state_space_map(data=data).values())
    return variable_types == {"discrete"}


def _is_data_continuous(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has continuous data.

    :param data: observational.
    :return: True, if all the columns have continuous data, False, otherwise
    """
    variable_types = set(_get_state_space_map(data).values())
    return variable_types == {"continuous"}


def _choose_default_test(data: pd.DataFrame) -> CITest:
    """Choose the default statistical test for testing conditional independencies based on the data.

    :param data: observational data.
    :return: the default test based on data
    :raises NotImplementedError: if data is of mixed type (contains both discrete and continuous columns)
    """
    if _is_data_discrete(data):
        return "chi-square"
    if _is_data_continuous(data):
        return "pearson"
    raise NotImplementedError(
        "Mixed data types are not allowed. Either all of the columns of data should be discrete / continuous."
    )


def _validate_test(
    data: pd.DataFrame,
    test: Optional[CITest],
) -> None:
    """Validate the conditional independency test passed by the user.

    :param data: observational data.
    :param test: the conditional independency test passed by the user.
    :raises ValueError: if the passed test is invalid / unsupported, pearson is used for discrete data or
        chi-square is used for continuous data
    """
    tests = get_conditional_independence_tests()
    if test not in tests:
        raise ValueError(f"`{test}` is invalid. Supported CI tests are: {sorted(tests)}")

    if _is_data_continuous(data) and test != "pearson":
        raise ValueError(
            "The data is continuous. Either discretize and use chi-square or use the pearson."
        )

    if _is_data_discrete(data) and test == "pearson":
        raise ValueError("Cannot run pearson on discrete data. Use chi-square instead.")


def conditional_independence_test_summary(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    test: Optional[CITest] = None,
    max_given: Optional[int] = 5,
    significance_level: Optional[float] = None,
    verbose: Optional[bool] = False,
) -> None:
    """Print the summary of conditional independency test results.

    Prints the summary to the console, which includes the total number of conditional independence tests,
    the number and percentage of failed tests, and statistical information about each test such as p-values,
    and test results.

    :param graph: an NxMixedGraph
    :param data: observational data corresponding to the graph
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param max_given: The maximum set size in the power set of the vertices minus the d-separable pairs
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :param verbose: If `False`, only print the details of failed tests.
        If 'True', print the details of all the conditional independency results. Defaults to `False`
    :raises NotImplementedError: if data is of mixed type (contains both discrete and continuous columns)
    """
    if significance_level is None:
        significance_level = 0.01
    if not test:
        test = _choose_default_test(data)
    else:
        # Validate test and data
        _validate_test(data=data, test=test)
        if len(set(_get_state_space_map(data).values())) > 1:
            raise NotImplementedError(
                "Mixed data types are not allowed. Either all of the columns of data should be discrete / continuous."
            )
    test_results = get_graph_falsifications(
        graph=graph,
        df=data,
        method=test,
        significance_level=significance_level,
        max_given=max_given,
    ).evidence
    # Find the result based on p-value
    test_results["result"] = test_results["p"].apply(
        lambda p_value: "fail" if p_value < significance_level else "pass"
    )
    # Selecting columns of interest
    test_results = test_results[["left", "right", "given", "p", "result"]]
    # Sorting the rows by index
    test_results = test_results.sort_index()
    test_results = test_results.rename(columns={"p": "p-value"})
    failed_tests = test_results[test_results["result"] == "fail"]
    total_no_of_tests = len(test_results)
    total_no_of_failed_tests = len(failed_tests)
    percentage_of_failed_tests = total_no_of_failed_tests / total_no_of_tests
    logger.info(f"Total number of conditional independencies: {total_no_of_tests:,}")
    logger.info(f"Total number of failed tests: {total_no_of_failed_tests:,}")
    logger.info(f"Percentage of failed tests: {percentage_of_failed_tests:.2%}")
    if verbose:
        logger.info(test_results.to_string(index=False))
    else:
        logger.info(failed_tests.to_string(index=False))


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
    if test is None:
        test = _choose_default_test(df)
    else:
        _validate_test(data=df, test=test)
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
    if not test:
        test = _choose_default_test(df)
    else:
        _validate_test(data=df, test=test)
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


def generate_plot_expected_p_value_vs_num_data_points(
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
