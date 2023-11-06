"""This module shows the relationship between p-value and sample size when testing conditional independencies.

The *p*-value of a data-driven conditional independency test
(e.g., the pearson test applied to continuous data) decreases as the number of data points increases.
This means that the chances of false negatives increases for larger datasets, i.e., a pair of variables that are
conditionally independent, be concluded as conditional dependent by the test.

We demonstrate this phenomena below using the following example graph, observational data
(simulated specifically for this graph using
:func:`eliater.frontdoor_backdoor.multiple_mediators_with_multiple_confounders.generate`),
and the application of subsampling.

.. todo::

    This name (frontdoor_backdoor.multiple_mediators_with_multiple_confounders) is so awful,
    way too hard to even write out or read. Maybe best to just call it example T1 or something short

.. image:: ../../docs/source/img/multiple_mediators_with_multiple_confounders.png
   :width: 200px

.. warning::

    This module is implemented based on Chapter 4 from
    https://livebook.manning.com/book/causal-ai/welcome/v-4/, however
    this resource is paywalled.

In this graph, $Y$ is conditionally independent (i.e., D-separated) of
$M_1$ given $M_2$, $Z_2$. The data has been generated based on this assumption. Hence, we expect the
$p$-value to be above 0.05, i.e., not rejecting the null hypothesis of conditional independence. We
use the following workflow to graphically assess how this compares to a data-driven approach.

.. code-block:: python

    import matplotlib.pyplot as plt
    from matplotlib_inline.backend_inline import set_matplotlib_formats

    from y0.graph import NxMixedGraph
    from eliater.frontdoor_backdoor.multiple_mediators_with_multiple_confounders import generate
    from eliater.sample_size_vs_pvalue import generate_plot_expected_p_value_vs_num_data_points

    graph = NxMixedGraph.from_edges(
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


.. image:: ../../docs/source/img/pvalue_vs_sample_size.svg
   :width: 350px
   :height: 250px
   :scale: 200 %
   :alt: alternate text
   :align: right

This plot shows that the expected $p$-value will decrease as number of data points increases. For number
of data points greater than 1,000, the test is more likely to reject the null hypothesis, and for number
of data points greater than 1,250, the test always rejects the null hypothesis, i.e., the data will
no longer support that $Y$ is independent of $M_1$ given $M_2$, and $Z_2$ where it should be.

.. todo::

    Several questions need to be answered here:

    - How do I interpret these results?
    - What follow-up am I supposed to do after seeing this? Do I modify my network? What is the workflow for deciding
      when to do that or when not to?
    - This is just a single D-separation. How do I think about scaling this to all possible d-separations?
    - How am I supposed to use this for my dataset?
    - What do I do once I see this graph for my data?
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean, quantile
from tqdm.auto import trange

from eliater.network_validation import CITest, choose_default_test, validate_test
from y0.struct import get_conditional_independence_tests

__all__ = [
    "p_value_of_bootstrap_data",
    "p_value_statistics",
    "generate_plot_expected_p_value_vs_num_data_points",
]


TESTS = get_conditional_independence_tests()


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
        test = choose_default_test(df)
    else:
        validate_test(data=df, test=test)
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
        test = choose_default_test(df)
    else:
        validate_test(data=df, test=test)
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
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.05.
    :param boot_size: total number of times a bootstrap data is sampled
    :return: the plot of expected p-value versus number of data points
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
        plt.title(
            f"Independence of {left} and {right}"
        )
    else:
        conditions_string = ", ".join(conditions)
        plt.title(
            f"Independence of {left} and {right} given {conditions_string}"
        )

    # TODO try using seaborn for this, gets much higher quality charts
    plt.xlabel("Data Points")
    plt.ylabel("Expected p-Value")
    plt.errorbar(
        list(range(start, stop, step)),
        p_vals,
        yerr=np.array([lower_errors, higher_errors]),
        ecolor="grey",
        elinewidth=0.5,
    )
    plt.hlines(significance_level, 0, stop, linestyles="dashed")
