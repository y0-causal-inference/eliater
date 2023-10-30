"""This module shows the relationship between p-value and sample size when testing conditional independencies.

*p*-values decrease as the number of data points used in the conditional independency test
increases, i.e., the larger the data, more conditional independences implied by the network
will be considered as dependent. Hence, chances of false negatives increases. This module
illustrates this. The content of this module are relied on chapter 4 of this reference:
https://livebook.manning.com/book/causal-ai/welcome/v-4/.

Here is an example that illustrates this point. In the provided graph, R2 is independent of
Z1 given R1. In addition, M1 is independent of R2 given R1. The data has been generated based
on these assumption, Hence, we expect the p-value to be above 0.05, i.e., not rejecting the null
hypothesis of conditional independence.

.. todo:: embed a nice screenshot of the graph

.. code-block:: python

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
    observational_data = generate(num_samples=2000, seed=1)

    generate_plot_expected_p_value_vs_num_data_points(
        full_data=observational_data,
        min_number_of_sampled_data_points=50,
        max_number_of_sampled_data_points=2000,
        step=100,
        left="Y",
        right="M1",
        conditions=["M2", "Z2"],
        test="pearson",
        significance_level=0.05,
        boot_size=1000
    )

.. todo:: Embed results of this plot. Reader is not able to understand the point of this with just code

This plot shows that the expected p-value will decrease as number of data points increases. For number
of data points greater than 1000, the test is more likely to reject the null hypothesis, and for number
of data points greater than 1250, the test always rejects the null hypothesis, i.e., the data will
no longer support that Y is independent of M1 given M2, and Z2 where it should be.

"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean, quantile
from pgmpy.estimators import CITests

from eliater.network_validation import CITest, choose_default_test, validate_test

__all__ = [
    "p_value_of_bootstrap_data",
    "p_value_statistics",
    "generate_plot_expected_p_value_vs_num_data_points",
]


tests = {
    "pearson": CITests.pearsonr,
    "chi-square": CITests.chi_square,
    "cressie_read": CITests.cressie_read,
    "freeman_tuckey": CITests.freeman_tuckey,
    "g_sq": CITests.g_sq,
    "log_likelihood": CITests.log_likelihood,
    "modified_log_likelihood": CITests.modified_log_likelihood,
    "power_divergence": CITests.power_divergence,
    "neyman": CITests.neyman,
}


def p_value_of_bootstrap_data(
    full_data: pd.DataFrame,
    sample_size: int,
    left: str,
    right: str,
    conditions: list,
    test: Optional[CITest],
    significance_level: Optional[float] = None,
) -> int:
    """Calculate the p-value for a bootstrap data.

    :param full_data: observational data
    :param sample_size: number of data points to sample a bootstrap data from full_data
    :param left: first variable name positioned on the left side of a conditional independence test
    :param right: second variable name positioned on the right side of a conditional independence test
    :param conditions: variables names to condition on in the conditional independence test
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :return: the p value for the given bootstrap data
    """
    if significance_level is None:
        significance_level = 0.01
    if not test:
        test = choose_default_test(full_data)
    else:
        validate_test(data=full_data, test=test)
    bootstrap_data = full_data.sample(n=sample_size, replace=True)
    result = tests[test](
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
    full_data: pd.DataFrame,
    sample_size: int,
    left: str,
    right: str,
    conditions: list,
    test: Optional[CITest],
    significance_level: float,
    boot_size: int = 1000,
):
    """Calculate mean of p-value, the 5th percentile and 95 percentile error, for several bootstrap data.

    :param full_data: observational data
    :param sample_size: number of data points to sample a bootstrap data from full_data
    :param left: first variable name positioned on the left side of a conditional independence test
    :param right: second variable name positioned on the right side of a conditional independence test
    :param conditions: variables names to condition on in the conditional independence test
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :param boot_size: total number of times a bootstrap data is sampled
    :return: the mean of p-value, the 5th percentile and 95 percentile error, for several bootstrap data
    """
    if significance_level is None:
        significance_level = 0.01
    if not test:
        test = choose_default_test(full_data)
    else:
        validate_test(data=full_data, test=test)
    samples = []
    for _ in range(boot_size):
        sample = p_value_of_bootstrap_data(
            full_data, sample_size, left, right, conditions, test, significance_level
        )
        samples.append(sample)
    p_val = mean(samples)  # Calculate the mean of the p-values to get the bootstrap mean.
    quantile_05, quantile_95 = quantile(samples, q=[0.05, 0.95])
    lower_error = np.absolute(p_val - quantile_05)  # Calculate the 5th percentile
    higher_error = np.absolute(quantile_95 - p_val)  # Calculate the 95th percentile

    return p_val, lower_error, higher_error


def generate_plot_expected_p_value_vs_num_data_points(
    full_data: pd.DataFrame,
    min_number_of_sampled_data_points: int,
    max_number_of_sampled_data_points: int,
    step: int,
    left: str,
    right: str,
    conditions: list,
    test: Optional[CITest],
    significance_level: float,
    boot_size: int,
):
    """Generate the plot of expected p-value versus number of data points.

    :param full_data: observational data
    :param min_number_of_sampled_data_points: minimum number of data points to sample from full_data
    :param max_number_of_sampled_data_points: maximum number of data points to sample from full_data
    :param step: minimum number of sampled data points increments by step number, and stops
        before maximum number of sampled data points
    :param left: first variable name positioned on the left side of a conditional independence test
    :param right: second variable name positioned on the right side of a conditional independence test
    :param conditions: variables names to condition on in the conditional independence test
    :param test: the conditional independency test to use. If None, defaults to ``pearson`` for continuous data
        and ``chi-square`` for discrete data.
    :param significance_level: The statistical tests employ this value for
        comparison with the p-value of the test to determine the independence of
        the tested variables. If none, defaults to 0.01.
    :param boot_size: total number of times a bootstrap data is sampled
    :return: the plot of expected p-value versus number of data points
    """
    if significance_level is None:
        significance_level = 0.01
    if not test:
        test = choose_default_test(full_data)
    else:
        validate_test(data=full_data, test=test)
    data_size = range(min_number_of_sampled_data_points, max_number_of_sampled_data_points, step)
    p_vals, lower_errors, higher_errors = zip(
        *[
            p_value_statistics(
                full_data=full_data,
                sample_size=size,
                left=left,
                right=right,
                conditions=conditions,
                test=test,
                significance_level=significance_level,
                boot_size=boot_size,
            )
            for size in data_size
        ]
    )

    if len(conditions) < 1:
        plt.title("# data points vs. expected p-value (Independence of" + left + "&" + right)
    else:
        conditions_string = ""
        for i in range(len(conditions)):
            if len(conditions) == 1:
                conditions_string = conditions[i]
            else:
                conditions_string = conditions_string + conditions[i] + ", "
        plt.title(
            "# data points vs. expected p-value (Ind. of "
            + left
            + " & "
            + right
            + " given "
            + conditions_string
        )

    plt.xlabel("number of data points")
    plt.ylabel("expected p-value")
    plt.errorbar(
        data_size,
        p_vals,
        yerr=np.array([lower_errors, higher_errors]),
        ecolor="grey",
        elinewidth=0.5,
    )
    plt.hlines(0.05, 0, max_number_of_sampled_data_points, linestyles="dashed")
    return plt.show()
