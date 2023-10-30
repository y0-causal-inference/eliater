"""This module checks the validity of network structure against observational data.

Given an acyclic directed mixed graph (ADMG) and corresponding observational data,
one can assess whether the conditional independences implied by the structure of the
ADMG are supported by the data with a statistical test for conditional independence.
By default, this workflow uses a chi-square test for discrete data and a Pearson test
for continuous data from :mod:`pgmpy.estimators.CITests`.

This module provides a summary statistics for the total number of tests, percentage of failed
tests, and a list of all (or the failed tests) with their corresponding p-value.

This process allows for checking the validity of network structure with respect to data.
If the percentage of failed tests is higher than the user expects, additional experiments
is required to change the model.

Here is an example of a protein signalling network of the T cell signaling pathway presented
in [Sachs2005]_. It models the molecular mechanisms and regulatory processes involved
in T cell activation, proliferation, and function.

.. image:: docs/source/img/Signaling.pdf
   :width: 200px
   :height: 100px
   :scale: 50 %
   :alt: alternate text
   :align: right


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

    # Get the data
    data = load_sachs_df()

    conditional_independence_test_summary(graph, data, verbose=True)

.. todo:: embed results table

The results show that out of 35 cases, 1 failed. The failed test is
the conditional independence between P38 and PIP2, given PKC, with a p-value of 0.00425.

This module relies on statistical tests, and statistical tests always have chances
of producing false negatives, i.e., a pair of variables that are conditionally
independent, be concluded as conditional dependent by the test, or producing false
positives, i.e., a pair of variables that are conditionally dependent be concluded
as conditionally independent by the test.

Here are some reasons that the result of the test may be false negative or false positive:

1. In pgmpy, the conditional independence tests assume that the alternative hypothesis is
   dependence, while the null hypothesis is conditional independence. However, when dealing
   with an ADMG and hypothetically assuming that the ADMG has the correct structure, it is
   more appropriate for the null hypothesis to be the hypothesis of dependence. This distinction
   can be significant as the p-value relies on the null hypothesis.

   It's worth noting that this module employs traditional tests where the null hypothesis is
   conditional independence.

2. In addition, p-values decrease as the number of data points used in the conditional
   independency test increases, i.e., the larger the data, more conditional independences
   implied by the network will be considered as dependent. Hence, chances of false negatives
   increases.

3. Conditional independence tests rely on probability assumptions regarding the data distribution.
   For instance, when dealing with discrete data, employing the chi-square test generates a test statistic
   that conforms to the Chi-squared probability distribution. Similarly, in the case of continuous data,
   utilizing the Pearson test yields a test statistic that adheres to the Normal distribution. If these
   assumptions are not satisfied by the data, the outcomes may lead to both false positives and false negatives.

As a result, the results obtained from this module should be regarded more as heuristics approach rather
than a systematic, strict step that provides precise results. For more reference on this topic, please see
chapter 4 of https://livebook.manning.com/book/causal-ai/welcome/v-4/.

.. [Sachs2005] K. Sachs, O. Perez, D. Pe’er, D. A. Lauffenburger, and G. P. Nolan.
Causal protein-signaling networks derived from multiparameter single-cell data. Science, 308(5721): 523–529, 2005.
"""

import logging
from typing import Dict, Literal, Optional

import pandas as pd

from y0.algorithm.falsification import get_graph_falsifications
from y0.dsl import Variable
from y0.graph import NxMixedGraph
from y0.struct import get_conditional_independence_tests

logging.basicConfig(format="%(message)s", level=logging.DEBUG)


__all__ = [
    "conditional_independence_test_summary",
    "validate_test",
    "get_state_space_map",
    "is_data_discrete",
    "is_data_continuous",
    "CITest",
    "choose_default_test",
]


def get_state_space_map(
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


def is_data_discrete(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has discrete data.

    :param data: observational data.
    :return: True, if all the columns have discrete data, False, otherwise
    """
    variable_types = set(get_state_space_map(data=data).values())
    return variable_types == {"discrete"}


def is_data_continuous(data: pd.DataFrame) -> bool:
    """Check if all the columns in the dataframe has continuous data.

    :param data: observational.
    :return: True, if all the columns have continuous data, False, otherwise
    """
    variable_types = set(get_state_space_map(data).values())
    return variable_types == {"continuous"}


CITest = Literal[
    "pearson",
    "chi-square",
    "cressie_read",
    "freeman_tuckey",
    "g_sq",
    "log_likelihood",
    "modified_log_likelihood",
    "power_divergence",
    "neyman",
]


def choose_default_test(data: pd.DataFrame) -> CITest:
    """Choose the default statistical test for testing conditional independencies based on the data.

    :param data: observational data.
    :return: the default test based on data
    :raises NotImplementedError: if data is of mixed type (contains both discrete and continuous columns)
    """
    if is_data_discrete(data):
        return "chi-square"
    if is_data_continuous(data):
        return "pearson"
    raise NotImplementedError(
        "Mixed data types are not allowed. Either all of the columns of data should be discrete / continuous."
    )


def validate_test(
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

    if is_data_continuous(data) and test != "pearson":
        raise ValueError(
            "The data is continuous. Either discretize and use chi-square or use the pearson."
        )

    if is_data_discrete(data) and test == "pearson":
        raise ValueError("Cannot run pearson on discrete data. Use chi-square instead.")


def conditional_independence_test_summary(
    graph: NxMixedGraph,
    data: pd.DataFrame,
    test: Optional[CITest] = None,
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
        test = choose_default_test(data)
    else:
        # Validate test and data
        validate_test(data=data, test=test)
        if len(set(get_state_space_map(data).values())) > 1:
            raise NotImplementedError(
                "Mixed data types are not allowed. Either all of the columns of data should be discrete / continuous."
            )
    test_results = get_graph_falsifications(
        graph=graph, df=data, method=test, significance_level=significance_level
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
    percentage_of_failed_tests = total_no_of_failed_tests / total_no_of_tests * 100
    logging.info("Total number of conditional independencies: " + str(total_no_of_tests))
    logging.info("Total number of failed tests: " + str(total_no_of_failed_tests))
    logging.info("Percentage of failed tests: " + str(percentage_of_failed_tests))
    if verbose:
        logging.info(test_results.to_string(index=False))
    else:
        logging.info(failed_tests.to_string(index=False))



