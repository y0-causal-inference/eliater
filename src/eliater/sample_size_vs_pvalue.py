from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean, quantile
from pgmpy.estimators import CITests

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


def sample_p_val(
    full_data: pd.DataFrame,
    sample_size: int,
    left: str,
    right: str,
    conditions: list,
    test: str,
    significance_level: Optional[float] = None,
) -> int:
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


def estimate_p_val(
    full_data: pd.DataFrame,
    sample_size: int,
    left: str,
    right: str,
    conditions: list,
    test: str,
    significance_level: int,
    boot_size: int = 1000,
):
    samples = []
    for _ in range(boot_size):
        sample = sample_p_val(
            full_data, sample_size, left, right, conditions, test, significance_level
        )
        samples.append(sample)
    p_val = mean(samples)  # Calculate the mean of the p-values to get the bootstrap mean.
    if p_val < 0.05:
        print(sample_size)
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
    test: str,
    significance_level: float,
    boot_size: int,
):
    data_size = range(min_number_of_sampled_data_points, max_number_of_sampled_data_points, step)
    p_vals, lower_errors, higher_errors = zip(
        *[
            estimate_p_val(
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


from eliater.frontdoor_backdoor.multiple_mediators_with_multiple_confounders_nuisances import (
    generate,
)

generate_plot_expected_p_value_vs_num_data_points(
    full_data=generate(num_samples=2000, seed=1),
    min_number_of_sampled_data_points=50,
    max_number_of_sampled_data_points=2000,
    step=50,
    left="M2",
    right="R2",
    conditions=["M1"],
    test="pearson",
    significance_level=0.05,
    boot_size=1000,
)
