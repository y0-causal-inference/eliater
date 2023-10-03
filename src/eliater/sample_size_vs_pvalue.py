import math
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean, quantile
import numpy as np
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
    sample_size: chr,
    left: chr,
    right: chr,
    conditions: list,
    test: chr,
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
    left: chr,
    right: chr,
    conditions: list,
    test: CITests,
    significance_level: Optional[float] = None,
    boot_size: int = 1000,
):
    samples = []
    for _ in range(boot_size):
        sample = sample_p_val(
            full_data, sample_size, left, right, conditions, test, significance_level
        )
        samples.append(sample)
    positive_tests = [p_val > significance_level for p_val in samples]
    prob_conclude_indep = mean(positive_tests)
    p_estimate = mean(samples)  # Calculate the mean of the p-values to get the bootstrap mean.
    quantile_05, quantile_95 = quantile(samples, q=[0.05, 0.95])  # E
    lower_error = p_estimate - quantile_05  # Calculate the 5th percentile
    higher_error = quantile_95 - p_estimate  # Calculate the 95th percentile
    return p_estimate, lower_error, higher_error, prob_conclude_indep


from frontdoor_backdoor import multiple_mediators_confounders_example

full_data = multiple_mediators_confounders_example.generate_data(num_samples=1000)

data_size = range(30, 1000, 50)
p_vals, lower_errors, higher_errors, probs_conclude_indep = zip(
    *[
        estimate_p_val(
            full_data=full_data,
            sample_size=size,
            left="M2",
            right="Z2",
            conditions=["M1"],
            test= "pearson",
            significance_level=0.05,
            boot_size=1000,
        )
        for size in data_size
    ]
)

plt.title("Amount of data vs. expected p-value (Ind. of M2 & Z2 given M1)")
plt.xlabel("Number of data points")
plt.ylabel("Expected p-value")
plt.errorbar(
    data_size, p_vals, yerr=np.array([lower_errors, higher_errors]), ecolor="grey", elinewidth=0.5
)
plt.hlines(0.05, 0, 1000, linestyles="dashed")
plt.show()