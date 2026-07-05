from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2


def safe_log(value: float) -> float:
    """
    Compute log safely, treating log(0) as zero contribution when needed.

    Parameters
    ----------
    value:
        Input value.

    Returns
    -------
    float
        Natural logarithm or zero if value is zero.
    """
    if value <= 0:
        return 0.0

    return float(np.log(value))


def bernoulli_loglikelihood(
    violations: int,
    observations: int,
    probability: float,
) -> float:
    """
    Compute Bernoulli log-likelihood.

    Parameters
    ----------
    violations:
        Number of violations.
    observations:
        Number of observations.
    probability:
        Violation probability.

    Returns
    -------
    float
        Bernoulli log-likelihood.
    """
    if probability <= 0:
        if violations == 0:
            return 0.0
        return -np.inf

    if probability >= 1:
        if violations == observations:
            return 0.0
        return -np.inf

    non_violations = observations - violations

    return (
        violations * safe_log(probability)
        + non_violations * safe_log(1.0 - probability)
    )


def kupiec_unconditional_coverage_test(
    violations: int,
    observations: int,
    expected_probability: float,
) -> tuple[float, float]:
    """
    Compute Kupiec unconditional coverage test.

    Parameters
    ----------
    violations:
        Number of VaR violations.
    observations:
        Number of observations.
    expected_probability:
        Expected violation probability, equal to 1 - VaR probability level.

    Returns
    -------
    tuple[float, float]
        LR statistic and p-value.
    """
    if observations <= 0:
        raise ValueError("observations must be positive.")

    observed_probability = violations / observations

    loglik_null = bernoulli_loglikelihood(
        violations=violations,
        observations=observations,
        probability=expected_probability,
    )

    loglik_alt = bernoulli_loglikelihood(
        violations=violations,
        observations=observations,
        probability=observed_probability,
    )

    lr_stat = -2.0 * (loglik_null - loglik_alt)

    if not np.isfinite(lr_stat):
        lr_stat = np.nan
        p_value = np.nan
    else:
        p_value = 1.0 - chi2.cdf(lr_stat, df=1)

    return float(lr_stat), float(p_value)


def compute_transition_counts(violation_series: pd.Series) -> dict[str, int]:
    """
    Compute first-order transition counts for VaR violations.

    Parameters
    ----------
    violation_series:
        Boolean or binary violation series.

    Returns
    -------
    dict[str, int]
        Transition counts n00, n01, n10, n11.
    """
    values = violation_series.astype(int).to_numpy()

    if len(values) < 2:
        raise ValueError("Need at least two observations for transition counts.")

    previous = values[:-1]
    current = values[1:]

    n00 = int(((previous == 0) & (current == 0)).sum())
    n01 = int(((previous == 0) & (current == 1)).sum())
    n10 = int(((previous == 1) & (current == 0)).sum())
    n11 = int(((previous == 1) & (current == 1)).sum())

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def christoffersen_independence_test(
    violation_series: pd.Series,
) -> tuple[float, float, dict[str, int]]:
    """
    Compute Christoffersen independence test for VaR violations.

    Parameters
    ----------
    violation_series:
        Boolean or binary violation series.

    Returns
    -------
    tuple[float, float, dict[str, int]]
        LR statistic, p-value, and transition counts.
    """
    counts = compute_transition_counts(violation_series)

    n00 = counts["n00"]
    n01 = counts["n01"]
    n10 = counts["n10"]
    n11 = counts["n11"]

    total_transitions = n00 + n01 + n10 + n11
    total_violations_after_transition = n01 + n11

    pi = (
        total_violations_after_transition / total_transitions
        if total_transitions > 0
        else 0.0
    )

    pi0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0

    loglik_independent = (
        (n00 + n10) * safe_log(1.0 - pi)
        + (n01 + n11) * safe_log(pi)
    )

    loglik_markov = (
        n00 * safe_log(1.0 - pi0)
        + n01 * safe_log(pi0)
        + n10 * safe_log(1.0 - pi1)
        + n11 * safe_log(pi1)
    )

    lr_stat = -2.0 * (loglik_independent - loglik_markov)

    if not np.isfinite(lr_stat):
        lr_stat = np.nan
        p_value = np.nan
    else:
        p_value = 1.0 - chi2.cdf(lr_stat, df=1)

    return float(lr_stat), float(p_value), counts


def merge_losses_and_var(
    losses: pd.DataFrame,
    conditional_var_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge actual losses with conditional VaR forecasts.

    Parameters
    ----------
    losses:
        Wide actual loss dataframe.
    conditional_var_long:
        Long conditional VaR table.

    Returns
    -------
    pd.DataFrame
        Long dataframe with actual losses, VaR forecasts, and violations.
    """
    losses_long = losses.stack().reset_index()
    losses_long.columns = ["date", "asset", "loss"]

    merged = conditional_var_long.merge(
        losses_long,
        on=["date", "asset"],
        how="inner",
    )

    merged["violation"] = merged["loss"] > merged["conditional_var"]

    merged = merged.sort_values(
        ["asset", "probability_level", "date"]
    ).reset_index(drop=True)

    return merged


def backtest_var_for_group(
    group: pd.DataFrame,
) -> dict[str, float | int | str]:
    """
    Backtest VaR forecasts for one asset and one probability level.

    Parameters
    ----------
    group:
        Dataframe for one asset and probability level.

    Returns
    -------
    dict
        Backtesting results.
    """
    asset = str(group["asset"].iloc[0])
    probability_level = float(group["probability_level"].iloc[0])
    expected_probability = 1.0 - probability_level

    violations = group["violation"].astype(bool)
    violation_count = int(violations.sum())
    observations = int(len(group))
    expected_violations = observations * expected_probability
    violation_rate = violation_count / observations

    kupiec_lr, kupiec_p = kupiec_unconditional_coverage_test(
        violations=violation_count,
        observations=observations,
        expected_probability=expected_probability,
    )

    independence_lr, independence_p, transition_counts = christoffersen_independence_test(
        violations
    )

    conditional_coverage_lr = kupiec_lr + independence_lr

    if np.isfinite(conditional_coverage_lr):
        conditional_coverage_p = 1.0 - chi2.cdf(conditional_coverage_lr, df=2)
    else:
        conditional_coverage_p = np.nan

    return {
        "asset": asset,
        "probability_level": probability_level,
        "observations": observations,
        "expected_violations": expected_violations,
        "violation_count": violation_count,
        "violation_rate": violation_rate,
        "expected_violation_rate": expected_probability,
        "kupiec_lr": kupiec_lr,
        "kupiec_p_value": kupiec_p,
        "independence_lr": independence_lr,
        "independence_p_value": independence_p,
        "conditional_coverage_lr": conditional_coverage_lr,
        "conditional_coverage_p_value": conditional_coverage_p,
        "n00": transition_counts["n00"],
        "n01": transition_counts["n01"],
        "n10": transition_counts["n10"],
        "n11": transition_counts["n11"],
    }


def run_var_backtests(
    losses: pd.DataFrame,
    conditional_var_long: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run VaR backtests for all assets and probability levels.

    Parameters
    ----------
    losses:
        Wide actual loss dataframe.
    conditional_var_long:
        Long conditional VaR table.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Backtesting result table and violation-level dataframe.
    """
    violations = merge_losses_and_var(
        losses=losses,
        conditional_var_long=conditional_var_long,
    )

    records = []

    grouped = violations.groupby(["asset", "probability_level"])

    for _, group in grouped:
        records.append(backtest_var_for_group(group))

    results = pd.DataFrame(records)

    results = results.sort_values(
        ["asset", "probability_level"]
    ).reset_index(drop=True)

    return results, violations