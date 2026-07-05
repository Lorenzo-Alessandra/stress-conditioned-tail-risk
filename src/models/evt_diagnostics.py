from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ClusterSummary:
    """
    Summary of exceedance clustering for one asset and one threshold.
    """

    cluster_count: int
    mean_cluster_size: float
    max_cluster_size: int
    cluster_ratio: float


def compute_threshold(series: pd.Series, quantile: float) -> float:
    """
    Compute an empirical threshold as a sample quantile.

    Parameters
    ----------
    series:
        Series of standardized residual losses.
    quantile:
        Quantile level used as threshold.

    Returns
    -------
    float
        Empirical threshold value.
    """
    clean_series = series.dropna()

    if clean_series.empty:
        raise ValueError("Cannot compute threshold from an empty series.")

    return float(clean_series.quantile(quantile))


def get_exceedance_indicator(series: pd.Series, threshold: float) -> pd.Series:
    """
    Compute the exceedance indicator for a threshold.

    Parameters
    ----------
    series:
        Series of standardized residual losses.
    threshold:
        Threshold value.

    Returns
    -------
    pd.Series
        Boolean series equal to True when the loss exceeds the threshold.
    """
    indicator = series > threshold
    indicator = indicator.fillna(False)

    return indicator


def summarize_exceedance_clusters(
    exceedance_indicator: pd.Series,
    run_length: int,
) -> ClusterSummary:
    """
    Summarize exceedance clusters using a runs rule.

    A new cluster starts after at least `run_length` consecutive
    non-exceedance observations.

    Parameters
    ----------
    exceedance_indicator:
        Boolean series indicating threshold exceedances.
    run_length:
        Number of consecutive non-exceedances required to separate clusters.

    Returns
    -------
    ClusterSummary
        Cluster count, average cluster size, maximum cluster size,
        and cluster-to-exceedance ratio.
    """
    if run_length < 1:
        raise ValueError("run_length must be at least 1.")

    exceedance_values = exceedance_indicator.astype(bool).to_numpy()

    cluster_sizes: list[int] = []
    current_cluster_size = 0
    quiet_days = run_length

    for is_exceedance in exceedance_values:
        if is_exceedance:
            if quiet_days >= run_length:
                if current_cluster_size > 0:
                    cluster_sizes.append(current_cluster_size)
                current_cluster_size = 1
            else:
                current_cluster_size += 1

            quiet_days = 0
        else:
            quiet_days += 1

    if current_cluster_size > 0:
        cluster_sizes.append(current_cluster_size)

    exceedance_count = int(exceedance_indicator.sum())
    cluster_count = len(cluster_sizes)

    if cluster_count == 0:
        mean_cluster_size = 0.0
        max_cluster_size = 0
    else:
        mean_cluster_size = float(sum(cluster_sizes) / cluster_count)
        max_cluster_size = int(max(cluster_sizes))

    if exceedance_count == 0:
        cluster_ratio = 0.0
    else:
        cluster_ratio = float(cluster_count / exceedance_count)

    return ClusterSummary(
        cluster_count=cluster_count,
        mean_cluster_size=mean_cluster_size,
        max_cluster_size=max_cluster_size,
        cluster_ratio=cluster_ratio,
    )


def make_evt_threshold_diagnostics_table(
    residual_losses: pd.DataFrame,
    threshold_quantiles: list[float],
    run_length: int,
) -> pd.DataFrame:
    """
    Create EVT threshold and clustering diagnostics for all assets.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    threshold_quantiles:
        List of quantile thresholds.
    run_length:
        Runs declustering parameter.

    Returns
    -------
    pd.DataFrame
        Diagnostics table.
    """
    records = []

    for asset in residual_losses.columns:
        series = residual_losses[asset].dropna()
        n_observations = len(series)

        if n_observations == 0:
            raise ValueError(f"No residual losses available for {asset}.")

        for threshold_quantile in threshold_quantiles:
            threshold_value = compute_threshold(series, threshold_quantile)
            exceedance_indicator = get_exceedance_indicator(series, threshold_value)

            exceedance_count = int(exceedance_indicator.sum())
            exceedance_rate = float(exceedance_count / n_observations)

            cluster_summary = summarize_exceedance_clusters(
                exceedance_indicator=exceedance_indicator,
                run_length=run_length,
            )

            records.append(
                {
                    "asset": asset,
                    "threshold_quantile": threshold_quantile,
                    "threshold_value": threshold_value,
                    "n_observations": n_observations,
                    "exceedance_count": exceedance_count,
                    "exceedance_rate": exceedance_rate,
                    "cluster_count": cluster_summary.cluster_count,
                    "mean_cluster_size": cluster_summary.mean_cluster_size,
                    "max_cluster_size": cluster_summary.max_cluster_size,
                    "cluster_ratio": cluster_summary.cluster_ratio,
                    "run_length": run_length,
                }
            )

    diagnostics = pd.DataFrame(records)

    diagnostics = diagnostics.sort_values(
        ["asset", "threshold_quantile"]
    ).reset_index(drop=True)

    return diagnostics


def format_evt_threshold_diagnostics_table(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Format EVT diagnostics table for output.

    Parameters
    ----------
    diagnostics:
        Raw diagnostics table.

    Returns
    -------
    pd.DataFrame
        Formatted diagnostics table.
    """
    formatted = diagnostics.copy()

    rounded_columns = [
        "threshold_quantile",
        "threshold_value",
        "exceedance_rate",
        "mean_cluster_size",
        "cluster_ratio",
    ]

    for column in rounded_columns:
        formatted[column] = formatted[column].round(4)

    integer_columns = [
        "n_observations",
        "exceedance_count",
        "cluster_count",
        "max_cluster_size",
        "run_length",
    ]

    for column in integer_columns:
        formatted[column] = formatted[column].astype(int)

    return formatted