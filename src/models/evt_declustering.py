from __future__ import annotations

import pandas as pd


def extract_runs_declustered_excesses(
    series: pd.Series,
    threshold: float,
    run_length: int,
) -> pd.Series:
    """
    Extract declustered threshold excesses using a runs rule.

    A new cluster begins after at least `run_length` consecutive
    non-exceedance observations. For each cluster, the maximum value
    is retained and converted into an excess over the threshold.

    Parameters
    ----------
    series:
        Series of standardized residual losses.
    threshold:
        Threshold value.
    run_length:
        Number of consecutive non-exceedance days required to separate clusters.

    Returns
    -------
    pd.Series
        Declustered excesses, indexed by the date of each cluster maximum.
    """
    if run_length < 1:
        raise ValueError("run_length must be at least 1.")

    clean_series = series.dropna()

    cluster_values: list[float] = []
    cluster_dates: list[pd.Timestamp] = []

    current_cluster_values: list[float] = []
    current_cluster_dates: list[pd.Timestamp] = []

    quiet_days = run_length

    for date, value in clean_series.items():
        is_exceedance = value > threshold

        if is_exceedance:
            if quiet_days >= run_length:
                if current_cluster_values:
                    max_position = int(pd.Series(current_cluster_values).idxmax())
                    cluster_values.append(current_cluster_values[max_position])
                    cluster_dates.append(current_cluster_dates[max_position])

                current_cluster_values = [float(value)]
                current_cluster_dates = [date]
            else:
                current_cluster_values.append(float(value))
                current_cluster_dates.append(date)

            quiet_days = 0
        else:
            quiet_days += 1

    if current_cluster_values:
        max_position = int(pd.Series(current_cluster_values).idxmax())
        cluster_values.append(current_cluster_values[max_position])
        cluster_dates.append(current_cluster_dates[max_position])

    cluster_maxima = pd.Series(cluster_values, index=cluster_dates)
    excesses = cluster_maxima - threshold
    excesses = excesses[excesses > 0]

    return excesses.sort_index()


def extract_excesses_by_declustering_method(
    series: pd.Series,
    threshold: float,
    method: str,
    run_length: int | None = None,
) -> pd.Series:
    """
    Extract excesses according to a declustering method.

    Parameters
    ----------
    series:
        Series of standardized residual losses.
    threshold:
        Threshold value.
    method:
        Either 'none' or 'runs'.
    run_length:
        Runs declustering length. Required when method='runs'.

    Returns
    -------
    pd.Series
        Threshold excesses.
    """
    clean_series = series.dropna()

    if method == "none":
        exceedances = clean_series[clean_series > threshold]
        return exceedances - threshold

    if method == "runs":
        if run_length is None:
            raise ValueError("run_length must be provided when method='runs'.")

        return extract_runs_declustered_excesses(
            series=clean_series,
            threshold=threshold,
            run_length=run_length,
        )

    raise ValueError(f"Unknown declustering method: {method}")