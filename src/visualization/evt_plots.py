from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_output_directory(path: str | Path) -> Path:
    """
    Create an output directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        Output directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def gpd_cdf(x: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """
    Compute the GPD cumulative distribution function.

    Parameters
    ----------
    x:
        Non-negative excess values.
    xi:
        GPD shape parameter.
    beta:
        GPD scale parameter.

    Returns
    -------
    np.ndarray
        GPD CDF values.
    """
    x = np.asarray(x, dtype=float)

    if abs(xi) < 1e-8:
        return 1.0 - np.exp(-x / beta)

    support = 1.0 + xi * x / beta

    cdf = np.full_like(x, np.nan, dtype=float)
    valid = support > 0
    cdf[valid] = 1.0 - support[valid] ** (-1.0 / xi)

    return cdf


def gpd_ppf(probabilities: np.ndarray, xi: float, beta: float) -> np.ndarray:
    """
    Compute the GPD quantile function.

    Parameters
    ----------
    probabilities:
        Probabilities in (0, 1).
    xi:
        GPD shape parameter.
    beta:
        GPD scale parameter.

    Returns
    -------
    np.ndarray
        GPD quantiles.
    """
    probabilities = np.asarray(probabilities, dtype=float)

    if np.any((probabilities <= 0) | (probabilities >= 1)):
        raise ValueError("Probabilities must be strictly between 0 and 1.")

    if abs(xi) < 1e-8:
        return -beta * np.log(1.0 - probabilities)

    return (beta / xi) * ((1.0 - probabilities) ** (-xi) - 1.0)


def compute_mean_excess_curve(
    series: pd.Series,
    threshold_quantiles: np.ndarray,
    min_exceedances: int = 30,
) -> pd.DataFrame:
    """
    Compute empirical mean excess values across candidate thresholds.

    Parameters
    ----------
    series:
        Series of standardized residual losses.
    threshold_quantiles:
        Quantile levels used as thresholds.
    min_exceedances:
        Minimum number of exceedances required to keep a threshold.

    Returns
    -------
    pd.DataFrame
        Mean excess curve with threshold, quantile, mean excess,
        and exceedance count.
    """
    clean_series = series.dropna().sort_values()

    records = []

    for quantile in threshold_quantiles:
        threshold = float(clean_series.quantile(quantile))
        exceedances = clean_series[clean_series > threshold]
        excesses = exceedances - threshold

        if len(excesses) < min_exceedances:
            continue

        records.append(
            {
                "threshold_quantile": quantile,
                "threshold": threshold,
                "mean_excess": float(excesses.mean()),
                "exceedance_count": int(len(excesses)),
            }
        )

    return pd.DataFrame(records)


def plot_mean_excess_plots(
    residual_losses: pd.DataFrame,
    output_path: str | Path,
    threshold_quantiles: np.ndarray | None = None,
) -> None:
    """
    Plot empirical mean excess curves for all assets.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    output_path:
        Path where the figure is saved.
    threshold_quantiles:
        Candidate threshold quantiles.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    if threshold_quantiles is None:
        threshold_quantiles = np.linspace(0.80, 0.99, 40)

    n_assets = len(residual_losses.columns)
    n_cols = 3
    n_rows = (n_assets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
        squeeze=False,
    )

    for index, asset in enumerate(residual_losses.columns):
        row = index // n_cols
        col = index % n_cols
        ax = axes[row][col]

        curve = compute_mean_excess_curve(
            residual_losses[asset],
            threshold_quantiles=threshold_quantiles,
            min_exceedances=30,
        )

        ax.plot(
            curve["threshold"],
            curve["mean_excess"],
            marker="o",
            linewidth=1,
            markersize=3,
        )

        baseline_threshold = residual_losses[asset].quantile(0.95)
        ax.axvline(baseline_threshold, linestyle="--", linewidth=1)

        ax.set_title(asset)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Mean excess")

    for index in range(n_assets, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        axes[row][col].axis("off")

    fig.suptitle("Mean Excess Plots for Standardized Residual Losses", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gpd_qq_plots(
    residual_losses: pd.DataFrame,
    gpd_params: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot GPD QQ plots for fitted baseline POT-GPD models.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    gpd_params:
        GPD parameter table indexed by asset.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    n_assets = len(residual_losses.columns)
    n_cols = 3
    n_rows = (n_assets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
        squeeze=False,
    )

    for index, asset in enumerate(residual_losses.columns):
        row = index // n_cols
        col = index % n_cols
        ax = axes[row][col]

        params = gpd_params.loc[asset]
        threshold = params["threshold_value"]
        xi = params["shape_xi"]
        beta = params["scale_beta"]

        series = residual_losses[asset].dropna()
        excesses = (series[series > threshold] - threshold).sort_values()
        excess_values = excesses.to_numpy()
        n = len(excess_values)

        probabilities = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = gpd_ppf(probabilities, xi=xi, beta=beta)

        ax.scatter(theoretical_quantiles, excess_values, s=12)

        min_value = min(theoretical_quantiles.min(), excess_values.min())
        max_value = max(theoretical_quantiles.max(), excess_values.max())
        ax.plot([min_value, max_value], [min_value, max_value], linewidth=1)

        ax.set_title(asset)
        ax.set_xlabel("Fitted GPD quantiles")
        ax.set_ylabel("Empirical excess quantiles")

    for index in range(n_assets, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        axes[row][col].axis("off")

    fig.suptitle("GPD QQ Plots for Baseline POT-GPD Fits", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_gpd_probability_plots(
    residual_losses: pd.DataFrame,
    gpd_params: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot GPD probability plots for fitted baseline POT-GPD models.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    gpd_params:
        GPD parameter table indexed by asset.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    n_assets = len(residual_losses.columns)
    n_cols = 3
    n_rows = (n_assets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
        squeeze=False,
    )

    for index, asset in enumerate(residual_losses.columns):
        row = index // n_cols
        col = index % n_cols
        ax = axes[row][col]

        params = gpd_params.loc[asset]
        threshold = params["threshold_value"]
        xi = params["shape_xi"]
        beta = params["scale_beta"]

        series = residual_losses[asset].dropna()
        excesses = (series[series > threshold] - threshold).sort_values()
        excess_values = excesses.to_numpy()
        n = len(excess_values)

        empirical_probabilities = (np.arange(1, n + 1) - 0.5) / n
        fitted_probabilities = gpd_cdf(excess_values, xi=xi, beta=beta)

        ax.scatter(fitted_probabilities, empirical_probabilities, s=12)
        ax.plot([0, 1], [0, 1], linewidth=1)

        ax.set_title(asset)
        ax.set_xlabel("Fitted GPD probabilities")
        ax.set_ylabel("Empirical probabilities")

    for index in range(n_assets, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        axes[row][col].axis("off")

    fig.suptitle("GPD Probability Plots for Baseline POT-GPD Fits", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()