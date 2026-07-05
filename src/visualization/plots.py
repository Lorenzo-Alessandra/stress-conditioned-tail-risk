from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def plot_bank_returns_timeseries(
    returns: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot daily bank log returns.

    Parameters
    ----------
    returns:
        Wide dataframe of daily bank returns.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    ax = returns.plot(
        figsize=(14, 8),
        linewidth=0.7,
        alpha=0.8,
    )

    ax.set_title("Daily Bank Log Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log return")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_system_loss_timeseries(
    system_loss: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot equal-weighted system loss.

    Parameters
    ----------
    system_loss:
        Dataframe containing system loss.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    column = system_loss.columns[0]

    ax = system_loss[column].plot(
        figsize=(14, 6),
        linewidth=0.8,
    )

    ax.set_title("Equal-Weighted System Loss")
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_bank_loss_histograms(
    losses: pd.DataFrame,
    output_path: str | Path,
    bins: int = 80,
) -> None:
    """
    Plot histograms of bank losses.

    Parameters
    ----------
    losses:
        Wide dataframe of daily bank losses.
    output_path:
        Path where the figure is saved.
    bins:
        Number of histogram bins.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    n_assets = len(losses.columns)
    n_cols = 3
    n_rows = (n_assets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 4 * n_rows),
        squeeze=False,
    )

    for index, column in enumerate(losses.columns):
        row = index // n_cols
        col = index % n_cols
        ax = axes[row][col]

        ax.hist(losses[column].dropna(), bins=bins)
        ax.set_title(column)
        ax.set_xlabel("Loss")
        ax.set_ylabel("Frequency")

    for index in range(n_assets, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        axes[row][col].axis("off")

    fig.suptitle("Histograms of Daily Bank Losses", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rolling_volatility(
    returns: pd.DataFrame,
    output_path: str | Path,
    window: int = 60,
) -> None:
    """
    Plot rolling volatility of bank returns.

    Parameters
    ----------
    returns:
        Wide dataframe of daily bank returns.
    output_path:
        Path where the figure is saved.
    window:
        Rolling window length in trading days.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    rolling_vol = returns.rolling(window=window).std()

    ax = rolling_vol.plot(
        figsize=(14, 8),
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_title(f"{window}-Day Rolling Volatility of Bank Returns")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling standard deviation")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_garch_conditional_volatility(
    conditional_volatility: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot GARCH conditional volatility estimates.

    Parameters
    ----------
    conditional_volatility:
        Wide dataframe of conditional volatility estimates.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    ax = conditional_volatility.plot(
        figsize=(14, 8),
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_title("GJR-GARCH Conditional Volatility")
    ax.set_xlabel("Date")
    ax.set_ylabel("Conditional volatility")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_standardized_residuals(
    standardized_residuals: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot GARCH standardized residuals.

    Parameters
    ----------
    standardized_residuals:
        Wide dataframe of standardized residuals.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    ax = standardized_residuals.plot(
        figsize=(14, 8),
        linewidth=0.7,
        alpha=0.8,
    )

    ax.set_title("GJR-GARCH Standardized Residuals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized residual")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_standardized_residual_loss_histograms(
    residual_losses: pd.DataFrame,
    output_path: str | Path,
    bins: int = 80,
) -> None:
    """
    Plot histograms of standardized residual losses.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    output_path:
        Path where the figure is saved.
    bins:
        Number of histogram bins.
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

    for index, column in enumerate(residual_losses.columns):
        row = index // n_cols
        col = index % n_cols
        ax = axes[row][col]

        ax.hist(residual_losses[column].dropna(), bins=bins)
        ax.set_title(column)
        ax.set_xlabel("Standardized residual loss")
        ax.set_ylabel("Frequency")

    for index in range(n_assets, n_rows * n_cols):
        row = index // n_cols
        col = index % n_cols
        axes[row][col].axis("off")

    fig.suptitle("Histograms of Standardized Residual Losses", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_rolling_squared_standardized_residuals(
    standardized_residuals: pd.DataFrame,
    output_path: str | Path,
    window: int = 60,
) -> None:
    """
    Plot rolling mean of squared standardized residuals.

    Parameters
    ----------
    standardized_residuals:
        Wide dataframe of standardized residuals.
    output_path:
        Path where the figure is saved.
    window:
        Rolling window length.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    rolling_squared = standardized_residuals.pow(2).rolling(window=window).mean()

    ax = rolling_squared.plot(
        figsize=(14, 8),
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_title(f"{window}-Day Rolling Mean of Squared Standardized Residuals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling mean of squared standardized residuals")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_evt_shape_threshold_stability(
    gpd_params: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot GPD shape estimates across threshold quantiles.

    Parameters
    ----------
    gpd_params:
        Dataframe containing GPD parameter estimates.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    plot_data = gpd_params.reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))

    for asset, asset_data in plot_data.groupby("asset"):
        asset_data = asset_data.sort_values("threshold_quantile")
        ax.plot(
            asset_data["threshold_quantile"],
            asset_data["shape_xi"],
            marker="o",
            linewidth=1.5,
            label=asset,
        )

    ax.set_title("GPD Shape Parameter across Thresholds")
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel("Shape parameter xi")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_evt_risk_measure_threshold_stability(
    var_es_table: pd.DataFrame,
    probability_level: float,
    risk_measure: str,
    output_path: str | Path,
) -> None:
    """
    Plot EVT VaR or ES estimates across threshold quantiles.

    Parameters
    ----------
    var_es_table:
        Dataframe containing EVT VaR and ES estimates.
    probability_level:
        Probability level to plot.
    risk_measure:
        Either 'residual_var' or 'residual_es'.
    output_path:
        Path where the figure is saved.
    """
    if risk_measure not in {"residual_var", "residual_es"}:
        raise ValueError("risk_measure must be 'residual_var' or 'residual_es'.")

    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    plot_data = var_es_table[
        var_es_table["probability_level"] == probability_level
    ].copy()

    fig, ax = plt.subplots(figsize=(12, 7))

    for asset, asset_data in plot_data.groupby("asset"):
        asset_data = asset_data.sort_values("threshold_quantile")
        ax.plot(
            asset_data["threshold_quantile"],
            asset_data[risk_measure],
            marker="o",
            linewidth=1.5,
            label=asset,
        )

    title_name = "VaR" if risk_measure == "residual_var" else "ES"

    ax.set_title(f"EVT Residual {title_name} across Thresholds, p={probability_level}")
    ax.set_xlabel("Threshold quantile")
    ax.set_ylabel(f"Residual {title_name}")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

