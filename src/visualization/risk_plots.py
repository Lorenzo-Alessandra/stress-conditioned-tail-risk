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


def filter_risk_by_probability(
    risk_long: pd.DataFrame,
    probability_level: float,
    value_column: str,
) -> pd.DataFrame:
    """
    Convert a long conditional-risk table into a wide panel for one probability level.

    Parameters
    ----------
    risk_long:
        Long risk dataframe with date, asset, probability_level, and value column.
    probability_level:
        Probability level to keep.
    value_column:
        Name of the risk value column.

    Returns
    -------
    pd.DataFrame
        Wide risk panel indexed by date with assets as columns.
    """
    required_columns = {"date", "asset", "probability_level", value_column}
    missing_columns = required_columns - set(risk_long.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    filtered = risk_long[risk_long["probability_level"] == probability_level].copy()

    if filtered.empty:
        raise ValueError(f"No rows found for probability level {probability_level}.")

    wide = filtered.pivot(
        index="date",
        columns="asset",
        values=value_column,
    )

    wide = wide.sort_index()

    return wide


def plot_conditional_risk_timeseries(
    risk_panel: pd.DataFrame,
    probability_level: float,
    risk_name: str,
    output_path: str | Path,
) -> None:
    """
    Plot conditional VaR or ES time series for all banks.

    Parameters
    ----------
    risk_panel:
        Wide conditional risk panel.
    probability_level:
        Probability level.
    risk_name:
        Name of the risk measure, e.g. 'VaR' or 'ES'.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    ax = risk_panel.plot(
        figsize=(14, 8),
        linewidth=0.8,
        alpha=0.9,
    )

    ax.set_title(f"Conditional {risk_name}, p={probability_level}")
    ax.set_xlabel("Date")
    ax.set_ylabel(f"Conditional {risk_name}")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_actual_losses_vs_var(
    losses: pd.DataFrame,
    var_panel: pd.DataFrame,
    selected_assets: list[str],
    probability_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot actual losses against conditional VaR for selected banks.

    Parameters
    ----------
    losses:
        Wide actual loss panel.
    var_panel:
        Wide conditional VaR panel.
    selected_assets:
        Assets to plot.
    probability_level:
        VaR probability level.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    n_assets = len(selected_assets)

    fig, axes = plt.subplots(
        n_assets,
        1,
        figsize=(14, 4 * n_assets),
        sharex=True,
    )

    if n_assets == 1:
        axes = [axes]

    for ax, asset in zip(axes, selected_assets):
        if asset not in losses.columns:
            raise ValueError(f"Asset not found in losses: {asset}")

        if asset not in var_panel.columns:
            raise ValueError(f"Asset not found in VaR panel: {asset}")

        aligned = pd.concat(
            [
                losses[asset].rename("loss"),
                var_panel[asset].rename("var"),
            ],
            axis=1,
        ).dropna()

        ax.plot(
            aligned.index,
            aligned["loss"],
            linewidth=0.7,
            label="Actual loss",
        )

        ax.plot(
            aligned.index,
            aligned["var"],
            linewidth=1.0,
            label=f"VaR p={probability_level}",
        )

        ax.set_title(asset)
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Date")

    fig.suptitle(f"Actual Losses vs Conditional VaR, p={probability_level}", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_var_violations(
    losses: pd.DataFrame,
    var_panel: pd.DataFrame,
    selected_assets: list[str],
    probability_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot VaR violations for selected banks.

    Parameters
    ----------
    losses:
        Wide actual loss panel.
    var_panel:
        Wide conditional VaR panel.
    selected_assets:
        Assets to plot.
    probability_level:
        VaR probability level.
    output_path:
        Path where the figure is saved.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    n_assets = len(selected_assets)

    fig, axes = plt.subplots(
        n_assets,
        1,
        figsize=(14, 3.5 * n_assets),
        sharex=True,
    )

    if n_assets == 1:
        axes = [axes]

    for ax, asset in zip(axes, selected_assets):
        if asset not in losses.columns:
            raise ValueError(f"Asset not found in losses: {asset}")

        if asset not in var_panel.columns:
            raise ValueError(f"Asset not found in VaR panel: {asset}")

        aligned = pd.concat(
            [
                losses[asset].rename("loss"),
                var_panel[asset].rename("var"),
            ],
            axis=1,
        ).dropna()

        violations = aligned["loss"] > aligned["var"]

        ax.plot(
            aligned.index,
            aligned["loss"],
            linewidth=0.7,
            label="Actual loss",
        )

        ax.plot(
            aligned.index,
            aligned["var"],
            linewidth=1.0,
            label=f"VaR p={probability_level}",
        )

        ax.scatter(
            aligned.index[violations],
            aligned.loc[violations, "loss"],
            s=20,
            label="Violation",
        )

        ax.set_title(
            f"{asset}: {violations.sum()} violations out of {len(violations)} observations"
        )
        ax.set_ylabel("Loss")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Date")

    fig.suptitle(f"VaR Violations, p={probability_level}", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def merge_baseline_regime_for_plot(
    baseline_long: pd.DataFrame,
    regime_long: pd.DataFrame,
    probability_level: float,
    value_column: str,
) -> pd.DataFrame:
    """
    Merge baseline and regime-specific risk forecasts for plotting.

    Parameters
    ----------
    baseline_long:
        Long baseline risk forecast dataframe.
    regime_long:
        Long regime-specific risk forecast dataframe.
    probability_level:
        Probability level to plot.
    value_column:
        Risk value column name, such as conditional_var or conditional_es.

    Returns
    -------
    pd.DataFrame
        Merged dataframe containing baseline, regime, stress regime, and difference.
    """
    baseline_required = {"date", "asset", "probability_level", value_column}
    regime_required = {
        "date",
        "asset",
        "probability_level",
        "stress_regime",
        value_column,
    }

    missing_baseline = baseline_required - set(baseline_long.columns)
    missing_regime = regime_required - set(regime_long.columns)

    if missing_baseline:
        raise ValueError(f"Missing baseline columns: {missing_baseline}")

    if missing_regime:
        raise ValueError(f"Missing regime columns: {missing_regime}")

    baseline = baseline_long[
        baseline_long["probability_level"] == probability_level
    ].copy()

    regime = regime_long[
        regime_long["probability_level"] == probability_level
    ].copy()

    baseline = baseline[
        ["date", "asset", "probability_level", value_column]
    ].rename(columns={value_column: f"{value_column}_baseline"})

    regime = regime[
        ["date", "asset", "probability_level", "stress_regime", value_column]
    ].rename(columns={value_column: f"{value_column}_regime"})

    merged = baseline.merge(
        regime,
        on=["date", "asset", "probability_level"],
        how="inner",
    )

    merged[f"{value_column}_difference"] = (
        merged[f"{value_column}_regime"]
        - merged[f"{value_column}_baseline"]
    )

    merged = merged.sort_values(["asset", "date"]).reset_index(drop=True)

    return merged


def plot_baseline_vs_regime_risk(
    merged: pd.DataFrame,
    selected_assets: list[str],
    value_column: str,
    risk_name: str,
    probability_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot baseline and regime-specific risk forecasts for selected assets.

    Parameters
    ----------
    merged:
        Merged baseline-regime risk dataframe.
    selected_assets:
        Assets to plot.
    value_column:
        Risk value column name.
    risk_name:
        Display name, such as VaR or ES.
    probability_level:
        Probability level.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    baseline_col = f"{value_column}_baseline"
    regime_col = f"{value_column}_regime"

    n_assets = len(selected_assets)

    fig, axes = plt.subplots(
        n_assets,
        1,
        figsize=(14, 4 * n_assets),
        sharex=True,
    )

    if n_assets == 1:
        axes = [axes]

    for ax, asset in zip(axes, selected_assets):
        asset_data = merged[merged["asset"] == asset].copy()

        if asset_data.empty:
            raise ValueError(f"No data found for asset: {asset}")

        ax.plot(
            asset_data["date"],
            asset_data[baseline_col],
            linewidth=0.9,
            label="Baseline static EVT",
        )

        ax.plot(
            asset_data["date"],
            asset_data[regime_col],
            linewidth=0.9,
            label="Regime-specific EVT",
        )

        stress_days = asset_data["stress_regime"] == 1

        ax.scatter(
            asset_data.loc[stress_days, "date"],
            asset_data.loc[stress_days, regime_col],
            s=4,
            alpha=0.4,
            label="Stress regime",
        )

        ax.set_title(asset)
        ax.set_ylabel(risk_name)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Date")

    fig.suptitle(
        f"Baseline vs Regime-Specific Conditional {risk_name}, p={probability_level}",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_regime_minus_baseline_difference(
    merged: pd.DataFrame,
    selected_assets: list[str],
    value_column: str,
    risk_name: str,
    probability_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot regime-specific minus baseline risk differences for selected assets.

    Parameters
    ----------
    merged:
        Merged baseline-regime risk dataframe.
    selected_assets:
        Assets to plot.
    value_column:
        Risk value column name.
    risk_name:
        Display name, such as VaR or ES.
    probability_level:
        Probability level.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    difference_col = f"{value_column}_difference"

    n_assets = len(selected_assets)

    fig, axes = plt.subplots(
        n_assets,
        1,
        figsize=(14, 3.5 * n_assets),
        sharex=True,
    )

    if n_assets == 1:
        axes = [axes]

    for ax, asset in zip(axes, selected_assets):
        asset_data = merged[merged["asset"] == asset].copy()

        if asset_data.empty:
            raise ValueError(f"No data found for asset: {asset}")

        ax.plot(
            asset_data["date"],
            asset_data[difference_col],
            linewidth=0.8,
            label="Regime minus baseline",
        )

        ax.axhline(0.0, linewidth=0.8)

        stress_days = asset_data["stress_regime"] == 1

        ax.scatter(
            asset_data.loc[stress_days, "date"],
            asset_data.loc[stress_days, difference_col],
            s=4,
            alpha=0.4,
            label="Stress regime",
        )

        ax.set_title(asset)
        ax.set_ylabel(f"Difference in {risk_name}")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Date")

    fig.suptitle(
        f"Regime-Specific minus Baseline Conditional {risk_name}, p={probability_level}",
        y=1.01,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()