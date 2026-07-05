from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def ensure_output_directory(path: str | Path) -> Path:
    """
    Create output directory if it does not already exist.

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


def plot_delta_covar_ranking(
    covar_table: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot Delta CoVaR ranking for one quantile level.

    Parameters
    ----------
    covar_table:
        Empirical CoVaR table.
    quantile_level:
        Quantile level to plot.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    filtered = covar_table[
        covar_table["quantile_level"] == quantile_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No CoVaR rows found for q={quantile_level}.")

    filtered = filtered.sort_values("delta_covar", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        filtered["asset"],
        filtered["delta_covar"],
    )

    ax.set_title(f"Empirical Delta CoVaR Ranking, q={quantile_level}")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Delta CoVaR")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_delta_covar_threshold_robustness(
    covar_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot Delta CoVaR across quantile levels for each asset.

    Parameters
    ----------
    covar_table:
        Empirical CoVaR table.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    required_columns = {"asset", "quantile_level", "delta_covar"}
    missing_columns = required_columns - set(covar_table.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    pivot = covar_table.pivot(
        index="quantile_level",
        columns="asset",
        values="delta_covar",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    for asset in pivot.columns:
        ax.plot(
            pivot.index,
            pivot[asset],
            marker="o",
            linewidth=1.2,
            label=asset,
        )

    ax.set_title("Empirical Delta CoVaR Threshold Robustness")
    ax.set_xlabel("Quantile level")
    ax.set_ylabel("Delta CoVaR")
    ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distress_and_normal_covar(
    covar_table: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot distress CoVaR and normal CoVaR for one quantile level.

    Parameters
    ----------
    covar_table:
        Empirical CoVaR table.
    quantile_level:
        Quantile level to plot.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    filtered = covar_table[
        covar_table["quantile_level"] == quantile_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No CoVaR rows found for q={quantile_level}.")

    filtered = filtered.sort_values("delta_covar", ascending=False)

    x_positions = range(len(filtered))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        filtered["distress_covar"],
        width=bar_width,
        label="Distress CoVaR",
    )

    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        filtered["normal_covar"],
        width=bar_width,
        label="Normal CoVaR",
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(filtered["asset"], rotation=45)

    ax.set_title(f"Distress vs Normal Empirical CoVaR, q={quantile_level}")
    ax.set_xlabel("Bank")
    ax.set_ylabel("System loss quantile")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()