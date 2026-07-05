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


def plot_tail_dependence_heatmap(
    matrix: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
    include_values: bool = True,
) -> None:
    """
    Plot a pairwise tail-dependence heatmap.

    Parameters
    ----------
    matrix:
        Square tail-dependence matrix.
        Rows are affected assets and columns are conditioning assets.
    quantile_level:
        Tail threshold quantile level.
    output_path:
        Figure output path.
    include_values:
        Whether to print numerical values inside cells.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(matrix.values, aspect="auto")

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))

    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)

    ax.set_xlabel("Conditioning bank in tail distress")
    ax.set_ylabel("Bank in tail distress")
    ax.set_title(f"Pairwise Tail Dependence Heatmap, q={quantile_level}")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Tail dependence")

    if include_values:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_average_tail_connectedness_bar(
    connectedness: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot average total tail connectedness by asset.

    Parameters
    ----------
    connectedness:
        Tail connectedness summary table.
    quantile_level:
        Quantile level to plot.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    filtered = connectedness[
        connectedness["quantile_level"] == quantile_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No connectedness rows found for q={quantile_level}.")

    filtered = filtered.sort_values(
        "average_total_tail_connectedness",
        ascending=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        filtered["asset"],
        filtered["average_total_tail_connectedness"],
    )

    ax.set_title(f"Average Tail Connectedness, q={quantile_level}")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Average tail connectedness")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_tail_dependence_difference_heatmap(
    difference_matrix: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
    include_values: bool = True,
) -> None:
    """
    Plot a stress-minus-calm tail-dependence heatmap.

    Parameters
    ----------
    difference_matrix:
        Matrix of stress tail dependence minus calm tail dependence.
    quantile_level:
        Tail threshold quantile level.
    output_path:
        Figure output path.
    include_values:
        Whether to print numerical values inside cells.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(difference_matrix.values, aspect="auto")

    ax.set_xticks(range(len(difference_matrix.columns)))
    ax.set_yticks(range(len(difference_matrix.index)))

    ax.set_xticklabels(difference_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(difference_matrix.index)

    ax.set_xlabel("Conditioning bank in tail distress")
    ax.set_ylabel("Bank in tail distress")
    ax.set_title(f"Stress minus Calm Tail Dependence, q={quantile_level}")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Stress minus calm tail dependence")

    if include_values:
        for row_index in range(difference_matrix.shape[0]):
            for column_index in range(difference_matrix.shape[1]):
                value = difference_matrix.iloc[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_calm_vs_stress_connectedness(
    connectedness_comparison: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot calm versus stress average tail connectedness.

    Parameters
    ----------
    connectedness_comparison:
        Table with connectedness_calm and connectedness_stress columns.
    quantile_level:
        Quantile level.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    required_columns = {
        "asset",
        "connectedness_calm",
        "connectedness_stress",
        "connectedness_stress_minus_calm",
    }

    missing_columns = required_columns - set(connectedness_comparison.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    plot_data = connectedness_comparison.sort_values(
        "connectedness_stress_minus_calm",
        ascending=False,
    ).copy()

    x_positions = range(len(plot_data))
    bar_width = 0.4

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(
        [x - bar_width / 2 for x in x_positions],
        plot_data["connectedness_calm"],
        width=bar_width,
        label="Calm",
    )

    ax.bar(
        [x + bar_width / 2 for x in x_positions],
        plot_data["connectedness_stress"],
        width=bar_width,
        label="Stress",
    )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(plot_data["asset"], rotation=45)

    ax.set_title(f"Calm vs Stress Tail Connectedness, q={quantile_level}")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Average tail connectedness")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stress_minus_calm_connectedness(
    connectedness_comparison: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
) -> None:
    """
    Plot stress-minus-calm average connectedness by bank.

    Parameters
    ----------
    connectedness_comparison:
        Table with connectedness_stress_minus_calm.
    quantile_level:
        Quantile level.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    required_columns = {"asset", "connectedness_stress_minus_calm"}
    missing_columns = required_columns - set(connectedness_comparison.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    plot_data = connectedness_comparison.sort_values(
        "connectedness_stress_minus_calm",
        ascending=False,
    ).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_data["asset"],
        plot_data["connectedness_stress_minus_calm"],
    )

    ax.axhline(0.0, linewidth=0.8)

    ax.set_title(f"Stress minus Calm Tail Connectedness, q={quantile_level}")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Stress minus calm connectedness")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_empirical_minus_copula_heatmap(
    comparison_matrix: pd.DataFrame,
    quantile_level: float,
    output_path: str | Path,
    include_values: bool = True,
) -> None:
    """
    Plot empirical minus copula-implied tail-dependence heatmap.

    Parameters
    ----------
    comparison_matrix:
        Matrix of empirical tail dependence minus copula tail dependence.
    quantile_level:
        Tail threshold quantile level.
    output_path:
        Figure output path.
    include_values:
        Whether to print numerical values inside cells.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    fig, ax = plt.subplots(figsize=(10, 8))

    image = ax.imshow(comparison_matrix.values, aspect="auto")

    ax.set_xticks(range(len(comparison_matrix.columns)))
    ax.set_yticks(range(len(comparison_matrix.index)))

    ax.set_xticklabels(comparison_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(comparison_matrix.index)

    ax.set_xlabel("Conditioning bank in tail distress")
    ax.set_ylabel("Bank in tail distress")
    ax.set_title(f"Empirical minus Student-t Copula Tail Dependence, q={quantile_level}")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Empirical minus copula tail dependence")

    if include_values:
        for row_index in range(comparison_matrix.shape[0]):
            for column_index in range(comparison_matrix.shape[1]):
                value = comparison_matrix.iloc[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_largest_copula_tail_dependence_errors(
    comparison: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """
    Plot largest absolute empirical-copula tail-dependence errors.

    Parameters
    ----------
    comparison:
        Long comparison table containing empirical and copula tail dependence.
    output_path:
        Figure output path.
    top_n:
        Number of largest absolute errors to plot.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    required_columns = {
        "asset",
        "conditioning_asset",
        "empirical_tail_dependence",
        "copula_tail_dependence",
        "copula_minus_empirical",
        "absolute_difference",
    }

    missing_columns = required_columns - set(comparison.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    plot_data = comparison[
        comparison["asset"] != comparison["conditioning_asset"]
    ].copy()

    plot_data = plot_data.sort_values(
        "absolute_difference",
        ascending=False,
    ).head(top_n)

    plot_data["pair"] = (
        plot_data["asset"] + " | " + plot_data["conditioning_asset"]
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.barh(
        plot_data["pair"],
        plot_data["copula_minus_empirical"],
    )

    ax.axvline(0.0, linewidth=0.8)

    ax.set_title("Largest Student-t Copula Tail-Dependence Errors, q=0.95")
    ax.set_xlabel("Copula minus empirical tail dependence")
    ax.set_ylabel("Pair")

    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()