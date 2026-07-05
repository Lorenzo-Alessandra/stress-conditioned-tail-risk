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


def plot_average_systemic_rank(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot average systemic-risk rank by bank.

    Lower average rank indicates higher systemic relevance.

    Parameters
    ----------
    summary:
        Integrated systemic-risk summary table.
    output_path:
        Figure output path.
    """
    required_columns = {"asset", "average_rank", "systemic_risk_group"}
    missing_columns = required_columns - set(summary.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    plot_data = summary.sort_values("average_rank", ascending=True).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_data["asset"],
        plot_data["average_rank"],
    )

    ax.set_title("Integrated Systemic-Risk Ranking")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Average rank, lower means more systemic")
    ax.tick_params(axis="x", rotation=45)

    for index, row in enumerate(plot_data.itertuples(index=False)):
        ax.text(
            index,
            row.average_rank,
            f"{row.average_rank:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_component_rank_heatmap(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot heatmap of component systemic-risk ranks.

    Parameters
    ----------
    summary:
        Integrated systemic-risk summary table.
    output_path:
        Figure output path.
    """
    rank_columns = [
        "rank_mean_es_995",
        "rank_tail_connectedness_q95",
        "rank_delta_covar_q95",
        "rank_network_weighted_degree_050",
    ]

    required_columns = {"asset"} | set(rank_columns)
    missing_columns = required_columns - set(summary.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    plot_data = summary.sort_values("average_rank", ascending=True).copy()

    rank_matrix = plot_data.set_index("asset")[rank_columns]

    display_columns = [
        "ES 99.5%",
        "Tail conn.",
        "Delta CoVaR",
        "Network degree",
    ]

    rank_matrix.columns = display_columns

    fig, ax = plt.subplots(figsize=(9, 6))

    image = ax.imshow(rank_matrix.values, aspect="auto")

    ax.set_xticks(range(len(rank_matrix.columns)))
    ax.set_yticks(range(len(rank_matrix.index)))

    ax.set_xticklabels(rank_matrix.columns, rotation=30, ha="right")
    ax.set_yticklabels(rank_matrix.index)

    ax.set_title("Component Systemic-Risk Ranks")
    ax.set_xlabel("Component")
    ax.set_ylabel("Bank")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Rank, lower means higher risk")

    for row_index in range(rank_matrix.shape[0]):
        for column_index in range(rank_matrix.shape[1]):
            value = rank_matrix.iloc[row_index, column_index]
            ax.text(
                column_index,
                row_index,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()