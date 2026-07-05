from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.visualization.plots import ensure_output_directory


def plot_stress_activation_scatter(
    stress_activation: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot Persistent Connectedness versus Stress Activation Index.

    Parameters
    ----------
    stress_activation:
        Stress Activation Index dataframe.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    required_columns = {
        "asset",
        "persistent_connectedness",
        "stress_activation_index",
        "systemic_tail_type",
    }

    missing_columns = required_columns - set(stress_activation.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    plot_data = stress_activation.copy()

    persistent_median = plot_data["persistent_connectedness"].median()
    activation_median = plot_data["stress_activation_index"].median()

    fig, ax = plt.subplots(figsize=(10, 7))

    for systemic_type, group in plot_data.groupby("systemic_tail_type"):
        ax.scatter(
            group["persistent_connectedness"],
            group["stress_activation_index"],
            label=systemic_type,
            s=90,
        )

    for _, row in plot_data.iterrows():
        ax.annotate(
            row["asset"],
            (
                row["persistent_connectedness"],
                row["stress_activation_index"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axvline(persistent_median, linewidth=0.9)
    ax.axhline(activation_median, linewidth=0.9)

    ax.set_title("Persistent Connectedness vs Stress Activation, q=0.95")
    ax.set_xlabel("Persistent Connectedness")
    ax.set_ylabel("Stress Activation Index")
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_stress_activation_bar(
    stress_activation: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot Stress Activation Index by bank.

    Parameters
    ----------
    stress_activation:
        Stress Activation Index dataframe.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    required_columns = {"asset", "stress_activation_index"}
    missing_columns = required_columns - set(stress_activation.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    plot_data = stress_activation.sort_values(
        "stress_activation_index",
        ascending=False,
    ).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_data["asset"],
        plot_data["stress_activation_index"],
    )

    ax.set_title("Stress Activation Index, q=0.95")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Stress Activation Index")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_persistent_connectedness_bar(
    stress_activation: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot Persistent Connectedness by bank.

    Parameters
    ----------
    stress_activation:
        Stress Activation Index dataframe.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    required_columns = {"asset", "persistent_connectedness"}
    missing_columns = required_columns - set(stress_activation.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    plot_data = stress_activation.sort_values(
        "persistent_connectedness",
        ascending=False,
    ).copy()

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        plot_data["asset"],
        plot_data["persistent_connectedness"],
    )

    ax.set_title("Persistent Connectedness, q=0.95")
    ax.set_xlabel("Bank")
    ax.set_ylabel("Persistent Connectedness")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_systemic_tail_type_counts(
    stress_activation: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Plot counts of banks by systemic tail type.

    Parameters
    ----------
    stress_activation:
        Stress Activation Index dataframe.
    output_path:
        Figure output path.
    """
    output_path = Path(output_path)
    ensure_output_directory(output_path.parent)

    if "systemic_tail_type" not in stress_activation.columns:
        raise ValueError("Missing column: systemic_tail_type")

    counts = (
        stress_activation["systemic_tail_type"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_title("Systemic Tail Type Classification, q=0.95")
    ax.set_xlabel("Systemic tail type")
    ax.set_ylabel("Number of banks")
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()