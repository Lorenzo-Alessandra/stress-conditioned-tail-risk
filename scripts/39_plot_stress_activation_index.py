import pandas as pd

from src.utils.config import load_yaml
from src.visualization.stress_activation_plots import (
    plot_persistent_connectedness_bar,
    plot_stress_activation_bar,
    plot_stress_activation_scatter,
    plot_systemic_tail_type_counts,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    stress_activation_path = paths_config["processed_files"][
        "stress_activation_index_q95"
    ]

    print("Reading Stress Activation Index table...")
    stress_activation = pd.read_parquet(stress_activation_path)

    print("Plotting Persistent Connectedness vs Stress Activation scatter...")
    plot_stress_activation_scatter(
        stress_activation=stress_activation,
        output_path="outputs/figures/stress_activation_scatter_q95.png",
    )

    print("Plotting Stress Activation Index bar chart...")
    plot_stress_activation_bar(
        stress_activation=stress_activation,
        output_path="outputs/figures/stress_activation_index_bar_q95.png",
    )

    print("Plotting Persistent Connectedness bar chart...")
    plot_persistent_connectedness_bar(
        stress_activation=stress_activation,
        output_path="outputs/figures/persistent_connectedness_bar_q95.png",
    )

    print("Plotting systemic tail type counts...")
    plot_systemic_tail_type_counts(
        stress_activation=stress_activation,
        output_path="outputs/figures/systemic_tail_type_classification_q95.png",
    )

    print("\nStress Activation Index figures saved successfully.")
    print("outputs/figures/stress_activation_scatter_q95.png")
    print("outputs/figures/stress_activation_index_bar_q95.png")
    print("outputs/figures/persistent_connectedness_bar_q95.png")
    print("outputs/figures/systemic_tail_type_classification_q95.png")


if __name__ == "__main__":
    main()