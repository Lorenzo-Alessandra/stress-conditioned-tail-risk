import pandas as pd

from src.utils.config import load_yaml
from src.visualization.tail_dependence_plots import (
    plot_calm_vs_stress_connectedness,
    plot_stress_minus_calm_connectedness,
    plot_tail_dependence_difference_heatmap,
    plot_tail_dependence_heatmap,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    calm_matrix_path = paths_config["processed_files"]["tail_dependence_matrix_calm_q95"]
    stress_matrix_path = paths_config["processed_files"][
        "tail_dependence_matrix_stress_q95"
    ]
    difference_matrix_path = paths_config["processed_files"][
        "tail_dependence_matrix_stress_minus_calm_q95"
    ]

    connectedness_comparison_path = (
        "outputs/tables/tail_connectedness_calm_stress_comparison_q95.csv"
    )

    quantile_level = 0.95

    print("Reading calm tail-dependence matrix...")
    calm_matrix = pd.read_parquet(calm_matrix_path)

    print("Reading stress tail-dependence matrix...")
    stress_matrix = pd.read_parquet(stress_matrix_path)

    print("Reading stress-minus-calm tail-dependence matrix...")
    difference_matrix = pd.read_parquet(difference_matrix_path)

    print("Reading calm-stress connectedness comparison...")
    connectedness_comparison = pd.read_csv(connectedness_comparison_path)

    print("Plotting calm tail-dependence heatmap...")
    plot_tail_dependence_heatmap(
        matrix=calm_matrix,
        quantile_level=quantile_level,
        output_path="outputs/figures/tail_dependence_heatmap_calm_q95.png",
        include_values=True,
    )

    print("Plotting stress tail-dependence heatmap...")
    plot_tail_dependence_heatmap(
        matrix=stress_matrix,
        quantile_level=quantile_level,
        output_path="outputs/figures/tail_dependence_heatmap_stress_q95.png",
        include_values=True,
    )

    print("Plotting stress-minus-calm tail-dependence heatmap...")
    plot_tail_dependence_difference_heatmap(
        difference_matrix=difference_matrix,
        quantile_level=quantile_level,
        output_path="outputs/figures/tail_dependence_stress_minus_calm_q95.png",
        include_values=True,
    )

    print("Plotting calm vs stress connectedness...")
    plot_calm_vs_stress_connectedness(
        connectedness_comparison=connectedness_comparison,
        quantile_level=quantile_level,
        output_path="outputs/figures/tail_connectedness_calm_vs_stress_q95.png",
    )

    print("Plotting stress-minus-calm connectedness...")
    plot_stress_minus_calm_connectedness(
        connectedness_comparison=connectedness_comparison,
        quantile_level=quantile_level,
        output_path="outputs/figures/tail_connectedness_stress_minus_calm_q95.png",
    )

    print("\nRegime-specific tail-dependence figures saved successfully.")
    print("outputs/figures/tail_dependence_heatmap_calm_q95.png")
    print("outputs/figures/tail_dependence_heatmap_stress_q95.png")
    print("outputs/figures/tail_dependence_stress_minus_calm_q95.png")
    print("outputs/figures/tail_connectedness_calm_vs_stress_q95.png")
    print("outputs/figures/tail_connectedness_stress_minus_calm_q95.png")


if __name__ == "__main__":
    main()