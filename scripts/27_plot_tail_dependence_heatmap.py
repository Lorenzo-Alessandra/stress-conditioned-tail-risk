import pandas as pd

from src.utils.config import load_yaml
from src.visualization.tail_dependence_plots import (
    plot_average_tail_connectedness_bar,
    plot_tail_dependence_heatmap,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    matrix_path = paths_config["processed_files"]["tail_dependence_matrix_q95"]

    matrix_csv_path = "outputs/tables/tail_dependence_matrix_q95.csv"
    connectedness_csv_path = "outputs/tables/tail_connectedness_summary.csv"

    quantile_level = 0.95

    print("Reading tail-dependence matrix...")
    matrix = pd.read_parquet(matrix_path)

    print("Reading average connectedness summary...")
    connectedness = pd.read_csv(connectedness_csv_path)

    corrected_matrix_csv_path = "outputs/tables/tail_dependence_matrix_q95_with_index.csv"
    matrix.to_csv(corrected_matrix_csv_path, index=True, index_label="asset")

    print("Plotting tail-dependence heatmap...")
    plot_tail_dependence_heatmap(
        matrix=matrix,
        quantile_level=quantile_level,
        output_path="outputs/figures/tail_dependence_heatmap_q95.png",
        include_values=True,
    )

    print("Plotting average tail connectedness ranking...")
    plot_average_tail_connectedness_bar(
        connectedness=connectedness,
        quantile_level=quantile_level,
        output_path="outputs/figures/tail_connectedness_bar_q95.png",
    )

    print("\nTail-dependence figures saved successfully.")
    print("outputs/figures/tail_dependence_heatmap_q95.png")
    print("outputs/figures/tail_connectedness_bar_q95.png")
    print(corrected_matrix_csv_path)


if __name__ == "__main__":
    main()