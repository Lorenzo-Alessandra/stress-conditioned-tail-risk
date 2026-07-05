import pandas as pd

from src.models.copula import compute_tail_dependence_from_uniforms
from src.utils.config import load_yaml
from src.visualization.tail_dependence_plots import (
    plot_empirical_minus_copula_heatmap,
    plot_largest_copula_tail_dependence_errors,
    plot_tail_dependence_heatmap,
)


def build_matrix_from_comparison(
    comparison: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Build a square matrix from a long empirical-copula comparison table.

    Parameters
    ----------
    comparison:
        Long comparison table.
    value_column:
        Column to place into the matrix.

    Returns
    -------
    pd.DataFrame
        Square matrix.
    """
    matrix = comparison.pivot(
        index="asset",
        columns="conditioning_asset",
        values=value_column,
    )

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    return matrix


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    empirical_matrix_path = paths_config["processed_files"]["tail_dependence_matrix_q95"]
    simulated_tail_dependence_path = paths_config["processed_files"][
        "student_t_copula_simulated_tail_dependence"
    ]
    comparison_path = paths_config["processed_files"][
        "student_t_copula_empirical_comparison_q95"
    ]

    quantile_level = 0.95

    print("Reading empirical q=0.95 tail-dependence matrix...")
    empirical_matrix = pd.read_parquet(empirical_matrix_path)

    print("Reading Student-t copula simulated tail-dependence table...")
    simulated_tail_dependence = pd.read_parquet(simulated_tail_dependence_path)

    print("Reading empirical-copula comparison table...")
    comparison = pd.read_parquet(comparison_path)

    print("Building simulated Student-t copula q=0.95 matrix...")
    simulated_q95 = simulated_tail_dependence[
        simulated_tail_dependence["quantile_level"] == quantile_level
    ].copy()

    copula_matrix = simulated_q95.pivot(
        index="asset",
        columns="conditioning_asset",
        values="simulated_tail_dependence",
    )

    copula_matrix = copula_matrix.sort_index(axis=0).sort_index(axis=1)

    print("Building empirical minus copula matrix...")
    empirical_minus_copula = build_matrix_from_comparison(
        comparison=comparison,
        value_column="copula_minus_empirical",
    )

    empirical_minus_copula = -empirical_minus_copula

    copula_matrix.to_csv(
        "outputs/tables/student_t_copula_tail_dependence_matrix_q95.csv",
        index=True,
        index_label="asset",
    )

    empirical_minus_copula.to_csv(
        "outputs/tables/empirical_minus_student_t_copula_tail_dependence_q95.csv",
        index=True,
        index_label="asset",
    )

    print("Plotting Student-t copula q=0.95 tail-dependence heatmap...")
    plot_tail_dependence_heatmap(
        matrix=copula_matrix,
        quantile_level=quantile_level,
        output_path="outputs/figures/student_t_copula_tail_dependence_heatmap_q95.png",
        include_values=True,
    )

    print("Plotting empirical minus Student-t copula heatmap...")
    plot_empirical_minus_copula_heatmap(
        comparison_matrix=empirical_minus_copula,
        quantile_level=quantile_level,
        output_path="outputs/figures/empirical_minus_student_t_copula_tail_dependence_q95.png",
        include_values=True,
    )

    print("Plotting largest copula errors...")
    plot_largest_copula_tail_dependence_errors(
        comparison=comparison,
        output_path="outputs/figures/student_t_copula_largest_errors_q95.png",
        top_n=15,
    )

    print("\nStudent-t copula benchmark figures saved successfully.")
    print("outputs/figures/student_t_copula_tail_dependence_heatmap_q95.png")
    print("outputs/figures/empirical_minus_student_t_copula_tail_dependence_q95.png")
    print("outputs/figures/student_t_copula_largest_errors_q95.png")
    print("outputs/tables/student_t_copula_tail_dependence_matrix_q95.csv")
    print("outputs/tables/empirical_minus_student_t_copula_tail_dependence_q95.csv")


if __name__ == "__main__":
    main()