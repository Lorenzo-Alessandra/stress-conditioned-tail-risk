import pandas as pd

from src.models.tail_dependence import (
    build_tail_dependence_matrix,
    compute_pairwise_tail_dependence,
    summarize_average_tail_connectedness,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    losses_path = paths_config["processed_files"]["losses"]

    quantile_levels = model_config["tail_dependence"]["q_levels"]
    main_quantile_level = 0.95

    print("Reading bank losses...")
    losses = pd.read_parquet(losses_path)

    print("Computing pairwise tail dependence...")
    pairwise = compute_pairwise_tail_dependence(
        losses=losses,
        quantile_levels=quantile_levels,
    )

    print(f"Building tail-dependence matrix for q={main_quantile_level}...")
    matrix_q95 = build_tail_dependence_matrix(
        pairwise_tail_dependence=pairwise,
        quantile_level=main_quantile_level,
    )

    print("Summarizing average tail connectedness...")
    connectedness = summarize_average_tail_connectedness(pairwise)

    pairwise.to_parquet(paths_config["processed_files"]["tail_dependence_pairwise"])
    matrix_q95.to_parquet(paths_config["processed_files"]["tail_dependence_matrix_q95"])

    pairwise_output = pairwise.copy()
    matrix_output = matrix_q95.copy()
    connectedness_output = connectedness.copy()

    for df in [pairwise_output, matrix_output, connectedness_output]:
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(6)

    pairwise_csv_path = "outputs/tables/tail_dependence_pairwise.csv"
    matrix_csv_path = "outputs/tables/tail_dependence_matrix_q95.csv"
    connectedness_csv_path = "outputs/tables/tail_connectedness_summary.csv"

    pairwise_latex_path = "outputs/tables/tail_dependence_pairwise.tex"
    matrix_latex_path = "outputs/tables/tail_dependence_matrix_q95.tex"
    connectedness_latex_path = "outputs/tables/tail_connectedness_summary.tex"

    save_table_csv(pairwise_output, pairwise_csv_path)
    save_table_csv(matrix_output, matrix_csv_path)
    save_table_csv(connectedness_output, connectedness_csv_path)

    save_table_latex(
        matrix_output,
        matrix_latex_path,
        caption="Pairwise tail-dependence matrix at the 95 percent loss threshold.",
        label="tab:tail_dependence_matrix_q95",
    )

    save_table_latex(
        connectedness_output,
        connectedness_latex_path,
        caption="Average incoming and outgoing tail connectedness.",
        label="tab:tail_connectedness_summary",
    )

    save_table_latex(
        pairwise_output,
        pairwise_latex_path,
        caption="Pairwise directional tail-dependence estimates.",
        label="tab:tail_dependence_pairwise",
    )

    print("\nTail-dependence matrix, q=0.95:")
    print(matrix_output.to_string())

    print("\nAverage tail connectedness:")
    print(connectedness_output.to_string(index=False))

    print("\nTail-dependence outputs saved successfully.")
    print(pairwise_csv_path)
    print(matrix_csv_path)
    print(connectedness_csv_path)


if __name__ == "__main__":
    main()