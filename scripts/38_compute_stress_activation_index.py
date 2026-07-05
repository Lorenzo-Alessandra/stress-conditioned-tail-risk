import pandas as pd

from src.models.stress_activation import (
    compute_pairwise_stress_activation,
    compute_stress_activation_index,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    connectedness_comparison_path = (
        "outputs/tables/tail_connectedness_calm_stress_comparison_q95.csv"
    )

    calm_matrix_path = paths_config["processed_files"]["tail_dependence_matrix_calm_q95"]
    stress_matrix_path = paths_config["processed_files"][
        "tail_dependence_matrix_stress_q95"
    ]

    print("Reading calm-stress connectedness comparison...")
    connectedness_comparison = pd.read_csv(connectedness_comparison_path)

    print("Computing Stress Activation Index...")
    stress_activation = compute_stress_activation_index(
        connectedness_comparison=connectedness_comparison,
    )

    print("Reading calm and stress tail-dependence matrices...")
    calm_matrix = pd.read_parquet(calm_matrix_path)
    stress_matrix = pd.read_parquet(stress_matrix_path)

    print("Computing pairwise stress activation...")
    pairwise_activation = compute_pairwise_stress_activation(
        calm_matrix=calm_matrix,
        stress_matrix=stress_matrix,
    )

    stress_activation.to_parquet(
        paths_config["processed_files"]["stress_activation_index_q95"]
    )
    pairwise_activation.to_parquet(
        paths_config["processed_files"]["stress_activation_pairwise_q95"]
    )

    stress_activation_output = stress_activation.copy()
    pairwise_activation_output = pairwise_activation.copy()

    for df in [stress_activation_output, pairwise_activation_output]:
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(6)

    save_table_csv(
        stress_activation_output,
        "outputs/tables/stress_activation_index_q95.csv",
    )

    save_table_csv(
        pairwise_activation_output,
        "outputs/tables/stress_activation_pairwise_q95.csv",
    )

    save_table_latex(
        stress_activation_output,
        "outputs/tables/stress_activation_index_q95.tex",
        caption=(
            "Persistent Connectedness and Stress Activation Index at the "
            "95 percent tail-dependence threshold."
        ),
        label="tab:stress_activation_index_q95",
    )

    print("\nStress Activation Index:")
    print(stress_activation_output.to_string(index=False))

    print("\nLargest pairwise stress activations:")
    print(
        pairwise_activation_output[
            pairwise_activation_output["asset"]
            != pairwise_activation_output["conditioning_asset"]
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nStress Activation Index outputs saved successfully.")


if __name__ == "__main__":
    main()