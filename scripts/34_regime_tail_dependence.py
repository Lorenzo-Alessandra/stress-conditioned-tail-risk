import pandas as pd

from src.models.tail_dependence import (
    build_regime_tail_dependence_matrix,
    compare_calm_stress_connectedness,
    compute_pairwise_tail_dependence_by_regime,
    compute_stress_minus_calm_matrix,
    summarize_regime_tail_connectedness,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    losses_path = paths_config["processed_files"]["losses"]
    stress_regime_path = paths_config["processed_files"]["stress_regime"]

    quantile_levels = model_config["tail_dependence"]["q_levels"]
    main_quantile_level = 0.95

    print("Reading bank losses...")
    losses = pd.read_parquet(losses_path)

    print("Reading stress regime...")
    stress_regime = pd.read_parquet(stress_regime_path)

    print("Computing regime-specific pairwise tail dependence...")
    pairwise_regime = compute_pairwise_tail_dependence_by_regime(
        losses=losses,
        stress_regime=stress_regime,
        quantile_levels=quantile_levels,
    )

    print("Building calm and stress tail-dependence matrices at q=0.95...")
    calm_matrix = build_regime_tail_dependence_matrix(
        pairwise_tail_dependence_regime=pairwise_regime,
        quantile_level=main_quantile_level,
        stress_regime=0,
    )

    stress_matrix = build_regime_tail_dependence_matrix(
        pairwise_tail_dependence_regime=pairwise_regime,
        quantile_level=main_quantile_level,
        stress_regime=1,
    )

    difference_matrix = compute_stress_minus_calm_matrix(
        calm_matrix=calm_matrix,
        stress_matrix=stress_matrix,
    )

    print("Summarizing regime-specific connectedness...")
    connectedness_regime = summarize_regime_tail_connectedness(pairwise_regime)

    print("Comparing calm and stress connectedness at q=0.95...")
    connectedness_comparison_q95 = compare_calm_stress_connectedness(
        regime_connectedness=connectedness_regime,
        quantile_level=main_quantile_level,
    )

    pairwise_regime.to_parquet(
        paths_config["processed_files"]["tail_dependence_pairwise_regime"]
    )
    calm_matrix.to_parquet(
        paths_config["processed_files"]["tail_dependence_matrix_calm_q95"]
    )
    stress_matrix.to_parquet(
        paths_config["processed_files"]["tail_dependence_matrix_stress_q95"]
    )
    difference_matrix.to_parquet(
        paths_config["processed_files"][
            "tail_dependence_matrix_stress_minus_calm_q95"
        ]
    )
    connectedness_regime.to_parquet(
        paths_config["processed_files"]["tail_connectedness_regime_summary"]
    )

    outputs = {
        "pairwise_regime": pairwise_regime.copy(),
        "calm_matrix": calm_matrix.copy(),
        "stress_matrix": stress_matrix.copy(),
        "difference_matrix": difference_matrix.copy(),
        "connectedness_regime": connectedness_regime.copy(),
        "connectedness_comparison_q95": connectedness_comparison_q95.copy(),
    }

    for df in outputs.values():
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(6)

    save_table_csv(
        outputs["pairwise_regime"],
        "outputs/tables/tail_dependence_pairwise_regime.csv",
    )
    save_table_csv(
        outputs["calm_matrix"],
        "outputs/tables/tail_dependence_matrix_calm_q95.csv",
    )
    save_table_csv(
        outputs["stress_matrix"],
        "outputs/tables/tail_dependence_matrix_stress_q95.csv",
    )
    save_table_csv(
        outputs["difference_matrix"],
        "outputs/tables/tail_dependence_matrix_stress_minus_calm_q95.csv",
    )
    save_table_csv(
        outputs["connectedness_regime"],
        "outputs/tables/tail_connectedness_regime_summary.csv",
    )
    save_table_csv(
        outputs["connectedness_comparison_q95"],
        "outputs/tables/tail_connectedness_calm_stress_comparison_q95.csv",
    )

    save_table_latex(
        outputs["connectedness_comparison_q95"],
        "outputs/tables/tail_connectedness_calm_stress_comparison_q95.tex",
        caption="Comparison of calm and stress tail connectedness at the 95 percent loss threshold.",
        label="tab:tail_connectedness_calm_stress_comparison_q95",
    )

    print("\nCalm tail-dependence matrix, q=0.95:")
    print(outputs["calm_matrix"].to_string())

    print("\nStress tail-dependence matrix, q=0.95:")
    print(outputs["stress_matrix"].to_string())

    print("\nStress minus calm tail-dependence matrix, q=0.95:")
    print(outputs["difference_matrix"].to_string())

    print("\nCalm vs stress connectedness comparison, q=0.95:")
    print(outputs["connectedness_comparison_q95"].to_string(index=False))

    print("\nRegime-specific tail dependence outputs saved successfully.")


if __name__ == "__main__":
    main()