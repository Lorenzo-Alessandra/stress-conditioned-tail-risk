import pandas as pd

from src.models.model_comparison import (
    combine_var_es_comparisons,
    merge_baseline_and_regime_risk,
    summarize_baseline_regime_comparison,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    baseline_var_path = paths_config["processed_files"]["conditional_var_baseline"]
    baseline_es_path = paths_config["processed_files"]["conditional_es_baseline"]

    regime_var_path = paths_config["processed_files"]["conditional_var_regime"]
    regime_es_path = paths_config["processed_files"]["conditional_es_regime"]

    print("Reading baseline conditional VaR...")
    baseline_var = pd.read_parquet(baseline_var_path)

    print("Reading baseline conditional ES...")
    baseline_es = pd.read_parquet(baseline_es_path)

    print("Reading regime-specific conditional VaR...")
    regime_var = pd.read_parquet(regime_var_path)

    print("Reading regime-specific conditional ES...")
    regime_es = pd.read_parquet(regime_es_path)

    print("Merging VaR forecasts...")
    var_merged = merge_baseline_and_regime_risk(
        baseline_risk=baseline_var,
        regime_risk=regime_var,
        value_column="conditional_var",
    )

    print("Merging ES forecasts...")
    es_merged = merge_baseline_and_regime_risk(
        baseline_risk=baseline_es,
        regime_risk=regime_es,
        value_column="conditional_es",
    )

    print("Summarizing VaR comparison...")
    var_summary = summarize_baseline_regime_comparison(
        merged=var_merged,
        value_column="conditional_var",
    )

    print("Summarizing ES comparison...")
    es_summary = summarize_baseline_regime_comparison(
        merged=es_merged,
        value_column="conditional_es",
    )

    combined = combine_var_es_comparisons(
        var_summary=var_summary,
        es_summary=es_summary,
    )

    combined.to_parquet(
        paths_config["processed_files"]["baseline_regime_risk_comparison"]
    )

    output = combined.copy()

    round_columns = [
        "baseline_mean",
        "regime_mean",
        "mean_difference",
        "mean_ratio",
        "baseline_median",
        "regime_median",
        "median_difference",
        "min_difference",
        "max_difference",
    ]

    for column in round_columns:
        output[column] = output[column].round(6)

    csv_path = "outputs/tables/baseline_regime_risk_comparison.csv"
    latex_path = "outputs/tables/baseline_regime_risk_comparison.tex"

    save_table_csv(output, csv_path)

    save_table_latex(
        output,
        latex_path,
        caption="Comparison of baseline and regime-specific conditional VaR and ES forecasts.",
        label="tab:baseline_regime_risk_comparison",
    )

    print("\nBaseline vs regime-specific risk comparison:")
    print(output.to_string(index=False))

    print("\nComparison saved successfully.")
    print(f"Parquet: {paths_config['processed_files']['baseline_regime_risk_comparison']}")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()