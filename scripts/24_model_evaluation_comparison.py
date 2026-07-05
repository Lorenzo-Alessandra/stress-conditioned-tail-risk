import pandas as pd

from src.models.evaluation_comparison import (
    build_compact_model_evaluation_table,
    build_model_evaluation_comparison,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    baseline_backtest_path = paths_config["processed_files"][
        "var_backtesting_baseline"
    ]
    regime_backtest_path = paths_config["processed_files"][
        "var_backtesting_regime"
    ]

    baseline_es_path = paths_config["processed_files"]["es_scoring_baseline"]
    regime_es_path = paths_config["processed_files"]["es_scoring_regime"]

    print("Reading baseline VaR backtesting results...")
    baseline_backtest = pd.read_parquet(baseline_backtest_path)

    print("Reading regime-specific VaR backtesting results...")
    regime_backtest = pd.read_parquet(regime_backtest_path)

    print("Reading baseline ES scoring results...")
    baseline_es_scores = pd.read_parquet(baseline_es_path)

    print("Reading regime-specific ES scoring results...")
    regime_es_scores = pd.read_parquet(regime_es_path)

    print("Building model evaluation comparison...")
    full_comparison = build_model_evaluation_comparison(
        baseline_backtest=baseline_backtest,
        regime_backtest=regime_backtest,
        baseline_es_scores=baseline_es_scores,
        regime_es_scores=regime_es_scores,
    )

    compact_comparison = build_compact_model_evaluation_table(full_comparison)

    full_comparison.to_parquet(
        paths_config["processed_files"]["model_evaluation_comparison"]
    )

    full_output = full_comparison.copy()
    compact_output = compact_comparison.copy()

    round_columns = [
        "baseline_expected_violations",
        "regime_expected_violations",
        "baseline_violation_rate",
        "regime_violation_rate",
        "baseline_kupiec_p_value",
        "regime_kupiec_p_value",
        "baseline_independence_p_value",
        "regime_independence_p_value",
        "baseline_conditional_coverage_p_value",
        "regime_conditional_coverage_p_value",
        "baseline_mean_es_score",
        "regime_mean_es_score",
        "es_score_difference_regime_minus_baseline",
        "es_score_improvement_percent",
        "baseline_mean_var",
        "regime_mean_var",
        "baseline_mean_es",
        "regime_mean_es",
    ]

    for df in [full_output, compact_output]:
        for column in round_columns:
            if column in df.columns:
                df[column] = df[column].round(6)

    full_csv_path = "outputs/tables/model_evaluation_comparison_full.csv"
    compact_csv_path = "outputs/tables/model_evaluation_comparison_compact.csv"
    compact_latex_path = "outputs/tables/model_evaluation_comparison_compact.tex"

    save_table_csv(full_output, full_csv_path)
    save_table_csv(compact_output, compact_csv_path)

    save_table_latex(
        compact_output,
        compact_latex_path,
        caption="Comparison of baseline and regime-specific GARCH-EVT model diagnostics.",
        label="tab:model_evaluation_comparison",
    )

    print("\nCompact model evaluation comparison:")
    print(compact_output.to_string(index=False))

    print("\nModel evaluation comparison saved successfully.")
    print(f"Full CSV: {full_csv_path}")
    print(f"Compact CSV: {compact_csv_path}")
    print(f"Compact LaTeX: {compact_latex_path}")
    print(f"Full parquet: {paths_config['processed_files']['model_evaluation_comparison']}")


if __name__ == "__main__":
    main()