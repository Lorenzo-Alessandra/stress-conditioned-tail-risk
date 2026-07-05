import pandas as pd

from src.models.backtesting import run_var_backtests
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    losses_path = paths_config["processed_files"]["losses"]
    conditional_var_path = paths_config["processed_files"]["conditional_var_regime"]

    print("Reading actual losses...")
    losses = pd.read_parquet(losses_path)

    print("Reading regime-specific conditional VaR forecasts...")
    conditional_var = pd.read_parquet(conditional_var_path)

    print("Running regime-specific VaR backtests...")
    backtest_results, violations = run_var_backtests(
        losses=losses,
        conditional_var_long=conditional_var,
    )

    backtest_results.to_parquet(
        paths_config["processed_files"]["var_backtesting_regime"]
    )

    violations.to_parquet(
        paths_config["processed_files"]["var_violations_regime"]
    )

    output = backtest_results.copy()

    round_columns = [
        "expected_violations",
        "violation_rate",
        "expected_violation_rate",
        "kupiec_lr",
        "kupiec_p_value",
        "independence_lr",
        "independence_p_value",
        "conditional_coverage_lr",
        "conditional_coverage_p_value",
    ]

    for column in round_columns:
        output[column] = output[column].round(6)

    csv_path = "outputs/tables/var_backtesting_regime.csv"
    latex_path = "outputs/tables/var_backtesting_regime.tex"

    save_table_csv(output, csv_path)

    save_table_latex(
        output,
        latex_path,
        caption="VaR backtesting results for regime-specific GARCH-EVT forecasts.",
        label="tab:var_backtesting_regime",
    )

    print("\nRegime-specific VaR backtesting results:")
    print(output.to_string(index=False))

    print("\nRegime-specific backtesting outputs saved successfully.")
    print(f"Backtesting parquet: {paths_config['processed_files']['var_backtesting_regime']}")
    print(f"Violation parquet: {paths_config['processed_files']['var_violations_regime']}")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()