import pandas as pd

from src.utils.config import load_yaml
from src.visualization.risk_plots import (
    filter_risk_by_probability,
    plot_actual_losses_vs_var,
    plot_conditional_risk_timeseries,
    plot_var_violations,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    losses_path = paths_config["processed_files"]["losses"]
    conditional_var_path = paths_config["processed_files"]["conditional_var_baseline"]
    conditional_es_path = paths_config["processed_files"]["conditional_es_baseline"]

    probability_level = 0.995

    selected_assets = [
        "CBK.DE",
        "DBK.DE",
        "GLE.PA",
        "INGA.AS",
    ]

    print("Reading actual losses...")
    losses = pd.read_parquet(losses_path)

    print("Reading conditional VaR...")
    conditional_var_long = pd.read_parquet(conditional_var_path)

    print("Reading conditional ES...")
    conditional_es_long = pd.read_parquet(conditional_es_path)

    print(f"Filtering conditional VaR at p={probability_level}...")
    conditional_var_panel = filter_risk_by_probability(
        risk_long=conditional_var_long,
        probability_level=probability_level,
        value_column="conditional_var",
    )

    print(f"Filtering conditional ES at p={probability_level}...")
    conditional_es_panel = filter_risk_by_probability(
        risk_long=conditional_es_long,
        probability_level=probability_level,
        value_column="conditional_es",
    )

    print("Creating conditional VaR time-series plot...")
    plot_conditional_risk_timeseries(
        risk_panel=conditional_var_panel,
        probability_level=probability_level,
        risk_name="VaR",
        output_path="outputs/figures/conditional_var_995_timeseries.png",
    )

    print("Creating conditional ES time-series plot...")
    plot_conditional_risk_timeseries(
        risk_panel=conditional_es_panel,
        probability_level=probability_level,
        risk_name="ES",
        output_path="outputs/figures/conditional_es_995_timeseries.png",
    )

    print("Creating actual losses vs VaR plot...")
    plot_actual_losses_vs_var(
        losses=losses,
        var_panel=conditional_var_panel,
        selected_assets=selected_assets,
        probability_level=probability_level,
        output_path="outputs/figures/actual_losses_vs_var_995_selected.png",
    )

    print("Creating VaR violation plot...")
    plot_var_violations(
        losses=losses,
        var_panel=conditional_var_panel,
        selected_assets=selected_assets,
        probability_level=probability_level,
        output_path="outputs/figures/var_violations_995_selected.png",
    )

    print("\nConditional risk plots saved successfully.")
    print("outputs/figures/conditional_var_995_timeseries.png")
    print("outputs/figures/conditional_es_995_timeseries.png")
    print("outputs/figures/actual_losses_vs_var_995_selected.png")
    print("outputs/figures/var_violations_995_selected.png")


if __name__ == "__main__":
    main()