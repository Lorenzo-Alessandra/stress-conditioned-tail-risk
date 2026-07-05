import pandas as pd

from src.utils.config import load_yaml
from src.visualization.risk_plots import (
    merge_baseline_regime_for_plot,
    plot_baseline_vs_regime_risk,
    plot_regime_minus_baseline_difference,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    baseline_var_path = paths_config["processed_files"]["conditional_var_baseline"]
    baseline_es_path = paths_config["processed_files"]["conditional_es_baseline"]

    regime_var_path = paths_config["processed_files"]["conditional_var_regime"]
    regime_es_path = paths_config["processed_files"]["conditional_es_regime"]

    probability_level = 0.995

    selected_assets = [
        "CBK.DE",
        "DBK.DE",
        "GLE.PA",
        "INGA.AS",
    ]

    print("Reading baseline conditional VaR...")
    baseline_var = pd.read_parquet(baseline_var_path)

    print("Reading baseline conditional ES...")
    baseline_es = pd.read_parquet(baseline_es_path)

    print("Reading regime-specific conditional VaR...")
    regime_var = pd.read_parquet(regime_var_path)

    print("Reading regime-specific conditional ES...")
    regime_es = pd.read_parquet(regime_es_path)

    print("Merging VaR forecasts for plotting...")
    var_merged = merge_baseline_regime_for_plot(
        baseline_long=baseline_var,
        regime_long=regime_var,
        probability_level=probability_level,
        value_column="conditional_var",
    )

    print("Merging ES forecasts for plotting...")
    es_merged = merge_baseline_regime_for_plot(
        baseline_long=baseline_es,
        regime_long=regime_es,
        probability_level=probability_level,
        value_column="conditional_es",
    )

    print("Plotting baseline vs regime VaR...")
    plot_baseline_vs_regime_risk(
        merged=var_merged,
        selected_assets=selected_assets,
        value_column="conditional_var",
        risk_name="VaR",
        probability_level=probability_level,
        output_path="outputs/figures/baseline_vs_regime_var_995_selected.png",
    )

    print("Plotting baseline vs regime ES...")
    plot_baseline_vs_regime_risk(
        merged=es_merged,
        selected_assets=selected_assets,
        value_column="conditional_es",
        risk_name="ES",
        probability_level=probability_level,
        output_path="outputs/figures/baseline_vs_regime_es_995_selected.png",
    )

    print("Plotting regime minus baseline VaR difference...")
    plot_regime_minus_baseline_difference(
        merged=var_merged,
        selected_assets=selected_assets,
        value_column="conditional_var",
        risk_name="VaR",
        probability_level=probability_level,
        output_path="outputs/figures/regime_minus_baseline_var_995_selected.png",
    )

    print("Plotting regime minus baseline ES difference...")
    plot_regime_minus_baseline_difference(
        merged=es_merged,
        selected_assets=selected_assets,
        value_column="conditional_es",
        risk_name="ES",
        probability_level=probability_level,
        output_path="outputs/figures/regime_minus_baseline_es_995_selected.png",
    )

    print("\nBaseline vs regime plots saved successfully.")
    print("outputs/figures/baseline_vs_regime_var_995_selected.png")
    print("outputs/figures/baseline_vs_regime_es_995_selected.png")
    print("outputs/figures/regime_minus_baseline_var_995_selected.png")
    print("outputs/figures/regime_minus_baseline_es_995_selected.png")


if __name__ == "__main__":
    main()