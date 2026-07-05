import pandas as pd

from src.models.evt import compute_evt_var_es_table, fit_gpd_panel
from src.utils.config import load_yaml
from src.visualization.plots import (
    plot_evt_risk_measure_threshold_stability,
    plot_evt_shape_threshold_stability,
)
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    residual_losses_path = paths_config["processed_files"][
        "standardized_residual_losses"
    ]

    threshold_quantiles = model_config["evt"]["threshold_quantiles"]
    probability_levels = model_config["evt"]["tail_probability_levels"]

    print("Reading standardized residual losses...")
    residual_losses = pd.read_parquet(residual_losses_path)

    all_param_tables = []
    all_var_es_tables = []

    for threshold_quantile in threshold_quantiles:
        print(f"\nFitting POT-GPD at threshold quantile {threshold_quantile}...")

        gpd_params = fit_gpd_panel(
            residual_losses=residual_losses,
            threshold_quantile=threshold_quantile,
        )

        var_es_table = compute_evt_var_es_table(
            gpd_params=gpd_params,
            probability_levels=probability_levels,
        )

        all_param_tables.append(gpd_params.reset_index())
        all_var_es_tables.append(var_es_table)

    params_robustness = pd.concat(all_param_tables, axis=0, ignore_index=True)
    var_es_robustness = pd.concat(all_var_es_tables, axis=0, ignore_index=True)

    params_robustness = params_robustness.sort_values(
        ["asset", "threshold_quantile"]
    ).reset_index(drop=True)

    var_es_robustness = var_es_robustness.sort_values(
        ["asset", "probability_level", "threshold_quantile"]
    ).reset_index(drop=True)

    print("\nSaving robustness outputs...")

    params_robustness.to_parquet(
        paths_config["processed_files"]["evt_gpd_params_threshold_robustness"]
    )

    var_es_robustness.to_parquet(
        paths_config["processed_files"]["evt_var_es_threshold_robustness"]
    )

    params_output = params_robustness.copy()
    var_es_output = var_es_robustness.copy()

    param_round_cols = [
        "threshold_value",
        "shape_xi",
        "scale_beta",
        "loglikelihood",
    ]

    for column in param_round_cols:
        params_output[column] = params_output[column].round(6)

    var_es_round_cols = [
        "residual_var",
        "residual_es",
        "shape_xi",
        "scale_beta",
        "threshold_value",
    ]

    for column in var_es_round_cols:
        var_es_output[column] = var_es_output[column].round(6)

    params_csv_path = "outputs/tables/evt_gpd_params_threshold_robustness.csv"
    params_latex_path = "outputs/tables/evt_gpd_params_threshold_robustness.tex"

    var_es_csv_path = "outputs/tables/evt_var_es_threshold_robustness.csv"
    var_es_latex_path = "outputs/tables/evt_var_es_threshold_robustness.tex"

    save_table_csv(params_output, params_csv_path)
    save_table_latex(
        params_output,
        params_latex_path,
        caption="POT-GPD parameter estimates across threshold choices.",
        label="tab:evt_gpd_params_threshold_robustness",
    )

    save_table_csv(var_es_output, var_es_csv_path)
    save_table_latex(
        var_es_output,
        var_es_latex_path,
        caption="EVT residual VaR and ES estimates across threshold choices.",
        label="tab:evt_var_es_threshold_robustness",
    )

    params_for_plot = params_robustness.set_index("asset")

    print("Creating threshold stability plots...")
    plot_evt_shape_threshold_stability(
        gpd_params=params_for_plot,
        output_path="outputs/figures/evt_shape_threshold_stability.png",
    )

    plot_evt_risk_measure_threshold_stability(
        var_es_table=var_es_robustness,
        probability_level=0.995,
        risk_measure="residual_var",
        output_path="outputs/figures/evt_var_995_threshold_stability.png",
    )

    plot_evt_risk_measure_threshold_stability(
        var_es_table=var_es_robustness,
        probability_level=0.995,
        risk_measure="residual_es",
        output_path="outputs/figures/evt_es_995_threshold_stability.png",
    )

    print("\nThreshold robustness parameter estimates:")
    print(params_output.to_string(index=False))

    print("\nThreshold robustness VaR/ES estimates:")
    print(var_es_output.to_string(index=False))

    print("\nThreshold robustness outputs saved successfully.")
    print(params_csv_path)
    print(var_es_csv_path)
    print("outputs/figures/evt_shape_threshold_stability.png")
    print("outputs/figures/evt_var_995_threshold_stability.png")
    print("outputs/figures/evt_es_995_threshold_stability.png")


if __name__ == "__main__":
    main()