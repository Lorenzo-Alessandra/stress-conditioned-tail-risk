import pandas as pd

from src.models.evt import compute_evt_var_es_table, fit_gpd_panel
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    residual_losses_path = paths_config["processed_files"][
        "standardized_residual_losses"
    ]

    threshold_quantile = model_config["evt"]["main_threshold_quantile"]
    probability_levels = model_config["evt"]["tail_probability_levels"]

    print("Reading standardized residual losses...")
    residual_losses = pd.read_parquet(residual_losses_path)

    print(f"Fitting baseline POT-GPD at threshold quantile {threshold_quantile}...")
    gpd_params = fit_gpd_panel(
        residual_losses=residual_losses,
        threshold_quantile=threshold_quantile,
    )

    print("Computing EVT residual VaR and ES...")
    var_es_table = compute_evt_var_es_table(
        gpd_params=gpd_params,
        probability_levels=probability_levels,
    )

    gpd_params_output = gpd_params.copy()
    var_es_output = var_es_table.copy()

    numeric_param_columns = [
        "threshold_value",
        "shape_xi",
        "scale_beta",
        "loglikelihood",
    ]

    for column in numeric_param_columns:
        gpd_params_output[column] = gpd_params_output[column].round(6)

    numeric_var_es_columns = [
        "residual_var",
        "residual_es",
        "shape_xi",
        "scale_beta",
        "threshold_value",
    ]

    for column in numeric_var_es_columns:
        var_es_output[column] = var_es_output[column].round(6)

    gpd_params.to_parquet(paths_config["processed_files"]["evt_gpd_params_baseline"])
    var_es_table.to_parquet(paths_config["processed_files"]["evt_var_es_baseline"])

    params_csv_path = "outputs/tables/evt_gpd_params_baseline.csv"
    params_latex_path = "outputs/tables/evt_gpd_params_baseline.tex"

    var_es_csv_path = "outputs/tables/evt_var_es_baseline.csv"
    var_es_latex_path = "outputs/tables/evt_var_es_baseline.tex"

    save_table_csv(gpd_params_output.reset_index(), params_csv_path)
    save_table_latex(
        gpd_params_output,
        params_latex_path,
        caption="Baseline POT-GPD parameter estimates for standardized residual losses.",
        label="tab:evt_gpd_params_baseline",
    )

    save_table_csv(var_es_output, var_es_csv_path)
    save_table_latex(
        var_es_output,
        var_es_latex_path,
        caption="Baseline EVT residual VaR and ES estimates.",
        label="tab:evt_var_es_baseline",
    )

    print("\nBaseline GPD parameter estimates:")
    print(gpd_params_output.to_string())

    print("\nBaseline EVT residual VaR and ES estimates:")
    print(var_es_output.to_string(index=False))

    print("\nBaseline EVT outputs saved successfully.")
    print(f"GPD params parquet: {paths_config['processed_files']['evt_gpd_params_baseline']}")
    print(f"VaR/ES parquet: {paths_config['processed_files']['evt_var_es_baseline']}")
    print(f"GPD params CSV: {params_csv_path}")
    print(f"VaR/ES CSV: {var_es_csv_path}")


if __name__ == "__main__":
    main()