import pandas as pd

from src.models.evt_regime import (
    compare_calm_stress_parameters,
    compute_regime_var_es_table,
    fit_regime_specific_gpd,
)
from src.models.regimes import (
    align_residual_losses_with_regime,
    build_quantile_stress_regime,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    residual_losses_path = paths_config["processed_files"][
        "standardized_residual_losses"
    ]
    stress_variables_path = paths_config["processed_files"]["stress_variables"]

    stress_variable = model_config["stress_regime"]["primary_variable"]
    stress_quantile = model_config["stress_regime"]["quantile_threshold"]

    threshold_quantile = model_config["evt"]["main_threshold_quantile"]
    probability_levels = model_config["evt"]["tail_probability_levels"]

    print("Reading standardized residual losses...")
    residual_losses = pd.read_parquet(residual_losses_path)

    print("Reading stress variables...")
    stress_variables = pd.read_parquet(stress_variables_path)

    print(
        f"Building stress regime using {stress_variable} "
        f"above quantile {stress_quantile}..."
    )

    stress_regime = build_quantile_stress_regime(
        stress_variables=stress_variables,
        variable=stress_variable,
        quantile=stress_quantile,
    )

    print("\nStress-regime summary:")
    print(stress_regime["stress_regime"].value_counts().sort_index())
    print(f"Stress threshold: {stress_regime['stress_threshold'].iloc[0]:.4f}")

    print("Aligning residual losses with stress regime...")
    residual_regime_long = align_residual_losses_with_regime(
        residual_losses=residual_losses,
        regime=stress_regime,
    )

    print("Fitting regime-specific POT-GPD models...")
    regime_params = fit_regime_specific_gpd(
        residual_regime_long=residual_regime_long,
        threshold_quantile=threshold_quantile,
    )

    print("Computing regime-specific residual VaR/ES...")
    regime_var_es = compute_regime_var_es_table(
        regime_gpd_params=regime_params,
        probability_levels=probability_levels,
    )

    print("Comparing calm and stress parameters...")
    regime_comparison = compare_calm_stress_parameters(regime_params)

    stress_regime.to_parquet(paths_config["processed_files"]["stress_regime"])
    regime_params.to_parquet(paths_config["processed_files"]["evt_gpd_params_regime"])
    regime_var_es.to_parquet(paths_config["processed_files"]["evt_var_es_regime"])

    params_csv_path = "outputs/tables/evt_gpd_params_regime.csv"
    params_latex_path = "outputs/tables/evt_gpd_params_regime.tex"

    var_es_csv_path = "outputs/tables/evt_var_es_regime.csv"
    var_es_latex_path = "outputs/tables/evt_var_es_regime.tex"

    comparison_csv_path = "outputs/tables/evt_regime_parameter_comparison.csv"
    comparison_latex_path = "outputs/tables/evt_regime_parameter_comparison.tex"

    params_output = regime_params.copy()
    var_es_output = regime_var_es.copy()
    comparison_output = regime_comparison.copy()

    for df in [params_output, var_es_output, comparison_output]:
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(6)

    save_table_csv(params_output, params_csv_path)
    save_table_latex(
        params_output,
        params_latex_path,
        caption="Regime-specific POT-GPD parameter estimates.",
        label="tab:evt_gpd_params_regime",
    )

    save_table_csv(var_es_output, var_es_csv_path)
    save_table_latex(
        var_es_output,
        var_es_latex_path,
        caption="Regime-specific EVT residual VaR and ES estimates.",
        label="tab:evt_var_es_regime",
    )

    save_table_csv(comparison_output, comparison_csv_path)
    save_table_latex(
        comparison_output,
        comparison_latex_path,
        caption="Comparison of calm and stress regime EVT parameters.",
        label="tab:evt_regime_parameter_comparison",
    )

    print("\nRegime-specific GPD parameters:")
    print(params_output.to_string(index=False))

    print("\nRegime-specific VaR/ES:")
    print(var_es_output.to_string(index=False))

    print("\nCalm vs stress parameter comparison:")
    print(comparison_output.to_string(index=False))

    print("\nRegime-specific EVT outputs saved successfully.")
    print(params_csv_path)
    print(var_es_csv_path)
    print(comparison_csv_path)


if __name__ == "__main__":
    main()