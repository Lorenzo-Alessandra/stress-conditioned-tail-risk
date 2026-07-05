import pandas as pd

from src.models.risk_forecasts import extract_garch_mean_raw_units
from src.models.risk_forecasts_regime import (
    build_regime_residual_risk_lookup,
    compute_regime_conditional_risk_forecasts,
    stack_regime_forecast_panels,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    garch_params_path = paths_config["processed_files"]["garch_params"]
    conditional_volatility_path = paths_config["processed_files"][
        "garch_conditional_volatility"
    ]
    regime_var_es_path = paths_config["processed_files"]["evt_var_es_regime"]
    stress_regime_path = paths_config["processed_files"]["stress_regime"]

    print("Reading GARCH parameters...")
    garch_params = pd.read_parquet(garch_params_path)

    print("Reading GARCH conditional volatility...")
    conditional_volatility = pd.read_parquet(conditional_volatility_path)

    print("Reading regime-specific EVT residual VaR/ES...")
    regime_var_es = pd.read_parquet(regime_var_es_path)

    print("Reading stress regime...")
    stress_regime = pd.read_parquet(stress_regime_path)

    print("Extracting GARCH mean in raw return units...")
    mu_raw = extract_garch_mean_raw_units(
        garch_params=garch_params,
        scaled_by_100=True,
    )

    print("Building regime-specific residual VaR lookup...")
    residual_var_lookup = build_regime_residual_risk_lookup(
        regime_var_es=regime_var_es,
        risk_measure="residual_var",
    )

    print("Building regime-specific residual ES lookup...")
    residual_es_lookup = build_regime_residual_risk_lookup(
        regime_var_es=regime_var_es,
        risk_measure="residual_es",
    )

    print("Computing regime-specific conditional VaR...")
    conditional_var_by_level = compute_regime_conditional_risk_forecasts(
        conditional_volatility=conditional_volatility,
        mu_raw=mu_raw,
        stress_regime=stress_regime,
        residual_risk_lookup=residual_var_lookup,
    )

    print("Computing regime-specific conditional ES...")
    conditional_es_by_level = compute_regime_conditional_risk_forecasts(
        conditional_volatility=conditional_volatility,
        mu_raw=mu_raw,
        stress_regime=stress_regime,
        residual_risk_lookup=residual_es_lookup,
    )

    conditional_var_long = stack_regime_forecast_panels(
        forecast_by_level=conditional_var_by_level,
        stress_regime=stress_regime,
        value_name="conditional_var",
    )

    conditional_es_long = stack_regime_forecast_panels(
        forecast_by_level=conditional_es_by_level,
        stress_regime=stress_regime,
        value_name="conditional_es",
    )

    conditional_var_long.to_parquet(
        paths_config["processed_files"]["conditional_var_regime"]
    )

    conditional_es_long.to_parquet(
        paths_config["processed_files"]["conditional_es_regime"]
    )

    var_csv_path = "outputs/tables/conditional_var_regime.csv"
    es_csv_path = "outputs/tables/conditional_es_regime.csv"

    save_table_csv(conditional_var_long, var_csv_path)
    save_table_csv(conditional_es_long, es_csv_path)

    print("\nRegime-specific conditional VaR preview:")
    print(conditional_var_long.head(20).to_string(index=False))

    print("\nRegime-specific conditional ES preview:")
    print(conditional_es_long.head(20).to_string(index=False))

    print("\nRegime-specific conditional VaR summary:")
    print(
        conditional_var_long.groupby(["asset", "probability_level", "stress_regime"])[
            "conditional_var"
        ]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nRegime-specific conditional ES summary:")
    print(
        conditional_es_long.groupby(["asset", "probability_level", "stress_regime"])[
            "conditional_es"
        ]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nRegime-specific conditional risk forecasts saved successfully.")
    print(f"Conditional VaR parquet: {paths_config['processed_files']['conditional_var_regime']}")
    print(f"Conditional ES parquet: {paths_config['processed_files']['conditional_es_regime']}")
    print(f"Conditional VaR CSV: {var_csv_path}")
    print(f"Conditional ES CSV: {es_csv_path}")


if __name__ == "__main__":
    main()