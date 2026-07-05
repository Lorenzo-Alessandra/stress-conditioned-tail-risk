import pandas as pd

from src.models.risk_forecasts import (
    build_residual_risk_lookup,
    compute_conditional_risk_forecasts,
    extract_garch_mean_raw_units,
    stack_forecast_panels,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    garch_params_path = paths_config["processed_files"]["garch_params"]
    conditional_volatility_path = paths_config["processed_files"][
        "garch_conditional_volatility"
    ]
    evt_var_es_path = paths_config["processed_files"]["evt_var_es_baseline"]

    print("Reading GARCH parameters...")
    garch_params = pd.read_parquet(garch_params_path)

    print("Reading GARCH conditional volatility...")
    conditional_volatility = pd.read_parquet(conditional_volatility_path)

    print("Reading baseline EVT residual VaR/ES estimates...")
    evt_var_es = pd.read_parquet(evt_var_es_path)

    print("Extracting GARCH mean in raw return units...")
    mu_raw = extract_garch_mean_raw_units(
        garch_params=garch_params,
        scaled_by_100=True,
    )

    print("Building residual VaR lookup...")
    residual_var_by_level = build_residual_risk_lookup(
        evt_var_es=evt_var_es,
        risk_measure="residual_var",
    )

    print("Building residual ES lookup...")
    residual_es_by_level = build_residual_risk_lookup(
        evt_var_es=evt_var_es,
        risk_measure="residual_es",
    )

    print("Computing conditional VaR forecasts...")
    conditional_var_by_level = compute_conditional_risk_forecasts(
        conditional_volatility=conditional_volatility,
        mu_raw=mu_raw,
        residual_risk_by_level=residual_var_by_level,
    )

    print("Computing conditional ES forecasts...")
    conditional_es_by_level = compute_conditional_risk_forecasts(
        conditional_volatility=conditional_volatility,
        mu_raw=mu_raw,
        residual_risk_by_level=residual_es_by_level,
    )

    conditional_var_long = stack_forecast_panels(
        conditional_var_by_level,
        value_name="conditional_var",
    )

    conditional_es_long = stack_forecast_panels(
        conditional_es_by_level,
        value_name="conditional_es",
    )

    conditional_var_long.to_parquet(
        paths_config["processed_files"]["conditional_var_baseline"]
    )

    conditional_es_long.to_parquet(
        paths_config["processed_files"]["conditional_es_baseline"]
    )

    var_csv_path = "outputs/tables/conditional_var_baseline.csv"
    es_csv_path = "outputs/tables/conditional_es_baseline.csv"

    save_table_csv(conditional_var_long, var_csv_path)
    save_table_csv(conditional_es_long, es_csv_path)

    print("\nConditional VaR preview:")
    print(conditional_var_long.head(20).to_string(index=False))

    print("\nConditional ES preview:")
    print(conditional_es_long.head(20).to_string(index=False))

    print("\nSummary of conditional VaR:")
    print(
        conditional_var_long.groupby(["asset", "probability_level"])[
            "conditional_var"
        ]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nSummary of conditional ES:")
    print(
        conditional_es_long.groupby(["asset", "probability_level"])[
            "conditional_es"
        ]
        .describe()
        .round(6)
        .to_string()
    )

    print("\nConditional risk forecasts saved successfully.")
    print(f"Conditional VaR parquet: {paths_config['processed_files']['conditional_var_baseline']}")
    print(f"Conditional ES parquet: {paths_config['processed_files']['conditional_es_baseline']}")
    print(f"Conditional VaR CSV: {var_csv_path}")
    print(f"Conditional ES CSV: {es_csv_path}")


if __name__ == "__main__":
    main()