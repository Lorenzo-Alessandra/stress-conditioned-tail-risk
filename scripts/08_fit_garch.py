import pandas as pd

from src.models.garch import (
    compute_standardized_residual_losses,
    fit_garch_panel,
)
from src.utils.config import load_yaml


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    returns_path = paths_config["processed_files"]["returns"]

    print("Reading bank returns...")
    returns = pd.read_parquet(returns_path)

    print(f"Fitting GARCH models for {returns.shape[1]} banks:")
    print(returns.columns.tolist())

    params, conditional_volatility, standardized_residuals = fit_garch_panel(
        returns=returns,
        scale_returns=True,
    )

    standardized_residual_losses = compute_standardized_residual_losses(
        standardized_residuals
    )

    print("\nGARCH parameter estimates:")
    print(params)

    print("\nMissing standardized residuals:")
    print(standardized_residuals.isna().sum())

    params.to_parquet(paths_config["processed_files"]["garch_params"])
    conditional_volatility.to_parquet(
        paths_config["processed_files"]["garch_conditional_volatility"]
    )
    standardized_residuals.to_parquet(
        paths_config["processed_files"]["garch_standardized_residuals"]
    )
    standardized_residual_losses.to_parquet(
        paths_config["processed_files"]["standardized_residual_losses"]
    )

    print("\nGARCH outputs saved successfully.")
    print(f"Parameters: {paths_config['processed_files']['garch_params']}")
    print(
        "Conditional volatility: "
        f"{paths_config['processed_files']['garch_conditional_volatility']}"
    )
    print(
        "Standardized residuals: "
        f"{paths_config['processed_files']['garch_standardized_residuals']}"
    )
    print(
        "Standardized residual losses: "
        f"{paths_config['processed_files']['standardized_residual_losses']}"
    )


if __name__ == "__main__":
    main()