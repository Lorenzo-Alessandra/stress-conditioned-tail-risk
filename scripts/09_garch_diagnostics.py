import pandas as pd

from src.utils.config import load_yaml
from src.visualization.plots import (
    plot_garch_conditional_volatility,
    plot_rolling_squared_standardized_residuals,
    plot_standardized_residual_loss_histograms,
    plot_standardized_residuals,
)
from src.visualization.tables import (
    format_descriptive_stats_table,
    make_descriptive_stats_table,
    save_table_csv,
    save_table_latex,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    conditional_volatility_path = paths_config["processed_files"][
        "garch_conditional_volatility"
    ]
    standardized_residuals_path = paths_config["processed_files"][
        "garch_standardized_residuals"
    ]
    standardized_residual_losses_path = paths_config["processed_files"][
        "standardized_residual_losses"
    ]

    print("Reading GARCH outputs...")
    conditional_volatility = pd.read_parquet(conditional_volatility_path)
    standardized_residuals = pd.read_parquet(standardized_residuals_path)
    standardized_residual_losses = pd.read_parquet(standardized_residual_losses_path)

    print(f"Diagnostic universe contains {standardized_residuals.shape[1]} banks:")
    print(standardized_residuals.columns.tolist())

    print("Creating conditional volatility plot...")
    plot_garch_conditional_volatility(
        conditional_volatility=conditional_volatility,
        output_path="outputs/figures/garch_conditional_volatility.png",
    )

    print("Creating standardized residuals plot...")
    plot_standardized_residuals(
        standardized_residuals=standardized_residuals,
        output_path="outputs/figures/garch_standardized_residuals.png",
    )

    print("Creating standardized residual loss histograms...")
    plot_standardized_residual_loss_histograms(
        residual_losses=standardized_residual_losses,
        output_path="outputs/figures/standardized_residual_losses_histograms.png",
    )

    print("Creating rolling squared standardized residual plot...")
    plot_rolling_squared_standardized_residuals(
        standardized_residuals=standardized_residuals,
        output_path="outputs/figures/rolling_squared_standardized_residuals_60d.png",
        window=60,
    )

    print("Creating standardized residual descriptive table...")
    residual_table = make_descriptive_stats_table(standardized_residuals)
    formatted_residual_table = format_descriptive_stats_table(residual_table)

    csv_path = "outputs/tables/standardized_residuals_descriptive_stats.csv"
    latex_path = "outputs/tables/standardized_residuals_descriptive_stats.tex"

    save_table_csv(formatted_residual_table, csv_path)

    save_table_latex(
        formatted_residual_table,
        latex_path,
        caption="Descriptive statistics of GJR-GARCH standardized residuals.",
        label="tab:standardized_residuals_descriptive_stats",
    )

    print("\nStandardized residual descriptive statistics:")
    print(formatted_residual_table)

    print("\nGARCH diagnostic outputs saved successfully.")
    print("outputs/figures/garch_conditional_volatility.png")
    print("outputs/figures/garch_standardized_residuals.png")
    print("outputs/figures/standardized_residual_losses_histograms.png")
    print("outputs/figures/rolling_squared_standardized_residuals_60d.png")
    print(csv_path)
    print(latex_path)


if __name__ == "__main__":
    main()