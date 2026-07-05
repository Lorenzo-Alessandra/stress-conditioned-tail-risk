import pandas as pd

from src.utils.config import load_yaml
from src.visualization.evt_plots import (
    plot_gpd_probability_plots,
    plot_gpd_qq_plots,
    plot_mean_excess_plots,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    residual_losses_path = paths_config["processed_files"][
        "standardized_residual_losses"
    ]
    gpd_params_path = paths_config["processed_files"]["evt_gpd_params_baseline"]

    print("Reading standardized residual losses...")
    residual_losses = pd.read_parquet(residual_losses_path)

    print("Reading baseline GPD parameters...")
    gpd_params = pd.read_parquet(gpd_params_path)

    print("Creating mean excess plots...")
    plot_mean_excess_plots(
        residual_losses=residual_losses,
        output_path="outputs/figures/evt_mean_excess_plots.png",
    )

    print("Creating GPD QQ plots...")
    plot_gpd_qq_plots(
        residual_losses=residual_losses,
        gpd_params=gpd_params,
        output_path="outputs/figures/evt_gpd_qq_plots_baseline.png",
    )

    print("Creating GPD probability plots...")
    plot_gpd_probability_plots(
        residual_losses=residual_losses,
        gpd_params=gpd_params,
        output_path="outputs/figures/evt_gpd_probability_plots_baseline.png",
    )

    print("\nEVT diagnostic plots saved successfully.")
    print("outputs/figures/evt_mean_excess_plots.png")
    print("outputs/figures/evt_gpd_qq_plots_baseline.png")
    print("outputs/figures/evt_gpd_probability_plots_baseline.png")


if __name__ == "__main__":
    main()