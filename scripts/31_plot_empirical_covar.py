import pandas as pd

from src.utils.config import load_yaml
from src.visualization.covar_plots import (
    plot_delta_covar_ranking,
    plot_delta_covar_threshold_robustness,
    plot_distress_and_normal_covar,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    covar_path = paths_config["processed_files"]["covar_empirical"]

    main_quantile_level = 0.95

    print("Reading empirical CoVaR table...")
    covar = pd.read_parquet(covar_path)

    print("Plotting Delta CoVaR ranking...")
    plot_delta_covar_ranking(
        covar_table=covar,
        quantile_level=main_quantile_level,
        output_path="outputs/figures/delta_covar_ranking_q95.png",
    )

    print("Plotting Delta CoVaR threshold robustness...")
    plot_delta_covar_threshold_robustness(
        covar_table=covar,
        output_path="outputs/figures/delta_covar_threshold_robustness.png",
    )

    print("Plotting distress vs normal CoVaR...")
    plot_distress_and_normal_covar(
        covar_table=covar,
        quantile_level=main_quantile_level,
        output_path="outputs/figures/distress_vs_normal_covar_q95.png",
    )

    print("\nEmpirical CoVaR plots saved successfully.")
    print("outputs/figures/delta_covar_ranking_q95.png")
    print("outputs/figures/delta_covar_threshold_robustness.png")
    print("outputs/figures/distress_vs_normal_covar_q95.png")


if __name__ == "__main__":
    main()