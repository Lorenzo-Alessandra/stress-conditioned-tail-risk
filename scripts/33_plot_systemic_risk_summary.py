import pandas as pd

from src.utils.config import load_yaml
from src.visualization.systemic_summary_plots import (
    plot_average_systemic_rank,
    plot_component_rank_heatmap,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    summary_path = paths_config["processed_files"]["systemic_risk_summary"]

    print("Reading integrated systemic-risk summary...")
    summary = pd.read_parquet(summary_path)

    print("Plotting average systemic-risk rank...")
    plot_average_systemic_rank(
        summary=summary,
        output_path="outputs/figures/systemic_risk_average_rank.png",
    )

    print("Plotting component rank heatmap...")
    plot_component_rank_heatmap(
        summary=summary,
        output_path="outputs/figures/systemic_risk_component_rank_heatmap.png",
    )

    print("\nSystemic-risk summary plots saved successfully.")
    print("outputs/figures/systemic_risk_average_rank.png")
    print("outputs/figures/systemic_risk_component_rank_heatmap.png")


if __name__ == "__main__":
    main()