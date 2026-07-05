import pandas as pd

from src.utils.config import load_yaml
from src.visualization.plots import (
    plot_bank_loss_histograms,
    plot_bank_returns_timeseries,
    plot_rolling_volatility,
    plot_system_loss_timeseries,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    returns_path = paths_config["processed_files"]["returns"]
    losses_path = paths_config["processed_files"]["losses"]
    system_loss_path = paths_config["processed_files"]["system_loss"]

    print("Reading processed returns, losses, and system loss...")
    returns = pd.read_parquet(returns_path)
    losses = pd.read_parquet(losses_path)
    system_loss = pd.read_parquet(system_loss_path)

    print("Creating bank returns time series plot...")
    plot_bank_returns_timeseries(
        returns=returns,
        output_path="outputs/figures/bank_returns_timeseries.png",
    )

    print("Creating system loss time series plot...")
    plot_system_loss_timeseries(
        system_loss=system_loss,
        output_path="outputs/figures/system_loss_timeseries.png",
    )

    print("Creating bank loss histogram plot...")
    plot_bank_loss_histograms(
        losses=losses,
        output_path="outputs/figures/bank_loss_histograms.png",
    )

    print("Creating rolling volatility plot...")
    plot_rolling_volatility(
        returns=returns,
        output_path="outputs/figures/rolling_volatility_60d.png",
        window=60,
    )

    print("\nInitial plots saved successfully.")
    print("outputs/figures/bank_returns_timeseries.png")
    print("outputs/figures/system_loss_timeseries.png")
    print("outputs/figures/bank_loss_histograms.png")
    print("outputs/figures/rolling_volatility_60d.png")


if __name__ == "__main__":
    main()