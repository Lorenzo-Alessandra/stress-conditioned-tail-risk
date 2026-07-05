from matplotlib.path import Path
import pandas as pd

from src.models.evt_diagnostics import (
    format_evt_threshold_diagnostics_table,
    make_evt_threshold_diagnostics_table,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    residual_losses_path = paths_config["processed_files"][
        "standardized_residual_losses"
    ]

    threshold_quantiles = model_config["evt"]["threshold_quantiles"]
    run_length = model_config["evt"]["clustering_run_length"]

    print("Reading standardized residual losses...")
    residual_losses = pd.read_parquet(residual_losses_path)

    print(f"Residual-loss panel shape: {residual_losses.shape}")
    print("Assets:")
    print(residual_losses.columns.tolist())

    print("Creating EVT threshold and clustering diagnostics...")
    diagnostics = make_evt_threshold_diagnostics_table(
        residual_losses=residual_losses,
        threshold_quantiles=threshold_quantiles,
        run_length=run_length,
    )

    formatted_diagnostics = format_evt_threshold_diagnostics_table(diagnostics)

    csv_path = "outputs/tables/evt_threshold_diagnostics.csv"
    latex_path = "outputs/tables/evt_threshold_diagnostics.tex"

    def save_table_csv(table: pd.DataFrame, path: str | Path) -> None:
        """
        Save a table as CSV.

        Parameters
        ----------
        table:
            Table to save.
        path:
            Output CSV path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    save_table_latex(
        formatted_diagnostics,
        latex_path,
        caption=(
            "EVT threshold and exceedance clustering diagnostics for "
            "standardized residual losses."
        ),
        label="tab:evt_threshold_diagnostics",
    )

    print("\nEVT threshold diagnostics:")
    print(formatted_diagnostics.to_string(index=False))

    print("\nSaved outputs:")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()