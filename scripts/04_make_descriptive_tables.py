import pandas as pd

from src.utils.config import load_yaml
from src.visualization.tables import (
    format_descriptive_stats_table,
    make_descriptive_stats_table,
    save_table_csv,
    save_table_latex,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    returns_path = paths_config["processed_files"]["returns"]

    print("Reading bank returns...")
    returns = pd.read_parquet(returns_path)

    print("Creating descriptive statistics table...")
    table = make_descriptive_stats_table(returns)
    formatted_table = format_descriptive_stats_table(table)

    csv_path = "outputs/tables/bank_returns_descriptive_stats.csv"
    latex_path = "outputs/tables/bank_returns_descriptive_stats.tex"

    save_table_csv(formatted_table, csv_path)

    save_table_latex(
        formatted_table,
        latex_path,
        caption="Descriptive statistics of daily bank log returns.",
        label="tab:bank_return_descriptive_stats",
    )

    print("\nDescriptive statistics table:")
    print(formatted_table)

    print("\nSaved outputs:")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()