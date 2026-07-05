import pandas as pd

from src.utils.config import load_yaml
from src.visualization.tables import (
    make_extreme_returns_audit_table,
    save_table_csv,
    save_table_latex,
)


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    returns_path = paths_config["processed_files"]["returns"]

    print("Reading bank returns...")
    returns = pd.read_parquet(returns_path)

    print("Creating extreme-return audit table...")
    audit_table = make_extreme_returns_audit_table(
        returns=returns,
        top_n=10,
    )

    audit_table_for_output = audit_table.copy()
    audit_table_for_output["date"] = audit_table_for_output["date"].dt.strftime("%Y-%m-%d")
    audit_table_for_output["return"] = audit_table_for_output["return"].round(6)
    audit_table_for_output["loss"] = audit_table_for_output["loss"].round(6)

    csv_path = "outputs/tables/extreme_returns_audit.csv"
    latex_path = "outputs/tables/extreme_returns_audit.tex"

    save_table_csv(audit_table_for_output, csv_path)

    save_table_latex(
        audit_table_for_output,
        latex_path,
        caption="Largest positive and negative daily log returns by bank.",
        label="tab:extreme_returns_audit",
    )

    print("\nExtreme-return audit table:")
    print(audit_table_for_output)

    print("\nSaved outputs:")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()