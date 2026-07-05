import pandas as pd

from src.models.es_scoring import (
    compute_simple_es_score,
    merge_losses_var_es,
    summarize_es_scores,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    losses_path = paths_config["processed_files"]["losses"]
    conditional_var_path = paths_config["processed_files"]["conditional_var_baseline"]
    conditional_es_path = paths_config["processed_files"]["conditional_es_baseline"]

    print("Reading actual losses...")
    losses = pd.read_parquet(losses_path)

    print("Reading conditional VaR forecasts...")
    conditional_var = pd.read_parquet(conditional_var_path)

    print("Reading conditional ES forecasts...")
    conditional_es = pd.read_parquet(conditional_es_path)

    print("Merging losses, VaR, and ES...")
    merged = merge_losses_var_es(
        losses=losses,
        conditional_var_long=conditional_var,
        conditional_es_long=conditional_es,
    )

    print("Computing simple ES scores...")
    scored = compute_simple_es_score(merged)

    print("Summarizing ES scores...")
    summary = summarize_es_scores(scored)

    scored.to_parquet(paths_config["processed_files"]["es_scores_daily_baseline"])
    summary.to_parquet(paths_config["processed_files"]["es_scoring_baseline"])

    summary_output = summary.copy()

    round_columns = [
        "mean_score",
        "median_score",
        "std_score",
        "min_score",
        "max_score",
        "mean_loss",
        "mean_var",
        "mean_es",
        "violation_rate",
    ]

    for column in round_columns:
        summary_output[column] = summary_output[column].round(6)

    csv_path = "outputs/tables/es_scoring_baseline.csv"
    latex_path = "outputs/tables/es_scoring_baseline.tex"

    save_table_csv(summary_output, csv_path)

    save_table_latex(
        summary_output,
        latex_path,
        caption="Simple ES scoring results for baseline GARCH-EVT forecasts.",
        label="tab:es_scoring_baseline",
    )

    print("\nES scoring summary:")
    print(summary_output.to_string(index=False))

    print("\nES scoring outputs saved successfully.")
    print(f"Daily ES scores: {paths_config['processed_files']['es_scores_daily_baseline']}")
    print(f"ES summary: {paths_config['processed_files']['es_scoring_baseline']}")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()