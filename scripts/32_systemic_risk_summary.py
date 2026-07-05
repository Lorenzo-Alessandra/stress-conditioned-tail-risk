import pandas as pd

from src.models.systemic_summary import build_systemic_risk_summary
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    es_scoring_path = paths_config["processed_files"]["es_scoring_regime"]

    connectedness_csv_path = "outputs/tables/tail_connectedness_summary.csv"
    covar_path = paths_config["processed_files"]["covar_empirical"]

    network_summary_path = paths_config["processed_files"][
        "tail_dependence_network_summary_q95_threshold_050"
    ]

    print("Reading regime-specific ES scoring summary...")
    es_scoring_regime = pd.read_parquet(es_scoring_path)

    print("Reading tail connectedness summary...")
    tail_connectedness = pd.read_csv(connectedness_csv_path)

    print("Reading empirical CoVaR table...")
    covar_empirical = pd.read_parquet(covar_path)

    print("Reading 0.50-threshold network summary...")
    network_summary_050 = pd.read_parquet(network_summary_path)

    print("Building integrated systemic-risk summary...")
    summary = build_systemic_risk_summary(
        es_scoring_regime=es_scoring_regime,
        tail_connectedness=tail_connectedness,
        covar_empirical=covar_empirical,
        network_summary_050=network_summary_050,
    )

    summary.to_parquet(paths_config["processed_files"]["systemic_risk_summary"])

    output = summary.copy()

    round_columns = [
        "mean_es_995",
        "average_tail_connectedness_q95",
        "delta_covar_q95",
        "network_weighted_degree_050",
        "rank_mean_es_995",
        "rank_tail_connectedness_q95",
        "rank_delta_covar_q95",
        "rank_network_weighted_degree_050",
        "average_rank",
    ]

    for column in round_columns:
        output[column] = output[column].round(6)

    csv_path = "outputs/tables/systemic_risk_summary.csv"
    latex_path = "outputs/tables/systemic_risk_summary.tex"

    save_table_csv(output, csv_path)

    save_table_latex(
        output,
        latex_path,
        caption="Integrated systemic-risk summary combining marginal tail risk, tail connectedness, Delta CoVaR, and network centrality.",
        label="tab:systemic_risk_summary",
    )

    print("\nIntegrated systemic-risk summary:")
    print(output.to_string(index=False))

    print("\nSystemic-risk summary saved successfully.")
    print(f"CSV: {csv_path}")
    print(f"LaTeX: {latex_path}")
    print(f"Parquet: {paths_config['processed_files']['systemic_risk_summary']}")


if __name__ == "__main__":
    main()