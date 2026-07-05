import pandas as pd

from src.models.covar import (
    compute_empirical_covar,
    rank_covar_systemic_contribution,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    losses_path = paths_config["processed_files"]["losses"]
    system_loss_path = paths_config["processed_files"]["system_loss"]

    quantile_levels = model_config["tail_dependence"]["q_levels"]
    main_quantile_level = 0.95

    print("Reading bank losses...")
    losses = pd.read_parquet(losses_path)

    print("Reading system loss...")
    system_loss = pd.read_parquet(system_loss_path)

    print("Computing empirical CoVaR and Delta CoVaR...")
    covar = compute_empirical_covar(
        losses=losses,
        system_loss=system_loss,
        quantile_levels=quantile_levels,
        normal_quantile_level=0.50,
    )

    print(f"Ranking Delta CoVaR at q={main_quantile_level}...")
    ranked_q95 = rank_covar_systemic_contribution(
        covar_table=covar,
        quantile_level=main_quantile_level,
    )

    covar.to_parquet(paths_config["processed_files"]["covar_empirical"])

    covar_output = covar.copy()
    ranked_output = ranked_q95.copy()

    for df in [covar_output, ranked_output]:
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(6)

    covar_csv_path = "outputs/tables/covar_empirical.csv"
    covar_latex_path = "outputs/tables/covar_empirical.tex"

    ranked_csv_path = "outputs/tables/covar_empirical_ranked_q95.csv"
    ranked_latex_path = "outputs/tables/covar_empirical_ranked_q95.tex"

    save_table_csv(covar_output, covar_csv_path)
    save_table_latex(
        covar_output,
        covar_latex_path,
        caption="Empirical CoVaR and Delta CoVaR estimates.",
        label="tab:covar_empirical",
    )

    save_table_csv(ranked_output, ranked_csv_path)
    save_table_latex(
        ranked_output,
        ranked_latex_path,
        caption="Empirical Delta CoVaR ranking at the 95 percent loss threshold.",
        label="tab:covar_empirical_ranked_q95",
    )

    print("\nEmpirical CoVaR table:")
    print(covar_output.to_string(index=False))

    print(f"\nDelta CoVaR ranking at q={main_quantile_level}:")
    print(ranked_output.to_string(index=False))

    print("\nEmpirical CoVaR outputs saved successfully.")
    print(covar_csv_path)
    print(ranked_csv_path)


if __name__ == "__main__":
    main()