import pandas as pd

from src.models.copula import (
    compare_empirical_and_copula_tail_dependence,
    compute_pseudo_observations,
    compute_simulated_tail_dependence_tables,
    compute_student_t_asymptotic_tail_dependence,
    compute_tail_dependence_from_uniforms,
    fit_student_t_copula,
    simulate_student_t_copula,
)
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    losses_path = paths_config["processed_files"]["losses"]
    empirical_matrix_path = paths_config["processed_files"][
        "tail_dependence_matrix_q95"
    ]

    quantile_levels = model_config["tail_dependence"]["q_levels"]
    main_quantile_level = 0.95

    copula_config = model_config["copula"]

    print("Reading bank losses...")
    losses = pd.read_parquet(losses_path).dropna(how="any")

    print("Computing pseudo-observations...")
    pseudo = compute_pseudo_observations(losses)

    print("Fitting Student-t copula...")
    params, correlation = fit_student_t_copula(
        pseudo_observations=pseudo,
        nu_lower_bound=copula_config["nu_lower_bound"],
        nu_upper_bound=copula_config["nu_upper_bound"],
    )

    fitted_nu = float(params["degrees_of_freedom"].iloc[0])

    print(f"Fitted Student-t copula degrees of freedom: {fitted_nu:.4f}")

    print("Computing asymptotic Student-t copula tail dependence...")
    asymptotic_tail_dependence = compute_student_t_asymptotic_tail_dependence(
        correlation=correlation,
        degrees_of_freedom=fitted_nu,
    )

    print("Simulating from fitted Student-t copula...")
    simulated_uniforms = simulate_student_t_copula(
        correlation=correlation,
        degrees_of_freedom=fitted_nu,
        observations=copula_config["simulation_observations"],
        random_seed=copula_config["random_seed"],
    )

    print("Computing simulated finite-threshold tail dependence...")
    simulated_tail_dependence = compute_simulated_tail_dependence_tables(
        simulated_uniforms=simulated_uniforms,
        quantile_levels=quantile_levels,
    )

    simulated_matrix_q95 = compute_tail_dependence_from_uniforms(
        pseudo_observations=simulated_uniforms,
        quantile_level=main_quantile_level,
    )

    print("Reading empirical q=0.95 tail-dependence matrix...")
    empirical_matrix_q95 = pd.read_parquet(empirical_matrix_path)

    print("Comparing empirical and Student-t copula q=0.95 tail dependence...")
    comparison_q95 = compare_empirical_and_copula_tail_dependence(
        empirical_matrix=empirical_matrix_q95,
        copula_matrix=simulated_matrix_q95,
    )

    params.to_parquet(paths_config["processed_files"]["student_t_copula_params"])
    correlation.to_parquet(
        paths_config["processed_files"]["student_t_copula_correlation"]
    )
    asymptotic_tail_dependence.to_parquet(
        paths_config["processed_files"][
            "student_t_copula_asymptotic_tail_dependence"
        ]
    )
    simulated_tail_dependence.to_parquet(
        paths_config["processed_files"]["student_t_copula_simulated_tail_dependence"]
    )
    comparison_q95.to_parquet(
        paths_config["processed_files"]["student_t_copula_empirical_comparison_q95"]
    )

    params_output = params.copy()
    correlation_output = correlation.copy()
    asymptotic_output = asymptotic_tail_dependence.copy()
    simulated_output = simulated_tail_dependence.copy()
    comparison_output = comparison_q95.copy()

    for df in [
        params_output,
        correlation_output,
        asymptotic_output,
        simulated_output,
        comparison_output,
    ]:
        for column in df.columns:
            if pd.api.types.is_float_dtype(df[column]):
                df[column] = df[column].round(6)

    save_table_csv(params_output, "outputs/tables/student_t_copula_params.csv")
    save_table_csv(
        correlation_output,
        "outputs/tables/student_t_copula_correlation.csv",
    )
    save_table_csv(
        asymptotic_output,
        "outputs/tables/student_t_copula_asymptotic_tail_dependence.csv",
    )
    save_table_csv(
        simulated_output,
        "outputs/tables/student_t_copula_simulated_tail_dependence.csv",
    )
    save_table_csv(
        comparison_output,
        "outputs/tables/student_t_copula_empirical_comparison_q95.csv",
    )

    save_table_latex(
        params_output,
        "outputs/tables/student_t_copula_params.tex",
        caption="Student-t copula parameter estimates.",
        label="tab:student_t_copula_params",
    )

    print("\nStudent-t copula parameters:")
    print(params_output.to_string(index=False))

    print("\nStudent-t copula correlation matrix:")
    print(correlation_output.to_string())

    print("\nStudent-t copula asymptotic tail-dependence matrix:")
    print(asymptotic_output.to_string())

    print("\nLargest empirical versus copula q=0.95 tail-dependence differences:")
    print(comparison_output.head(30).to_string(index=False))

    print("\nStudent-t copula benchmark outputs saved successfully.")


if __name__ == "__main__":
    main()