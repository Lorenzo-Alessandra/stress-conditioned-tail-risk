import pandas as pd

from src.models.evt import (
    compute_evt_var_es_table,
    fit_gpd_to_excesses_for_asset,
)
from src.models.evt_declustering import extract_excesses_by_declustering_method
from src.utils.config import load_yaml
from src.visualization.tables import save_table_csv, save_table_latex


def fit_declustering_specification(
    residual_losses: pd.DataFrame,
    threshold_quantile: float,
    method: str,
    run_length: int | None,
) -> pd.DataFrame:
    """
    Fit GPD models for one declustering specification.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    threshold_quantile:
        Threshold quantile.
    method:
        Declustering method: 'none' or 'runs'.
    run_length:
        Runs declustering length if method='runs'.

    Returns
    -------
    pd.DataFrame
        GPD parameter table for this specification.
    """
    records = []

    for asset in residual_losses.columns:
        print(
            f"Fitting {asset}: method={method}, "
            f"run_length={run_length}, threshold={threshold_quantile}"
        )

        series = residual_losses[asset].dropna()
        threshold_value = float(series.quantile(threshold_quantile))

        excesses = extract_excesses_by_declustering_method(
            series=series,
            threshold=threshold_value,
            method=method,
            run_length=run_length,
        )

        result = fit_gpd_to_excesses_for_asset(
            excesses=excesses,
            asset=asset,
            threshold_quantile=threshold_quantile,
            threshold_value=threshold_value,
            n_observations=len(series),
            declustering_method=method,
            run_length=run_length,
        )

        records.append(
            {
                "asset": result.asset,
                "threshold_quantile": result.threshold_quantile,
                "threshold_value": result.threshold_value,
                "declustering_method": method,
                "run_length": run_length if run_length is not None else 0,
                "n_observations": result.n_observations,
                "exceedance_count": result.exceedance_count,
                "shape_xi": result.shape_xi,
                "scale_beta": result.scale_beta,
                "loglikelihood": result.loglikelihood,
                "convergence_success": result.convergence_success,
                "optimizer_message": result.optimizer_message,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    residual_losses_path = paths_config["processed_files"][
        "standardized_residual_losses"
    ]

    threshold_quantile = model_config["evt"]["main_threshold_quantile"]
    probability_levels = model_config["evt"]["tail_probability_levels"]

    print("Reading standardized residual losses...")
    residual_losses = pd.read_parquet(residual_losses_path)

    specifications = [
        {"method": "none", "run_length": None},
        {"method": "runs", "run_length": 3},
        {"method": "runs", "run_length": 5},
    ]

    all_params = []
    all_var_es = []

    for spec in specifications:
        params = fit_declustering_specification(
            residual_losses=residual_losses,
            threshold_quantile=threshold_quantile,
            method=spec["method"],
            run_length=spec["run_length"],
        )

        var_es = compute_evt_var_es_table(
            gpd_params=params.set_index("asset"),
            probability_levels=probability_levels,
        )

        var_es["declustering_method"] = spec["method"]
        var_es["run_length"] = spec["run_length"] if spec["run_length"] is not None else 0

        all_params.append(params)
        all_var_es.append(var_es)

    params_robustness = pd.concat(all_params, ignore_index=True)
    var_es_robustness = pd.concat(all_var_es, ignore_index=True)

    params_robustness = params_robustness.sort_values(
        ["asset", "declustering_method", "run_length"]
    ).reset_index(drop=True)

    var_es_robustness = var_es_robustness.sort_values(
        ["asset", "probability_level", "declustering_method", "run_length"]
    ).reset_index(drop=True)

    params_robustness.to_parquet(
        paths_config["processed_files"]["evt_gpd_params_declustering_robustness"]
    )

    var_es_robustness.to_parquet(
        paths_config["processed_files"]["evt_var_es_declustering_robustness"]
    )

    params_output = params_robustness.copy()
    var_es_output = var_es_robustness.copy()

    param_round_cols = [
        "threshold_value",
        "shape_xi",
        "scale_beta",
        "loglikelihood",
    ]

    for column in param_round_cols:
        params_output[column] = params_output[column].round(6)

    var_es_round_cols = [
        "residual_var",
        "residual_es",
        "shape_xi",
        "scale_beta",
        "threshold_value",
    ]

    for column in var_es_round_cols:
        var_es_output[column] = var_es_output[column].round(6)

    params_csv_path = "outputs/tables/evt_gpd_params_declustering_robustness.csv"
    params_latex_path = "outputs/tables/evt_gpd_params_declustering_robustness.tex"

    var_es_csv_path = "outputs/tables/evt_var_es_declustering_robustness.csv"
    var_es_latex_path = "outputs/tables/evt_var_es_declustering_robustness.tex"

    save_table_csv(params_output, params_csv_path)
    save_table_latex(
        params_output,
        params_latex_path,
        caption="POT-GPD parameter estimates across declustering specifications.",
        label="tab:evt_gpd_params_declustering_robustness",
    )

    save_table_csv(var_es_output, var_es_csv_path)
    save_table_latex(
        var_es_output,
        var_es_latex_path,
        caption="EVT residual VaR and ES estimates across declustering specifications.",
        label="tab:evt_var_es_declustering_robustness",
    )

    print("\nDeclustering robustness parameter estimates:")
    print(params_output.to_string(index=False))

    print("\nDeclustering robustness VaR/ES estimates:")
    print(var_es_output.to_string(index=False))

    print("\nDeclustering robustness outputs saved successfully.")
    print(params_csv_path)
    print(var_es_csv_path)


if __name__ == "__main__":
    main()