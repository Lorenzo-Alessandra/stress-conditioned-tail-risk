from __future__ import annotations

import pandas as pd

from src.models.evt import compute_evt_var_es_table, fit_gpd_for_asset


def fit_regime_specific_gpd(
    residual_regime_long: pd.DataFrame,
    threshold_quantile: float,
) -> pd.DataFrame:
    """
    Fit separate POT-GPD models by asset and stress regime.

    Parameters
    ----------
    residual_regime_long:
        Long dataframe with date, asset, residual_loss, and stress_regime.
    threshold_quantile:
        Within-regime threshold quantile.

    Returns
    -------
    pd.DataFrame
        Regime-specific GPD parameter table.
    """
    required_columns = {"date", "asset", "residual_loss", "stress_regime"}
    missing_columns = required_columns - set(residual_regime_long.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    records = []

    grouped = residual_regime_long.groupby(["asset", "stress_regime"])

    for (asset, stress_regime), group in grouped:
        regime_label = "stress" if int(stress_regime) == 1 else "calm"

        print(
            f"Fitting regime-specific POT-GPD for {asset}, "
            f"regime={regime_label}, threshold={threshold_quantile}"
        )

        series = group.set_index("date")["residual_loss"].dropna()

        result = fit_gpd_for_asset(
            residual_losses=series,
            asset=asset,
            threshold_quantile=threshold_quantile,
        )

        records.append(
            {
                "asset": asset,
                "stress_regime": int(stress_regime),
                "regime_label": regime_label,
                "threshold_quantile": result.threshold_quantile,
                "threshold_value": result.threshold_value,
                "n_observations": result.n_observations,
                "exceedance_count": result.exceedance_count,
                "shape_xi": result.shape_xi,
                "scale_beta": result.scale_beta,
                "loglikelihood": result.loglikelihood,
                "convergence_success": result.convergence_success,
                "optimizer_message": result.optimizer_message,
            }
        )

    params = pd.DataFrame(records)

    params = params.sort_values(["asset", "stress_regime"]).reset_index(drop=True)

    return params


def compute_regime_var_es_table(
    regime_gpd_params: pd.DataFrame,
    probability_levels: list[float],
) -> pd.DataFrame:
    """
    Compute regime-specific residual VaR and ES estimates.

    Parameters
    ----------
    regime_gpd_params:
        Regime-specific GPD parameter table.
    probability_levels:
        Probability levels.

    Returns
    -------
    pd.DataFrame
        Regime-specific residual VaR and ES estimates.
    """
    all_tables = []

    grouped = regime_gpd_params.groupby("stress_regime")

    for stress_regime, group in grouped:
        params_for_evt = group.set_index("asset")

        var_es = compute_evt_var_es_table(
            gpd_params=params_for_evt,
            probability_levels=probability_levels,
        )

        labels = group.set_index("asset")["regime_label"]
        var_es["stress_regime"] = int(stress_regime)
        var_es["regime_label"] = var_es["asset"].map(labels)

        all_tables.append(var_es)

    result = pd.concat(all_tables, axis=0, ignore_index=True)

    result = result.sort_values(
        ["asset", "probability_level", "stress_regime"]
    ).reset_index(drop=True)

    return result


def compare_calm_stress_parameters(
    regime_gpd_params: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare stress and calm GPD parameters for each asset.

    Parameters
    ----------
    regime_gpd_params:
        Regime-specific GPD parameter table.

    Returns
    -------
    pd.DataFrame
        Wide comparison table.
    """
    calm = regime_gpd_params[regime_gpd_params["stress_regime"] == 0].set_index(
        "asset"
    )
    stress = regime_gpd_params[regime_gpd_params["stress_regime"] == 1].set_index(
        "asset"
    )

    comparison = pd.DataFrame(index=calm.index)

    comparison["xi_calm"] = calm["shape_xi"]
    comparison["xi_stress"] = stress["shape_xi"]
    comparison["xi_difference_stress_minus_calm"] = (
        comparison["xi_stress"] - comparison["xi_calm"]
    )

    comparison["beta_calm"] = calm["scale_beta"]
    comparison["beta_stress"] = stress["scale_beta"]
    comparison["beta_ratio_stress_to_calm"] = (
        comparison["beta_stress"] / comparison["beta_calm"]
    )

    comparison["threshold_calm"] = calm["threshold_value"]
    comparison["threshold_stress"] = stress["threshold_value"]
    comparison["threshold_ratio_stress_to_calm"] = (
        comparison["threshold_stress"] / comparison["threshold_calm"]
    )

    comparison["n_calm"] = calm["n_observations"]
    comparison["n_stress"] = stress["n_observations"]
    comparison["exceedances_calm"] = calm["exceedance_count"]
    comparison["exceedances_stress"] = stress["exceedance_count"]

    comparison = comparison.reset_index()

    return comparison