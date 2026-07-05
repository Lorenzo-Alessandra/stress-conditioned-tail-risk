from __future__ import annotations

import pandas as pd


def extract_garch_mean_raw_units(
    garch_params: pd.DataFrame,
    scaled_by_100: bool = True,
) -> pd.Series:
    """
    Extract GARCH conditional mean estimates in raw return units.

    Parameters
    ----------
    garch_params:
        GARCH parameter table indexed by asset.
    scaled_by_100:
        Whether GARCH estimation was performed on returns multiplied by 100.

    Returns
    -------
    pd.Series
        Mean estimates in raw return units.
    """
    if "mu" not in garch_params.columns:
        raise ValueError("GARCH parameter table must contain a 'mu' column.")

    mu = garch_params["mu"].copy()

    if scaled_by_100:
        mu = mu / 100.0

    return mu


def build_residual_risk_lookup(
    evt_var_es: pd.DataFrame,
    risk_measure: str,
) -> dict[float, pd.Series]:
    """
    Build a lookup from probability level to residual VaR or ES by asset.

    Parameters
    ----------
    evt_var_es:
        Table containing residual EVT VaR and ES estimates.
    risk_measure:
        Either 'residual_var' or 'residual_es'.

    Returns
    -------
    dict[float, pd.Series]
        Mapping from probability level to asset-indexed residual risk estimates.
    """
    if risk_measure not in {"residual_var", "residual_es"}:
        raise ValueError("risk_measure must be 'residual_var' or 'residual_es'.")

    required_columns = {"asset", "probability_level", risk_measure}

    missing_columns = required_columns - set(evt_var_es.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in EVT table: {missing_columns}")

    lookup: dict[float, pd.Series] = {}

    for probability_level, group in evt_var_es.groupby("probability_level"):
        series = group.set_index("asset")[risk_measure]
        lookup[float(probability_level)] = series

    return lookup


def compute_conditional_risk_forecasts(
    conditional_volatility: pd.DataFrame,
    mu_raw: pd.Series,
    residual_risk_by_level: dict[float, pd.Series],
) -> dict[float, pd.DataFrame]:
    """
    Compute conditional VaR or ES forecasts on the original loss scale.

    Formula:
        risk_{L,t,p} = -mu_i + sigma_{i,t} * risk_{Y,i,p}

    Parameters
    ----------
    conditional_volatility:
        Wide dataframe of GARCH conditional volatility in raw return units.
    mu_raw:
        Series of GARCH mean estimates in raw return units, indexed by asset.
    residual_risk_by_level:
        Mapping from probability level to residual risk estimates by asset.

    Returns
    -------
    dict[float, pd.DataFrame]
        Mapping from probability level to conditional risk forecast panel.
    """
    forecasts: dict[float, pd.DataFrame] = {}

    for probability_level, residual_risk in residual_risk_by_level.items():
        missing_assets = [
            asset
            for asset in conditional_volatility.columns
            if asset not in residual_risk.index or asset not in mu_raw.index
        ]

        if missing_assets:
            raise ValueError(
                f"Missing residual risk or mean estimates for assets: {missing_assets}"
            )

        forecast = pd.DataFrame(index=conditional_volatility.index)

        for asset in conditional_volatility.columns:
            forecast[asset] = (
                -mu_raw.loc[asset]
                + conditional_volatility[asset] * residual_risk.loc[asset]
            )

        forecasts[probability_level] = forecast

    return forecasts


def stack_forecast_panels(
    forecast_by_level: dict[float, pd.DataFrame],
    value_name: str,
) -> pd.DataFrame:
    """
    Stack probability-level forecast panels into a long dataframe.

    Parameters
    ----------
    forecast_by_level:
        Mapping from probability level to forecast panel.
    value_name:
        Name of the forecast value column.

    Returns
    -------
    pd.DataFrame
        Long dataframe with date, asset, probability_level, and forecast values.
    """
    records = []

    for probability_level, forecast_panel in forecast_by_level.items():
        stacked = forecast_panel.stack().reset_index()
        stacked.columns = ["date", "asset", value_name]
        stacked["probability_level"] = probability_level
        records.append(stacked)

    result = pd.concat(records, axis=0, ignore_index=True)

    result = result[["date", "asset", "probability_level", value_name]]
    result = result.sort_values(["asset", "probability_level", "date"]).reset_index(
        drop=True
    )

    return result