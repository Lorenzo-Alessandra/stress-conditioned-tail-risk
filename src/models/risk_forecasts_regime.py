from __future__ import annotations

import pandas as pd


def build_regime_residual_risk_lookup(
    regime_var_es: pd.DataFrame,
    risk_measure: str,
) -> dict[tuple[float, int], pd.Series]:
    """
    Build a lookup from probability level and regime to residual VaR or ES by asset.

    Parameters
    ----------
    regime_var_es:
        Regime-specific residual EVT VaR/ES table.
    risk_measure:
        Either 'residual_var' or 'residual_es'.

    Returns
    -------
    dict[tuple[float, int], pd.Series]
        Mapping from (probability_level, stress_regime) to asset-indexed risk.
    """
    if risk_measure not in {"residual_var", "residual_es"}:
        raise ValueError("risk_measure must be 'residual_var' or 'residual_es'.")

    required_columns = {
        "asset",
        "probability_level",
        "stress_regime",
        risk_measure,
    }

    missing_columns = required_columns - set(regime_var_es.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in regime EVT table: {missing_columns}")

    lookup: dict[tuple[float, int], pd.Series] = {}

    grouped = regime_var_es.groupby(["probability_level", "stress_regime"])

    for (probability_level, stress_regime), group in grouped:
        series = group.set_index("asset")[risk_measure]
        lookup[(float(probability_level), int(stress_regime))] = series

    return lookup


def compute_regime_conditional_risk_forecasts(
    conditional_volatility: pd.DataFrame,
    mu_raw: pd.Series,
    stress_regime: pd.DataFrame,
    residual_risk_lookup: dict[tuple[float, int], pd.Series],
) -> dict[float, pd.DataFrame]:
    """
    Compute regime-specific conditional VaR or ES forecasts.

    Formula:
        risk_{L,t,p}^{regime}
        = -mu_i + sigma_{i,t} * risk_{Y,i,p}^{D_t}

    Parameters
    ----------
    conditional_volatility:
        Wide dataframe of GARCH conditional volatility in raw return units.
    mu_raw:
        GARCH mean estimates in raw return units, indexed by asset.
    stress_regime:
        Dataframe containing a stress_regime column indexed by date.
    residual_risk_lookup:
        Mapping from (probability_level, stress_regime) to residual risk by asset.

    Returns
    -------
    dict[float, pd.DataFrame]
        Mapping from probability level to conditional risk forecast panel.
    """
    if "stress_regime" not in stress_regime.columns:
        raise ValueError("stress_regime dataframe must contain 'stress_regime'.")

    common_index = conditional_volatility.index.intersection(stress_regime.index)

    sigma = conditional_volatility.loc[common_index].copy()
    regime = stress_regime.loc[common_index, "stress_regime"].astype(int)

    probability_levels = sorted(
        {probability_level for probability_level, _ in residual_risk_lookup.keys()}
    )

    forecasts: dict[float, pd.DataFrame] = {}

    for probability_level in probability_levels:
        forecast = pd.DataFrame(index=sigma.index, columns=sigma.columns, dtype=float)

        for regime_value in [0, 1]:
            key = (probability_level, regime_value)

            if key not in residual_risk_lookup:
                raise ValueError(f"Missing residual risk estimates for key: {key}")

            residual_risk = residual_risk_lookup[key]

            regime_dates = regime[regime == regime_value].index

            for asset in sigma.columns:
                if asset not in residual_risk.index:
                    raise ValueError(
                        f"Missing residual risk estimate for asset {asset}, "
                        f"probability={probability_level}, regime={regime_value}"
                    )

                if asset not in mu_raw.index:
                    raise ValueError(f"Missing GARCH mean estimate for asset {asset}")

                forecast.loc[regime_dates, asset] = (
                    -mu_raw.loc[asset]
                    + sigma.loc[regime_dates, asset] * residual_risk.loc[asset]
                )

        forecasts[probability_level] = forecast.sort_index()

    return forecasts


def stack_regime_forecast_panels(
    forecast_by_level: dict[float, pd.DataFrame],
    stress_regime: pd.DataFrame,
    value_name: str,
) -> pd.DataFrame:
    """
    Stack regime-specific forecast panels into long format.

    Parameters
    ----------
    forecast_by_level:
        Mapping from probability level to forecast panel.
    stress_regime:
        Dataframe containing stress_regime indexed by date.
    value_name:
        Name of forecast column.

    Returns
    -------
    pd.DataFrame
        Long dataframe with date, asset, probability_level, stress_regime, and value.
    """
    records = []

    for probability_level, forecast_panel in forecast_by_level.items():
        stacked = forecast_panel.stack().reset_index()
        stacked.columns = ["date", "asset", value_name]
        stacked["probability_level"] = probability_level

        regime_dates = stress_regime[["stress_regime"]].reset_index()
        regime_dates.columns = ["date", "stress_regime"]

        stacked = stacked.merge(regime_dates, on="date", how="left")

        records.append(stacked)

    result = pd.concat(records, axis=0, ignore_index=True)

    result = result[
        ["date", "asset", "probability_level", "stress_regime", value_name]
    ]

    result = result.sort_values(["asset", "probability_level", "date"]).reset_index(
        drop=True
    )

    return result