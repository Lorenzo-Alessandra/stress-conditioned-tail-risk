from __future__ import annotations

import pandas as pd


def build_quantile_stress_regime(
    stress_variables: pd.DataFrame,
    variable: str,
    quantile: float,
) -> pd.DataFrame:
    """
    Build a binary stress-regime indicator using a quantile threshold.

    Parameters
    ----------
    stress_variables:
        Dataframe containing stress variables indexed by date.
    variable:
        Stress variable used to define the regime.
    quantile:
        Quantile used as the stress threshold.

    Returns
    -------
    pd.DataFrame
        Dataframe with stress variable, threshold, and regime indicator.
    """
    if variable not in stress_variables.columns:
        raise ValueError(f"Variable not found in stress variables: {variable}")

    series = stress_variables[variable].dropna()

    if series.empty:
        raise ValueError(f"Stress variable {variable} has no non-missing values.")

    threshold = float(series.quantile(quantile))

    regime = stress_variables[[variable]].copy()
    regime["stress_threshold"] = threshold
    regime["stress_regime"] = regime[variable] > threshold

    regime = regime.dropna(subset=[variable])
    regime["stress_regime"] = regime["stress_regime"].astype(int)

    return regime


def align_residual_losses_with_regime(
    residual_losses: pd.DataFrame,
    regime: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align standardized residual losses with a stress-regime dataframe.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    regime:
        Dataframe containing a stress_regime column.

    Returns
    -------
    pd.DataFrame
        Long dataframe with date, asset, residual_loss, and stress_regime.
    """
    if "stress_regime" not in regime.columns:
        raise ValueError("Regime dataframe must contain 'stress_regime'.")

    common_index = residual_losses.index.intersection(regime.index)

    residual_losses_aligned = residual_losses.loc[common_index]
    regime_aligned = regime.loc[common_index]

    residual_long = residual_losses_aligned.stack().reset_index()
    residual_long.columns = ["date", "asset", "residual_loss"]

    regime_long = regime_aligned[["stress_regime"]].reset_index()
    regime_long.columns = ["date", "stress_regime"]

    merged = residual_long.merge(
        regime_long,
        on="date",
        how="inner",
    )

    merged = merged.sort_values(["asset", "date"]).reset_index(drop=True)

    return merged