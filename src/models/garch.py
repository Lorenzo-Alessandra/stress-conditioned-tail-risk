from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from arch import arch_model


@dataclass
class GarchFitResult:
    """
    Container for GARCH model outputs for one asset.
    """

    asset: str
    params: pd.Series
    conditional_volatility: pd.Series
    standardized_residuals: pd.Series
    convergence_flag: int
    loglikelihood: float
    aic: float
    bic: float


def fit_gjr_garch_student_t(
    returns: pd.Series,
    asset: str,
    scale_returns: bool = True,
) -> GarchFitResult:
    """
    Fit a GJR-GARCH(1,1) model with Student-t innovations to one return series.

    Parameters
    ----------
    returns:
        Daily log return series.
    asset:
        Asset name.
    scale_returns:
        If True, multiply returns by 100 before estimation.

    Returns
    -------
    GarchFitResult
        Model outputs for the asset.
    """
    returns = returns.dropna().copy()

    if returns.empty:
        raise ValueError(f"Return series is empty for asset {asset}.")

    estimation_returns = returns * 100 if scale_returns else returns

    model = arch_model(
        estimation_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="StudentsT",
        rescale=False,
    )

    fitted = model.fit(disp="off")

    conditional_volatility = fitted.conditional_volatility

    residuals = fitted.resid
    standardized_residuals = residuals / conditional_volatility

    if scale_returns:
        conditional_volatility = conditional_volatility / 100
        residuals = residuals / 100

    conditional_volatility.name = asset
    standardized_residuals.name = asset

    return GarchFitResult(
        asset=asset,
        params=fitted.params,
        conditional_volatility=conditional_volatility,
        standardized_residuals=standardized_residuals,
        convergence_flag=fitted.convergence_flag,
        loglikelihood=fitted.loglikelihood,
        aic=fitted.aic,
        bic=fitted.bic,
    )


def fit_garch_panel(
    returns: pd.DataFrame,
    scale_returns: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit GJR-GARCH(1,1) models to every column in a return panel.

    Parameters
    ----------
    returns:
        Wide dataframe of daily returns.
    scale_returns:
        If True, multiply returns by 100 before estimation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Parameter table, conditional volatility panel, standardized residual panel.
    """
    parameter_records = []
    volatility_series = []
    residual_series = []

    for asset in returns.columns:
        print(f"Fitting GJR-GARCH(1,1)-Student-t for {asset}...")

        result = fit_gjr_garch_student_t(
            returns=returns[asset],
            asset=asset,
            scale_returns=scale_returns,
        )

        parameter_record = result.params.to_dict()
        parameter_record["asset"] = asset
        parameter_record["convergence_flag"] = result.convergence_flag
        parameter_record["loglikelihood"] = result.loglikelihood
        parameter_record["aic"] = result.aic
        parameter_record["bic"] = result.bic

        parameter_records.append(parameter_record)
        volatility_series.append(result.conditional_volatility)
        residual_series.append(result.standardized_residuals)

    params = pd.DataFrame(parameter_records).set_index("asset")
    conditional_volatility = pd.concat(volatility_series, axis=1)
    standardized_residuals = pd.concat(residual_series, axis=1)

    conditional_volatility = conditional_volatility.reindex(returns.index)
    standardized_residuals = standardized_residuals.reindex(returns.index)

    return params, conditional_volatility, standardized_residuals


def compute_standardized_residual_losses(
    standardized_residuals: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute standardized residual losses.

    Parameters
    ----------
    standardized_residuals:
        Wide dataframe of standardized residuals.

    Returns
    -------
    pd.DataFrame
        Standardized residual losses.
    """
    residual_losses = -standardized_residuals
    residual_losses = residual_losses.sort_index()

    return residual_losses