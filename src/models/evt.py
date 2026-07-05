from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class GPDFitResult:
    """
    Container for one fitted GPD model.
    """

    asset: str
    threshold_quantile: float
    threshold_value: float
    n_observations: int
    exceedance_count: int
    shape_xi: float
    scale_beta: float
    loglikelihood: float
    convergence_success: bool
    optimizer_message: str


def gpd_negative_loglikelihood(
    parameters: np.ndarray,
    excesses: np.ndarray,
) -> float:
    """
    Compute the negative GPD log-likelihood.

    Parameters
    ----------
    parameters:
        Array containing shape xi and log scale.
        The scale is parameterized as beta = exp(log_beta)
        to enforce beta > 0.
    excesses:
        Positive threshold excesses.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    xi = parameters[0]
    log_beta = parameters[1]
    beta = np.exp(log_beta)

    if beta <= 0:
        return np.inf

    support = 1.0 + xi * excesses / beta

    if np.any(support <= 0):
        return np.inf

    n = len(excesses)

    if abs(xi) < 1e-6:
        loglikelihood = -n * np.log(beta) - np.sum(excesses / beta)
    else:
        loglikelihood = (
            -n * np.log(beta)
            - (1.0 + 1.0 / xi) * np.sum(np.log(support))
        )

    return -float(loglikelihood)


def fit_gpd_mle(
    excesses: pd.Series | np.ndarray,
    initial_xi: float = 0.1,
) -> tuple[float, float, float, bool, str]:
    """
    Fit a GPD model to threshold excesses by maximum likelihood.

    Parameters
    ----------
    excesses:
        Positive threshold excesses.
    initial_xi:
        Initial value for the shape parameter.

    Returns
    -------
    tuple[float, float, float, bool, str]
        Estimated xi, estimated beta, log-likelihood,
        optimizer success flag, and optimizer message.
    """
    excesses_array = np.asarray(excesses, dtype=float)
    excesses_array = excesses_array[np.isfinite(excesses_array)]

    if len(excesses_array) == 0:
        raise ValueError("Cannot fit GPD with zero excesses.")

    if np.any(excesses_array < 0):
        raise ValueError("GPD excesses must be non-negative.")

    initial_beta = np.std(excesses_array)

    if initial_beta <= 0:
        initial_beta = np.mean(excesses_array)

    if initial_beta <= 0:
        raise ValueError("Cannot initialize GPD scale parameter.")

    initial_parameters = np.array([initial_xi, np.log(initial_beta)])

    result = minimize(
        fun=gpd_negative_loglikelihood,
        x0=initial_parameters,
        args=(excesses_array,),
        method="Nelder-Mead",
        options={
            "maxiter": 10000,
            "xatol": 1e-8,
            "fatol": 1e-8,
        },
    )

    xi_hat = float(result.x[0])
    beta_hat = float(np.exp(result.x[1]))
    loglikelihood = -float(result.fun)

    return xi_hat, beta_hat, loglikelihood, bool(result.success), str(result.message)


def extract_excesses(
    series: pd.Series,
    threshold_quantile: float,
) -> tuple[float, pd.Series]:
    """
    Extract threshold excesses from a residual loss series.

    Parameters
    ----------
    series:
        Series of standardized residual losses.
    threshold_quantile:
        Empirical quantile used as threshold.

    Returns
    -------
    tuple[float, pd.Series]
        Threshold value and positive excesses.
    """
    clean_series = series.dropna()

    if clean_series.empty:
        raise ValueError("Cannot extract excesses from empty series.")

    threshold = float(clean_series.quantile(threshold_quantile))
    exceedances = clean_series[clean_series > threshold]
    excesses = exceedances - threshold

    return threshold, excesses


def fit_gpd_for_asset(
    residual_losses: pd.Series,
    asset: str,
    threshold_quantile: float,
) -> GPDFitResult:
    """
    Fit a GPD model to one asset's standardized residual losses.

    Parameters
    ----------
    residual_losses:
        Series of standardized residual losses.
    asset:
        Asset name.
    threshold_quantile:
        Empirical threshold quantile.

    Returns
    -------
    GPDFitResult
        Fitted GPD result.
    """
    clean_series = residual_losses.dropna()
    threshold, excesses = extract_excesses(
        series=clean_series,
        threshold_quantile=threshold_quantile,
    )

    xi_hat, beta_hat, loglikelihood, success, message = fit_gpd_mle(excesses)

    return GPDFitResult(
        asset=asset,
        threshold_quantile=threshold_quantile,
        threshold_value=threshold,
        n_observations=len(clean_series),
        exceedance_count=len(excesses),
        shape_xi=xi_hat,
        scale_beta=beta_hat,
        loglikelihood=loglikelihood,
        convergence_success=success,
        optimizer_message=message,
    )


def fit_gpd_to_excesses_for_asset(
    excesses: pd.Series,
    asset: str,
    threshold_quantile: float,
    threshold_value: float,
    n_observations: int,
    declustering_method: str,
    run_length: int | None,
) -> GPDFitResult:
    """
    Fit a GPD model to already extracted excesses for one asset.

    Parameters
    ----------
    excesses:
        Threshold excesses.
    asset:
        Asset name.
    threshold_quantile:
        Empirical threshold quantile.
    threshold_value:
        Threshold value.
    n_observations:
        Total number of original observations.
    declustering_method:
        Declustering method used.
    run_length:
        Run length if runs declustering is used.

    Returns
    -------
    GPDFitResult
        Fitted GPD result.
    """
    xi_hat, beta_hat, loglikelihood, success, message = fit_gpd_mle(excesses)

    return GPDFitResult(
        asset=asset,
        threshold_quantile=threshold_quantile,
        threshold_value=threshold_value,
        n_observations=n_observations,
        exceedance_count=len(excesses),
        shape_xi=xi_hat,
        scale_beta=beta_hat,
        loglikelihood=loglikelihood,
        convergence_success=success,
        optimizer_message=message,
    )


def fit_gpd_panel(
    residual_losses: pd.DataFrame,
    threshold_quantile: float,
) -> pd.DataFrame:
    """
    Fit static POT-GPD models to every asset in a residual-loss panel.

    Parameters
    ----------
    residual_losses:
        Wide dataframe of standardized residual losses.
    threshold_quantile:
        Empirical threshold quantile.

    Returns
    -------
    pd.DataFrame
        GPD parameter table.
    """
    records = []

    for asset in residual_losses.columns:
        print(f"Fitting POT-GPD for {asset} at threshold {threshold_quantile}...")

        result = fit_gpd_for_asset(
            residual_losses=residual_losses[asset],
            asset=asset,
            threshold_quantile=threshold_quantile,
        )

        records.append(
            {
                "asset": result.asset,
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

    params = pd.DataFrame(records).set_index("asset")

    return params


def compute_evt_var(
    threshold_value: float,
    shape_xi: float,
    scale_beta: float,
    n_observations: int,
    exceedance_count: int,
    probability_level: float,
) -> float:
    """
    Compute EVT-based residual VaR from fitted GPD parameters.

    Parameters
    ----------
    threshold_value:
        POT threshold u.
    shape_xi:
        GPD shape parameter xi.
    scale_beta:
        GPD scale parameter beta.
    n_observations:
        Total number of observations.
    exceedance_count:
        Number of threshold exceedances.
    probability_level:
        VaR probability level p.

    Returns
    -------
    float
        EVT residual VaR estimate.
    """
    tail_probability = 1.0 - probability_level
    exceedance_rate = exceedance_count / n_observations

    if tail_probability <= 0:
        raise ValueError("probability_level must be below 1.")

    if probability_level <= 1.0 - exceedance_rate:
        raise ValueError(
            "probability_level must be above the threshold probability."
        )

    if abs(shape_xi) < 1e-6:
        var = threshold_value + scale_beta * np.log(exceedance_rate / tail_probability)
    else:
        var = threshold_value + (scale_beta / shape_xi) * (
            (tail_probability / exceedance_rate) ** (-shape_xi) - 1.0
        )

    return float(var)


def compute_evt_es(
    var_value: float,
    threshold_value: float,
    shape_xi: float,
    scale_beta: float,
) -> float:
    """
    Compute EVT-based residual Expected Shortfall.

    Parameters
    ----------
    var_value:
        EVT residual VaR estimate.
    threshold_value:
        POT threshold u.
    shape_xi:
        GPD shape parameter xi.
    scale_beta:
        GPD scale parameter beta.

    Returns
    -------
    float
        EVT residual ES estimate.
    """
    if shape_xi >= 1:
        return np.nan

    es = (var_value + scale_beta - shape_xi * threshold_value) / (1.0 - shape_xi)

    return float(es)


def compute_evt_var_es_table(
    gpd_params: pd.DataFrame,
    probability_levels: list[float],
) -> pd.DataFrame:
    """
    Compute EVT VaR and ES estimates for every fitted GPD model.

    Parameters
    ----------
    gpd_params:
        GPD parameter table.
    probability_levels:
        Probability levels for VaR and ES.

    Returns
    -------
    pd.DataFrame
        VaR and ES estimates.
    """
    records = []

    for asset, row in gpd_params.iterrows():
        for probability_level in probability_levels:
            var_value = compute_evt_var(
                threshold_value=row["threshold_value"],
                shape_xi=row["shape_xi"],
                scale_beta=row["scale_beta"],
                n_observations=int(row["n_observations"]),
                exceedance_count=int(row["exceedance_count"]),
                probability_level=probability_level,
            )

            es_value = compute_evt_es(
                var_value=var_value,
                threshold_value=row["threshold_value"],
                shape_xi=row["shape_xi"],
                scale_beta=row["scale_beta"],
            )

            records.append(
                {
                    "asset": asset,
                    "probability_level": probability_level,
                    "residual_var": var_value,
                    "residual_es": es_value,
                    "shape_xi": row["shape_xi"],
                    "scale_beta": row["scale_beta"],
                    "threshold_quantile": row["threshold_quantile"],
                    "threshold_value": row["threshold_value"],
                    "exceedance_count": int(row["exceedance_count"]),
                }
            )

    return pd.DataFrame(records)