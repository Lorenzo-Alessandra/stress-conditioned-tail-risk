from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_t, t


def compute_pseudo_observations(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert each marginal series to pseudo-uniform observations using ranks.

    Parameters
    ----------
    data:
        Wide dataframe of losses.

    Returns
    -------
    pd.DataFrame
        Pseudo-observations in (0, 1).
    """
    if data.empty:
        raise ValueError("Input dataframe is empty.")

    ranks = data.rank(axis=0, method="average")
    pseudo = ranks / (len(data) + 1.0)

    pseudo = pseudo.clip(lower=1e-6, upper=1.0 - 1e-6)

    return pseudo


def estimate_correlation_from_t_scores(
    pseudo_observations: pd.DataFrame,
    degrees_of_freedom: float,
) -> pd.DataFrame:
    """
    Estimate copula correlation by transforming pseudo-observations to t scores.

    Parameters
    ----------
    pseudo_observations:
        Pseudo-uniform observations.
    degrees_of_freedom:
        Student-t degrees of freedom.

    Returns
    -------
    pd.DataFrame
        Correlation matrix.
    """
    scores = pd.DataFrame(
        t.ppf(pseudo_observations.values, df=degrees_of_freedom),
        index=pseudo_observations.index,
        columns=pseudo_observations.columns,
    )

    correlation = scores.corr()

    return correlation


def student_t_copula_loglikelihood(
    pseudo_observations: pd.DataFrame,
    degrees_of_freedom: float,
) -> tuple[float, pd.DataFrame]:
    """
    Compute Student-t copula pseudo log-likelihood.

    Parameters
    ----------
    pseudo_observations:
        Pseudo-uniform observations.
    degrees_of_freedom:
        Student-t degrees of freedom.

    Returns
    -------
    tuple[float, pd.DataFrame]
        Log-likelihood and estimated correlation matrix.
    """
    if degrees_of_freedom <= 2.0:
        return -np.inf, pd.DataFrame()

    correlation = estimate_correlation_from_t_scores(
        pseudo_observations=pseudo_observations,
        degrees_of_freedom=degrees_of_freedom,
    )

    z = t.ppf(pseudo_observations.values, df=degrees_of_freedom)

    correlation_values = correlation.values

    sign, logdet = np.linalg.slogdet(correlation_values)

    if sign <= 0:
        return -np.inf, correlation

    try:
        joint_log_density = multivariate_t.logpdf(
            z,
            loc=np.zeros(z.shape[1]),
            shape=correlation_values,
            df=degrees_of_freedom,
        )
    except Exception:
        return -np.inf, correlation

    marginal_log_density = t.logpdf(
        z,
        df=degrees_of_freedom,
    ).sum(axis=1)

    copula_log_density = joint_log_density - marginal_log_density

    loglikelihood = float(np.sum(copula_log_density))

    return loglikelihood, correlation


def fit_student_t_copula(
    pseudo_observations: pd.DataFrame,
    nu_lower_bound: float = 2.1,
    nu_upper_bound: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a static Student-t copula by pseudo-likelihood.

    Parameters
    ----------
    pseudo_observations:
        Pseudo-uniform observations.
    nu_lower_bound:
        Lower bound for degrees of freedom.
    nu_upper_bound:
        Upper bound for degrees of freedom.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Parameter table and fitted correlation matrix.
    """
    def objective(nu: float) -> float:
        loglikelihood, _ = student_t_copula_loglikelihood(
            pseudo_observations=pseudo_observations,
            degrees_of_freedom=nu,
        )

        if not np.isfinite(loglikelihood):
            return 1e12

        return -loglikelihood

    result = minimize_scalar(
        objective,
        bounds=(nu_lower_bound, nu_upper_bound),
        method="bounded",
        options={"xatol": 1e-4},
    )

    fitted_nu = float(result.x)

    fitted_loglikelihood, fitted_correlation = student_t_copula_loglikelihood(
        pseudo_observations=pseudo_observations,
        degrees_of_freedom=fitted_nu,
    )

    params = pd.DataFrame(
        [
            {
                "model": "student_t_copula",
                "degrees_of_freedom": fitted_nu,
                "loglikelihood": fitted_loglikelihood,
                "observations": int(len(pseudo_observations)),
                "dimension": int(pseudo_observations.shape[1]),
                "convergence_success": bool(result.success),
                "optimizer_message": str(result.message),
            }
        ]
    )

    return params, fitted_correlation


def compute_student_t_asymptotic_tail_dependence(
    correlation: pd.DataFrame,
    degrees_of_freedom: float,
) -> pd.DataFrame:
    """
    Compute pairwise asymptotic tail dependence implied by a Student-t copula.

    Parameters
    ----------
    correlation:
        Copula correlation matrix.
    degrees_of_freedom:
        Student-t degrees of freedom.

    Returns
    -------
    pd.DataFrame
        Pairwise asymptotic tail-dependence matrix.
    """
    assets = list(correlation.index)

    matrix = pd.DataFrame(
        index=assets,
        columns=assets,
        dtype=float,
    )

    for asset_i in assets:
        for asset_j in assets:
            rho = float(correlation.loc[asset_i, asset_j])

            if asset_i == asset_j:
                matrix.loc[asset_i, asset_j] = 1.0
            else:
                argument = -np.sqrt(
                    ((degrees_of_freedom + 1.0) * (1.0 - rho))
                    / (1.0 + rho)
                )

                lambda_ij = 2.0 * t.cdf(
                    argument,
                    df=degrees_of_freedom + 1.0,
                )

                matrix.loc[asset_i, asset_j] = lambda_ij

    return matrix


def simulate_student_t_copula(
    correlation: pd.DataFrame,
    degrees_of_freedom: float,
    observations: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    Simulate pseudo-observations from a fitted Student-t copula.

    Parameters
    ----------
    correlation:
        Copula correlation matrix.
    degrees_of_freedom:
        Student-t degrees of freedom.
    observations:
        Number of simulated observations.
    random_seed:
        Random seed.

    Returns
    -------
    pd.DataFrame
        Simulated pseudo-uniform observations.
    """
    rng = np.random.default_rng(random_seed)

    simulated_t = multivariate_t.rvs(
        loc=np.zeros(correlation.shape[0]),
        shape=correlation.values,
        df=degrees_of_freedom,
        size=observations,
        random_state=rng,
    )

    simulated_u = t.cdf(simulated_t, df=degrees_of_freedom)

    simulated = pd.DataFrame(
        simulated_u,
        columns=correlation.columns,
    )

    return simulated


def compute_tail_dependence_from_uniforms(
    pseudo_observations: pd.DataFrame,
    quantile_level: float,
) -> pd.DataFrame:
    """
    Compute finite-threshold tail dependence from uniform observations.

    Parameters
    ----------
    pseudo_observations:
        Uniform observations.
    quantile_level:
        Threshold level.

    Returns
    -------
    pd.DataFrame
        Tail-dependence matrix.
    """
    indicators = pseudo_observations.gt(quantile_level).astype(int)

    assets = list(pseudo_observations.columns)

    matrix = pd.DataFrame(index=assets, columns=assets, dtype=float)

    for asset_i in assets:
        for conditioning_asset_j in assets:
            conditioning_count = indicators[conditioning_asset_j].sum()
            joint_count = (
                (indicators[asset_i] == 1)
                & (indicators[conditioning_asset_j] == 1)
            ).sum()

            if conditioning_count == 0:
                matrix.loc[asset_i, conditioning_asset_j] = np.nan
            else:
                matrix.loc[asset_i, conditioning_asset_j] = (
                    joint_count / conditioning_count
                )

    return matrix


def compute_simulated_tail_dependence_tables(
    simulated_uniforms: pd.DataFrame,
    quantile_levels: list[float],
) -> pd.DataFrame:
    """
    Compute simulated finite-threshold tail dependence for multiple q levels.

    Parameters
    ----------
    simulated_uniforms:
        Simulated uniform observations from copula.
    quantile_levels:
        Threshold levels.

    Returns
    -------
    pd.DataFrame
        Long pairwise simulated tail-dependence table.
    """
    records = []

    for quantile_level in quantile_levels:
        matrix = compute_tail_dependence_from_uniforms(
            pseudo_observations=simulated_uniforms,
            quantile_level=quantile_level,
        )

        for asset in matrix.index:
            for conditioning_asset in matrix.columns:
                records.append(
                    {
                        "asset": asset,
                        "conditioning_asset": conditioning_asset,
                        "quantile_level": quantile_level,
                        "simulated_tail_dependence": matrix.loc[
                            asset,
                            conditioning_asset,
                        ],
                    }
                )

    result = pd.DataFrame(records)

    return result


def compare_empirical_and_copula_tail_dependence(
    empirical_matrix: pd.DataFrame,
    copula_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare empirical and copula-implied tail dependence matrices.

    Parameters
    ----------
    empirical_matrix:
        Empirical tail-dependence matrix.
    copula_matrix:
        Copula-implied tail-dependence matrix.

    Returns
    -------
    pd.DataFrame
        Long comparison table.
    """
    common_rows = empirical_matrix.index.intersection(copula_matrix.index)
    common_columns = empirical_matrix.columns.intersection(copula_matrix.columns)

    records = []

    for asset in common_rows:
        for conditioning_asset in common_columns:
            empirical_value = float(empirical_matrix.loc[asset, conditioning_asset])
            copula_value = float(copula_matrix.loc[asset, conditioning_asset])

            records.append(
                {
                    "asset": asset,
                    "conditioning_asset": conditioning_asset,
                    "empirical_tail_dependence": empirical_value,
                    "copula_tail_dependence": copula_value,
                    "copula_minus_empirical": copula_value - empirical_value,
                    "absolute_difference": abs(copula_value - empirical_value),
                }
            )

    comparison = pd.DataFrame(records)

    comparison = comparison.sort_values(
        "absolute_difference",
        ascending=False,
    ).reset_index(drop=True)

    return comparison