from __future__ import annotations

import pandas as pd


def extract_mean_es_995(
    es_scoring: pd.DataFrame,
    probability_level: float = 0.995,
) -> pd.DataFrame:
    """
    Extract mean conditional ES at a selected probability level.

    Parameters
    ----------
    es_scoring:
        ES scoring summary table.
    probability_level:
        Probability level.

    Returns
    -------
    pd.DataFrame
        Table with asset and mean_es_995.
    """
    filtered = es_scoring[
        es_scoring["probability_level"] == probability_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No ES scoring rows found for p={probability_level}.")

    result = filtered[["asset", "mean_es"]].copy()
    result = result.rename(columns={"mean_es": "mean_es_995"})

    return result


def extract_tail_connectedness_q95(
    connectedness: pd.DataFrame,
    quantile_level: float = 0.95,
) -> pd.DataFrame:
    """
    Extract average tail connectedness at q=0.95.

    Parameters
    ----------
    connectedness:
        Tail connectedness summary table.
    quantile_level:
        Quantile level.

    Returns
    -------
    pd.DataFrame
        Table with asset and average_tail_connectedness_q95.
    """
    filtered = connectedness[
        connectedness["quantile_level"] == quantile_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No connectedness rows found for q={quantile_level}.")

    result = filtered[
        ["asset", "average_total_tail_connectedness"]
    ].copy()

    result = result.rename(
        columns={
            "average_total_tail_connectedness": "average_tail_connectedness_q95"
        }
    )

    return result


def extract_delta_covar_q95(
    covar: pd.DataFrame,
    quantile_level: float = 0.95,
) -> pd.DataFrame:
    """
    Extract Delta CoVaR at q=0.95.

    Parameters
    ----------
    covar:
        Empirical CoVaR table.
    quantile_level:
        Quantile level.

    Returns
    -------
    pd.DataFrame
        Table with asset and delta_covar_q95.
    """
    filtered = covar[
        covar["quantile_level"] == quantile_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No CoVaR rows found for q={quantile_level}.")

    result = filtered[["asset", "delta_covar"]].copy()
    result = result.rename(columns={"delta_covar": "delta_covar_q95"})

    return result


def extract_network_centrality_050(
    network_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Extract network degree and weighted degree from the 0.50-threshold network.

    Parameters
    ----------
    network_summary:
        Tail-dependence network summary table.

    Returns
    -------
    pd.DataFrame
        Table with asset, degree, and weighted degree.
    """
    required_columns = {"asset", "degree", "weighted_degree"}
    missing_columns = required_columns - set(network_summary.columns)

    if missing_columns:
        raise ValueError(f"Missing network columns: {missing_columns}")

    result = network_summary[
        ["asset", "degree", "weighted_degree"]
    ].copy()

    result = result.rename(
        columns={
            "degree": "network_degree_050",
            "weighted_degree": "network_weighted_degree_050",
        }
    )

    return result


def assign_systemic_risk_group(average_rank: float) -> str:
    """
    Assign a descriptive systemic-risk group based on average rank.

    Parameters
    ----------
    average_rank:
        Equal-weighted average rank.

    Returns
    -------
    str
        Risk group label.
    """
    if average_rank <= 3.0:
        return "high"
    if average_rank <= 5.5:
        return "medium"
    return "lower"


def build_systemic_risk_summary(
    es_scoring_regime: pd.DataFrame,
    tail_connectedness: pd.DataFrame,
    covar_empirical: pd.DataFrame,
    network_summary_050: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build integrated systemic-risk summary table.

    Parameters
    ----------
    es_scoring_regime:
        Regime-specific ES scoring summary.
    tail_connectedness:
        Average tail connectedness summary.
    covar_empirical:
        Empirical CoVaR table.
    network_summary_050:
        Tail-dependence network summary using 0.50 threshold.

    Returns
    -------
    pd.DataFrame
        Integrated systemic-risk summary.
    """
    mean_es = extract_mean_es_995(es_scoring_regime)
    connectedness = extract_tail_connectedness_q95(tail_connectedness)
    delta_covar = extract_delta_covar_q95(covar_empirical)
    network = extract_network_centrality_050(network_summary_050)

    summary = mean_es.merge(
        connectedness,
        on="asset",
        how="inner",
    )

    summary = summary.merge(
        delta_covar,
        on="asset",
        how="inner",
    )

    summary = summary.merge(
        network,
        on="asset",
        how="inner",
    )

    summary["rank_mean_es_995"] = summary["mean_es_995"].rank(
        ascending=False,
        method="min",
    )

    summary["rank_tail_connectedness_q95"] = summary[
        "average_tail_connectedness_q95"
    ].rank(
        ascending=False,
        method="min",
    )

    summary["rank_delta_covar_q95"] = summary["delta_covar_q95"].rank(
        ascending=False,
        method="min",
    )

    summary["rank_network_weighted_degree_050"] = summary[
        "network_weighted_degree_050"
    ].rank(
        ascending=False,
        method="min",
    )

    rank_columns = [
        "rank_mean_es_995",
        "rank_tail_connectedness_q95",
        "rank_delta_covar_q95",
        "rank_network_weighted_degree_050",
    ]

    summary["average_rank"] = summary[rank_columns].mean(axis=1)

    summary["systemic_risk_group"] = summary["average_rank"].apply(
        assign_systemic_risk_group
    )

    summary = summary.sort_values("average_rank", ascending=True).reset_index(
        drop=True
    )

    return summary