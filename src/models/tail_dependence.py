from __future__ import annotations

import itertools

import pandas as pd


def compute_loss_thresholds(
    losses: pd.DataFrame,
    quantile_level: float,
) -> pd.Series:
    """
    Compute empirical loss thresholds for each asset.

    Parameters
    ----------
    losses:
        Wide dataframe of losses.
    quantile_level:
        Tail threshold quantile.

    Returns
    -------
    pd.Series
        Asset-indexed empirical loss thresholds.
    """
    if losses.empty:
        raise ValueError("Loss dataframe is empty.")

    return losses.quantile(quantile_level)


def compute_tail_indicators(
    losses: pd.DataFrame,
    thresholds: pd.Series,
) -> pd.DataFrame:
    """
    Compute binary tail exceedance indicators.

    Parameters
    ----------
    losses:
        Wide dataframe of losses.
    thresholds:
        Asset-indexed loss thresholds.

    Returns
    -------
    pd.DataFrame
        Binary dataframe equal to 1 when loss exceeds threshold.
    """
    missing_assets = [
        asset for asset in losses.columns if asset not in thresholds.index
    ]

    if missing_assets:
        raise ValueError(f"Missing thresholds for assets: {missing_assets}")

    indicators = losses.gt(thresholds, axis=1).astype(int)

    return indicators


def compute_pairwise_tail_dependence_for_level(
    losses: pd.DataFrame,
    quantile_level: float,
) -> pd.DataFrame:
    """
    Compute pairwise directional tail dependence for one threshold level.

    Tail dependence is estimated as:
        P(asset_i is extreme | conditioning_asset_j is extreme)

    Parameters
    ----------
    losses:
        Wide dataframe of losses.
    quantile_level:
        Tail threshold quantile.

    Returns
    -------
    pd.DataFrame
        Pairwise directional tail-dependence table.
    """
    thresholds = compute_loss_thresholds(
        losses=losses,
        quantile_level=quantile_level,
    )

    indicators = compute_tail_indicators(
        losses=losses,
        thresholds=thresholds,
    )

    records = []

    assets = list(losses.columns)

    for asset_i, conditioning_asset_j in itertools.product(assets, assets):
        indicator_i = indicators[asset_i]
        indicator_j = indicators[conditioning_asset_j]

        conditioning_count = int(indicator_j.sum())
        joint_count = int(((indicator_i == 1) & (indicator_j == 1)).sum())

        if conditioning_count == 0:
            tail_dependence = float("nan")
        else:
            tail_dependence = joint_count / conditioning_count

        records.append(
            {
                "asset": asset_i,
                "conditioning_asset": conditioning_asset_j,
                "quantile_level": quantile_level,
                "threshold_asset": thresholds[asset_i],
                "threshold_conditioning_asset": thresholds[conditioning_asset_j],
                "observations": int(len(losses)),
                "asset_exceedance_count": int(indicator_i.sum()),
                "conditioning_exceedance_count": conditioning_count,
                "joint_exceedance_count": joint_count,
                "tail_dependence": tail_dependence,
            }
        )

    result = pd.DataFrame(records)

    result = result.sort_values(
        ["quantile_level", "conditioning_asset", "asset"]
    ).reset_index(drop=True)

    return result


def compute_pairwise_tail_dependence(
    losses: pd.DataFrame,
    quantile_levels: list[float],
) -> pd.DataFrame:
    """
    Compute pairwise directional tail dependence for multiple threshold levels.

    Parameters
    ----------
    losses:
        Wide dataframe of losses.
    quantile_levels:
        List of threshold quantile levels.

    Returns
    -------
    pd.DataFrame
        Pairwise directional tail-dependence table.
    """
    tables = []

    for quantile_level in quantile_levels:
        table = compute_pairwise_tail_dependence_for_level(
            losses=losses,
            quantile_level=quantile_level,
        )
        tables.append(table)

    result = pd.concat(tables, axis=0, ignore_index=True)

    return result


def build_tail_dependence_matrix(
    pairwise_tail_dependence: pd.DataFrame,
    quantile_level: float,
) -> pd.DataFrame:
    """
    Build a square tail-dependence matrix for one threshold level.

    Rows are affected assets.
    Columns are conditioning assets.

    Entry (i, j) is:
        P(asset_i extreme | asset_j extreme)

    Parameters
    ----------
    pairwise_tail_dependence:
        Pairwise directional tail-dependence table.
    quantile_level:
        Quantile level to select.

    Returns
    -------
    pd.DataFrame
        Tail-dependence matrix.
    """
    filtered = pairwise_tail_dependence[
        pairwise_tail_dependence["quantile_level"] == quantile_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No tail-dependence rows for q={quantile_level}.")

    matrix = filtered.pivot(
        index="asset",
        columns="conditioning_asset",
        values="tail_dependence",
    )

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    return matrix


def summarize_average_tail_connectedness(
    pairwise_tail_dependence: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize average incoming and outgoing tail connectedness.

    For directional dependence:
        incoming for asset i:
            average P(asset_i extreme | asset_j extreme), j != i

        outgoing for asset j:
            average P(asset_i extreme | asset_j extreme), i != j

    Parameters
    ----------
    pairwise_tail_dependence:
        Pairwise directional tail-dependence table.

    Returns
    -------
    pd.DataFrame
        Average incoming and outgoing tail connectedness by asset and q level.
    """
    df = pairwise_tail_dependence.copy()

    df = df[df["asset"] != df["conditioning_asset"]].copy()

    incoming = (
        df.groupby(["asset", "quantile_level"])["tail_dependence"]
        .mean()
        .rename("average_incoming_tail_dependence")
        .reset_index()
    )

    outgoing = (
        df.groupby(["conditioning_asset", "quantile_level"])["tail_dependence"]
        .mean()
        .rename("average_outgoing_tail_dependence")
        .reset_index()
        .rename(columns={"conditioning_asset": "asset"})
    )

    summary = incoming.merge(
        outgoing,
        on=["asset", "quantile_level"],
        how="inner",
    )

    summary["average_total_tail_connectedness"] = (
        summary["average_incoming_tail_dependence"]
        + summary["average_outgoing_tail_dependence"]
    ) / 2.0

    summary = summary.sort_values(
        ["quantile_level", "average_total_tail_connectedness"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return summary


def compute_pairwise_tail_dependence_by_regime(
    losses: pd.DataFrame,
    stress_regime: pd.DataFrame,
    quantile_levels: list[float],
) -> pd.DataFrame:
    """
    Compute pairwise tail dependence separately for calm and stress regimes.

    Parameters
    ----------
    losses:
        Wide dataframe of bank losses.
    stress_regime:
        Dataframe containing stress_regime indexed by date.
    quantile_levels:
        List of threshold quantile levels.

    Returns
    -------
    pd.DataFrame
        Regime-specific pairwise tail-dependence table.
    """
    if "stress_regime" not in stress_regime.columns:
        raise ValueError("stress_regime must contain a 'stress_regime' column.")

    common_index = losses.index.intersection(stress_regime.index)

    losses_aligned = losses.loc[common_index].copy()
    regime_aligned = stress_regime.loc[common_index, "stress_regime"].astype(int)

    records = []

    for regime_value in [0, 1]:
        regime_label = "stress" if regime_value == 1 else "calm"

        regime_dates = regime_aligned[regime_aligned == regime_value].index
        regime_losses = losses_aligned.loc[regime_dates].dropna(how="any")

        if regime_losses.empty:
            raise ValueError(f"No loss observations for regime {regime_label}.")

        for quantile_level in quantile_levels:
            table = compute_pairwise_tail_dependence_for_level(
                losses=regime_losses,
                quantile_level=quantile_level,
            )

            table["stress_regime"] = regime_value
            table["regime_label"] = regime_label

            records.append(table)

    result = pd.concat(records, axis=0, ignore_index=True)

    result = result.sort_values(
        ["quantile_level", "stress_regime", "conditioning_asset", "asset"]
    ).reset_index(drop=True)

    return result


def build_regime_tail_dependence_matrix(
    pairwise_tail_dependence_regime: pd.DataFrame,
    quantile_level: float,
    stress_regime: int,
) -> pd.DataFrame:
    """
    Build a tail-dependence matrix for one regime and quantile level.

    Parameters
    ----------
    pairwise_tail_dependence_regime:
        Regime-specific pairwise tail-dependence table.
    quantile_level:
        Quantile level.
    stress_regime:
        Regime value, 0 for calm and 1 for stress.

    Returns
    -------
    pd.DataFrame
        Regime-specific tail-dependence matrix.
    """
    filtered = pairwise_tail_dependence_regime[
        (pairwise_tail_dependence_regime["quantile_level"] == quantile_level)
        & (pairwise_tail_dependence_regime["stress_regime"] == stress_regime)
    ].copy()

    if filtered.empty:
        raise ValueError(
            f"No rows found for q={quantile_level}, stress_regime={stress_regime}."
        )

    matrix = filtered.pivot(
        index="asset",
        columns="conditioning_asset",
        values="tail_dependence",
    )

    matrix = matrix.sort_index(axis=0).sort_index(axis=1)

    return matrix


def compute_stress_minus_calm_matrix(
    calm_matrix: pd.DataFrame,
    stress_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute stress-minus-calm tail-dependence matrix.

    Parameters
    ----------
    calm_matrix:
        Calm-regime tail-dependence matrix.
    stress_matrix:
        Stress-regime tail-dependence matrix.

    Returns
    -------
    pd.DataFrame
        Difference matrix.
    """
    common_rows = calm_matrix.index.intersection(stress_matrix.index)
    common_columns = calm_matrix.columns.intersection(stress_matrix.columns)

    calm = calm_matrix.loc[common_rows, common_columns]
    stress = stress_matrix.loc[common_rows, common_columns]

    return stress - calm


def summarize_regime_tail_connectedness(
    pairwise_tail_dependence_regime: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize average incoming, outgoing, and total connectedness by regime.

    Parameters
    ----------
    pairwise_tail_dependence_regime:
        Regime-specific pairwise tail-dependence table.

    Returns
    -------
    pd.DataFrame
        Connectedness summary by asset, quantile level, and regime.
    """
    df = pairwise_tail_dependence_regime.copy()
    df = df[df["asset"] != df["conditioning_asset"]].copy()

    incoming = (
        df.groupby(["asset", "quantile_level", "stress_regime", "regime_label"])[
            "tail_dependence"
        ]
        .mean()
        .rename("average_incoming_tail_dependence")
        .reset_index()
    )

    outgoing = (
        df.groupby(
            ["conditioning_asset", "quantile_level", "stress_regime", "regime_label"]
        )["tail_dependence"]
        .mean()
        .rename("average_outgoing_tail_dependence")
        .reset_index()
        .rename(columns={"conditioning_asset": "asset"})
    )

    summary = incoming.merge(
        outgoing,
        on=["asset", "quantile_level", "stress_regime", "regime_label"],
        how="inner",
    )

    summary["average_total_tail_connectedness"] = (
        summary["average_incoming_tail_dependence"]
        + summary["average_outgoing_tail_dependence"]
    ) / 2.0

    summary = summary.sort_values(
        ["quantile_level", "stress_regime", "average_total_tail_connectedness"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    return summary


def compare_calm_stress_connectedness(
    regime_connectedness: pd.DataFrame,
    quantile_level: float,
) -> pd.DataFrame:
    """
    Compare calm and stress average connectedness for one quantile level.

    Parameters
    ----------
    regime_connectedness:
        Regime-specific connectedness summary.
    quantile_level:
        Quantile level.

    Returns
    -------
    pd.DataFrame
        Calm-stress connectedness comparison.
    """
    filtered = regime_connectedness[
        regime_connectedness["quantile_level"] == quantile_level
    ].copy()

    calm = filtered[filtered["stress_regime"] == 0].set_index("asset")
    stress = filtered[filtered["stress_regime"] == 1].set_index("asset")

    comparison = pd.DataFrame(index=calm.index)

    comparison["connectedness_calm"] = calm["average_total_tail_connectedness"]
    comparison["connectedness_stress"] = stress["average_total_tail_connectedness"]

    comparison["connectedness_stress_minus_calm"] = (
        comparison["connectedness_stress"] - comparison["connectedness_calm"]
    )

    comparison["connectedness_stress_to_calm_ratio"] = (
        comparison["connectedness_stress"] / comparison["connectedness_calm"]
    )

    comparison = comparison.reset_index()

    comparison = comparison.sort_values(
        "connectedness_stress_minus_calm",
        ascending=False,
    ).reset_index(drop=True)

    return comparison