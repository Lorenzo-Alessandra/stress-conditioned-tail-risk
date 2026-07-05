from __future__ import annotations

import pandas as pd


def compute_empirical_covar_for_asset(
    losses: pd.DataFrame,
    system_loss: pd.Series,
    asset: str,
    quantile_level: float,
    normal_quantile_level: float = 0.50,
) -> dict[str, float | int | str]:
    """
    Compute empirical CoVaR and Delta CoVaR for one asset.

    Distress state:
        L_i > Q_q(L_i)

    Normal state:
        L_i <= Q_0.50(L_i)

    Distress CoVaR:
        Q_q(L_system | L_i > Q_q(L_i))

    Normal CoVaR:
        Q_q(L_system | L_i <= Q_0.50(L_i))

    Delta CoVaR:
        Distress CoVaR - Normal CoVaR

    Parameters
    ----------
    losses:
        Wide dataframe of bank losses.
    system_loss:
        System loss series.
    asset:
        Asset ticker.
    quantile_level:
        CoVaR and distress threshold quantile level.
    normal_quantile_level:
        Quantile level defining normal state. Default is median.

    Returns
    -------
    dict
        CoVaR estimates and supporting counts.
    """
    if asset not in losses.columns:
        raise ValueError(f"Asset not found in losses: {asset}")

    common_index = losses.index.intersection(system_loss.index)

    asset_loss = losses.loc[common_index, asset].dropna()
    system_loss_aligned = system_loss.loc[asset_loss.index].dropna()

    common_index = asset_loss.index.intersection(system_loss_aligned.index)

    asset_loss = asset_loss.loc[common_index]
    system_loss_aligned = system_loss_aligned.loc[common_index]

    if asset_loss.empty:
        raise ValueError(f"No aligned observations for asset: {asset}")

    distress_threshold = float(asset_loss.quantile(quantile_level))
    normal_threshold = float(asset_loss.quantile(normal_quantile_level))

    distress_indicator = asset_loss > distress_threshold
    normal_indicator = asset_loss <= normal_threshold

    distress_system_losses = system_loss_aligned.loc[distress_indicator]
    normal_system_losses = system_loss_aligned.loc[normal_indicator]

    if distress_system_losses.empty:
        distress_covar = float("nan")
    else:
        distress_covar = float(distress_system_losses.quantile(quantile_level))

    if normal_system_losses.empty:
        normal_covar = float("nan")
    else:
        normal_covar = float(normal_system_losses.quantile(quantile_level))

    delta_covar = distress_covar - normal_covar

    return {
        "asset": asset,
        "quantile_level": quantile_level,
        "normal_quantile_level": normal_quantile_level,
        "observations": int(len(asset_loss)),
        "distress_threshold": distress_threshold,
        "normal_threshold": normal_threshold,
        "distress_observations": int(distress_indicator.sum()),
        "normal_observations": int(normal_indicator.sum()),
        "distress_covar": distress_covar,
        "normal_covar": normal_covar,
        "delta_covar": delta_covar,
    }


def compute_empirical_covar(
    losses: pd.DataFrame,
    system_loss: pd.Series | pd.DataFrame,
    quantile_levels: list[float],
    normal_quantile_level: float = 0.50,
) -> pd.DataFrame:
    """
    Compute empirical CoVaR and Delta CoVaR for all assets and quantile levels.

    Parameters
    ----------
    losses:
        Wide dataframe of bank losses.
    system_loss:
        System loss series or dataframe.
    quantile_levels:
        Quantile levels.
    normal_quantile_level:
        Quantile level defining normal state.

    Returns
    -------
    pd.DataFrame
        Empirical CoVaR table.
    """
    if isinstance(system_loss, pd.DataFrame):
        if system_loss.shape[1] != 1:
            raise ValueError("system_loss dataframe must have exactly one column.")
        system_loss_series = system_loss.iloc[:, 0]
    else:
        system_loss_series = system_loss

    records = []

    for quantile_level in quantile_levels:
        for asset in losses.columns:
            record = compute_empirical_covar_for_asset(
                losses=losses,
                system_loss=system_loss_series,
                asset=asset,
                quantile_level=quantile_level,
                normal_quantile_level=normal_quantile_level,
            )
            records.append(record)

    result = pd.DataFrame(records)

    result = result.sort_values(
        ["quantile_level", "delta_covar"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return result


def rank_covar_systemic_contribution(
    covar_table: pd.DataFrame,
    quantile_level: float,
) -> pd.DataFrame:
    """
    Rank banks by Delta CoVaR for a selected quantile level.

    Parameters
    ----------
    covar_table:
        Empirical CoVaR table.
    quantile_level:
        Quantile level to rank.

    Returns
    -------
    pd.DataFrame
        Ranked Delta CoVaR table.
    """
    filtered = covar_table[
        covar_table["quantile_level"] == quantile_level
    ].copy()

    if filtered.empty:
        raise ValueError(f"No CoVaR rows found for q={quantile_level}.")

    filtered = filtered.sort_values("delta_covar", ascending=False).reset_index(
        drop=True
    )

    filtered["rank_delta_covar"] = range(1, len(filtered) + 1)

    return filtered