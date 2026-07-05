from __future__ import annotations

import pandas as pd


def compute_stress_activation_index(
    connectedness_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Persistent Connectedness and Stress Activation Index.

    Parameters
    ----------
    connectedness_comparison:
        Dataframe containing calm and stress connectedness columns.

    Returns
    -------
    pd.DataFrame
        Stress activation summary by bank.
    """
    required_columns = {
        "asset",
        "connectedness_calm",
        "connectedness_stress",
        "connectedness_stress_minus_calm",
        "connectedness_stress_to_calm_ratio",
    }

    missing_columns = required_columns - set(connectedness_comparison.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    result = connectedness_comparison.copy()

    result["persistent_connectedness"] = (
        result["connectedness_calm"] + result["connectedness_stress"]
    ) / 2.0

    result["stress_activation_index"] = (
        result["connectedness_stress"] - result["connectedness_calm"]
    )

    result["stress_activation_ratio"] = (
        result["connectedness_stress"] / result["connectedness_calm"]
    )

    persistent_median = result["persistent_connectedness"].median()
    activation_median = result["stress_activation_index"].median()

    result["persistent_connectedness_group"] = result[
        "persistent_connectedness"
    ].apply(lambda value: "high" if value >= persistent_median else "low")

    result["stress_activation_group"] = result[
        "stress_activation_index"
    ].apply(lambda value: "high" if value >= activation_median else "low")

    result["systemic_tail_type"] = result.apply(
        classify_systemic_tail_type,
        axis=1,
    )

    result["rank_persistent_connectedness"] = result[
        "persistent_connectedness"
    ].rank(ascending=False, method="min")

    result["rank_stress_activation"] = result[
        "stress_activation_index"
    ].rank(ascending=False, method="min")

    result["rank_stress_connectedness"] = result[
        "connectedness_stress"
    ].rank(ascending=False, method="min")

    result = result.sort_values(
        ["stress_activation_index", "persistent_connectedness"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return result


def classify_systemic_tail_type(row: pd.Series) -> str:
    """
    Classify banks by persistent connectedness and stress activation.

    Parameters
    ----------
    row:
        Row containing persistent_connectedness_group and stress_activation_group.

    Returns
    -------
    str
        Systemic tail type.
    """
    persistent_group = row["persistent_connectedness_group"]
    activation_group = row["stress_activation_group"]

    if persistent_group == "high" and activation_group == "high":
        return "stress-amplified core"

    if persistent_group == "high" and activation_group == "low":
        return "persistent core"

    if persistent_group == "low" and activation_group == "high":
        return "stress activator"

    return "peripheral"


def compute_pairwise_stress_activation(
    calm_matrix: pd.DataFrame,
    stress_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute pairwise stress activation from calm and stress matrices.

    Parameters
    ----------
    calm_matrix:
        Calm-regime pairwise tail-dependence matrix.
    stress_matrix:
        Stress-regime pairwise tail-dependence matrix.

    Returns
    -------
    pd.DataFrame
        Long pairwise stress-activation table.
    """
    common_rows = calm_matrix.index.intersection(stress_matrix.index)
    common_columns = calm_matrix.columns.intersection(stress_matrix.columns)

    calm = calm_matrix.loc[common_rows, common_columns]
    stress = stress_matrix.loc[common_rows, common_columns]

    records = []

    for asset in common_rows:
        for conditioning_asset in common_columns:
            calm_value = float(calm.loc[asset, conditioning_asset])
            stress_value = float(stress.loc[asset, conditioning_asset])

            records.append(
                {
                    "asset": asset,
                    "conditioning_asset": conditioning_asset,
                    "tail_dependence_calm": calm_value,
                    "tail_dependence_stress": stress_value,
                    "pairwise_stress_activation": stress_value - calm_value,
                }
            )

    result = pd.DataFrame(records)

    result = result.sort_values(
        "pairwise_stress_activation",
        ascending=False,
    ).reset_index(drop=True)

    return result