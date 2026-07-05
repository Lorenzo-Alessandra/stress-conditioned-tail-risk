from __future__ import annotations

import pandas as pd


def merge_baseline_and_regime_risk(
    baseline_risk: pd.DataFrame,
    regime_risk: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Merge baseline and regime-specific conditional risk forecasts.

    Parameters
    ----------
    baseline_risk:
        Long baseline risk dataframe.
    regime_risk:
        Long regime-specific risk dataframe.
    value_column:
        Risk column name, either conditional_var or conditional_es.

    Returns
    -------
    pd.DataFrame
        Merged dataframe with baseline and regime-specific risk forecasts.
    """
    required_baseline = {"date", "asset", "probability_level", value_column}
    required_regime = {
        "date",
        "asset",
        "probability_level",
        "stress_regime",
        value_column,
    }

    missing_baseline = required_baseline - set(baseline_risk.columns)
    missing_regime = required_regime - set(regime_risk.columns)

    if missing_baseline:
        raise ValueError(f"Missing baseline columns: {missing_baseline}")

    if missing_regime:
        raise ValueError(f"Missing regime columns: {missing_regime}")

    baseline = baseline_risk[
        ["date", "asset", "probability_level", value_column]
    ].copy()

    regime = regime_risk[
        ["date", "asset", "probability_level", "stress_regime", value_column]
    ].copy()

    baseline = baseline.rename(columns={value_column: f"{value_column}_baseline"})
    regime = regime.rename(columns={value_column: f"{value_column}_regime"})

    merged = baseline.merge(
        regime,
        on=["date", "asset", "probability_level"],
        how="inner",
    )

    merged[f"{value_column}_difference"] = (
        merged[f"{value_column}_regime"]
        - merged[f"{value_column}_baseline"]
    )

    merged[f"{value_column}_ratio"] = (
        merged[f"{value_column}_regime"]
        / merged[f"{value_column}_baseline"]
    )

    return merged


def summarize_baseline_regime_comparison(
    merged: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """
    Summarize baseline vs regime-specific risk forecasts.

    Parameters
    ----------
    merged:
        Merged baseline-regime risk dataframe.
    value_column:
        Risk column name, either conditional_var or conditional_es.

    Returns
    -------
    pd.DataFrame
        Summary by asset, probability level, and stress regime.
    """
    baseline_col = f"{value_column}_baseline"
    regime_col = f"{value_column}_regime"
    difference_col = f"{value_column}_difference"
    ratio_col = f"{value_column}_ratio"

    grouped = merged.groupby(["asset", "probability_level", "stress_regime"])

    summary = grouped.agg(
        observations=(regime_col, "count"),
        baseline_mean=(baseline_col, "mean"),
        regime_mean=(regime_col, "mean"),
        mean_difference=(difference_col, "mean"),
        mean_ratio=(ratio_col, "mean"),
        baseline_median=(baseline_col, "median"),
        regime_median=(regime_col, "median"),
        median_difference=(difference_col, "median"),
        min_difference=(difference_col, "min"),
        max_difference=(difference_col, "max"),
    ).reset_index()

    summary = summary.sort_values(
        ["asset", "probability_level", "stress_regime"]
    ).reset_index(drop=True)

    return summary


def combine_var_es_comparisons(
    var_summary: pd.DataFrame,
    es_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine VaR and ES comparison summaries into one table.

    Parameters
    ----------
    var_summary:
        VaR comparison summary.
    es_summary:
        ES comparison summary.

    Returns
    -------
    pd.DataFrame
        Combined comparison table.
    """
    var = var_summary.copy()
    es = es_summary.copy()

    var["risk_measure"] = "VaR"
    es["risk_measure"] = "ES"

    combined = pd.concat([var, es], axis=0, ignore_index=True)

    combined = combined[
        [
            "risk_measure",
            "asset",
            "probability_level",
            "stress_regime",
            "observations",
            "baseline_mean",
            "regime_mean",
            "mean_difference",
            "mean_ratio",
            "baseline_median",
            "regime_median",
            "median_difference",
            "min_difference",
            "max_difference",
        ]
    ]

    combined = combined.sort_values(
        ["risk_measure", "asset", "probability_level", "stress_regime"]
    ).reset_index(drop=True)

    return combined