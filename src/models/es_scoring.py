from __future__ import annotations

import pandas as pd


def merge_losses_var_es(
    losses: pd.DataFrame,
    conditional_var_long: pd.DataFrame,
    conditional_es_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge actual losses, conditional VaR, and conditional ES forecasts.

    Parameters
    ----------
    losses:
        Wide dataframe of actual losses.
    conditional_var_long:
        Long dataframe with conditional VaR forecasts.
    conditional_es_long:
        Long dataframe with conditional ES forecasts.

    Returns
    -------
    pd.DataFrame
        Long dataframe containing loss, VaR, ES, and violation indicator.
    """
    losses_long = losses.stack().reset_index()
    losses_long.columns = ["date", "asset", "loss"]

    merged = conditional_var_long.merge(
        conditional_es_long,
        on=["date", "asset", "probability_level"],
        how="inner",
    )

    merged = merged.merge(
        losses_long,
        on=["date", "asset"],
        how="inner",
    )

    merged["violation"] = merged["loss"] > merged["conditional_var"]

    merged = merged.sort_values(
        ["asset", "probability_level", "date"]
    ).reset_index(drop=True)

    return merged


def compute_simple_es_score(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute a simple joint VaR/ES tail score.

    The score is:
        S_t = (ES_t - VaR_t)
              + (1 / alpha) * (L_t - VaR_t) * I{L_t > VaR_t}

    where alpha = 1 - probability_level.

    Parameters
    ----------
    merged:
        Dataframe containing loss, conditional_var, conditional_es,
        and probability_level.

    Returns
    -------
    pd.DataFrame
        Dataframe with daily ES scores.
    """
    required_columns = {
        "loss",
        "conditional_var",
        "conditional_es",
        "probability_level",
        "violation",
    }

    missing_columns = required_columns - set(merged.columns)

    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    scored = merged.copy()

    scored["tail_probability"] = 1.0 - scored["probability_level"]

    scored["simple_es_score"] = (
        scored["conditional_es"]
        - scored["conditional_var"]
        + (
            (scored["loss"] - scored["conditional_var"])
            * scored["violation"].astype(float)
            / scored["tail_probability"]
        )
    )

    return scored


def summarize_es_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize ES scores by asset and probability level.

    Parameters
    ----------
    scored:
        Dataframe containing daily ES scores.

    Returns
    -------
    pd.DataFrame
        Summary table of ES scores.
    """
    grouped = scored.groupby(["asset", "probability_level"])

    summary = grouped.agg(
        observations=("simple_es_score", "count"),
        mean_score=("simple_es_score", "mean"),
        median_score=("simple_es_score", "median"),
        std_score=("simple_es_score", "std"),
        min_score=("simple_es_score", "min"),
        max_score=("simple_es_score", "max"),
        violation_count=("violation", "sum"),
        mean_loss=("loss", "mean"),
        mean_var=("conditional_var", "mean"),
        mean_es=("conditional_es", "mean"),
    ).reset_index()

    summary["violation_rate"] = (
        summary["violation_count"] / summary["observations"]
    )

    summary = summary.sort_values(["asset", "probability_level"]).reset_index(
        drop=True
    )

    return summary