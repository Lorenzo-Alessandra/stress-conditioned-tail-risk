from __future__ import annotations

import pandas as pd


def compare_var_backtests(
    baseline_backtest: pd.DataFrame,
    regime_backtest: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare baseline and regime-specific VaR backtesting results.

    Parameters
    ----------
    baseline_backtest:
        Baseline VaR backtesting table.
    regime_backtest:
        Regime-specific VaR backtesting table.

    Returns
    -------
    pd.DataFrame
        Merged VaR backtesting comparison table.
    """
    key_columns = ["asset", "probability_level"]

    selected_columns = [
        "asset",
        "probability_level",
        "observations",
        "expected_violations",
        "violation_count",
        "violation_rate",
        "kupiec_p_value",
        "independence_p_value",
        "conditional_coverage_p_value",
    ]

    baseline = baseline_backtest[selected_columns].copy()
    regime = regime_backtest[selected_columns].copy()

    baseline = baseline.rename(
        columns={
            "observations": "baseline_observations",
            "expected_violations": "baseline_expected_violations",
            "violation_count": "baseline_violation_count",
            "violation_rate": "baseline_violation_rate",
            "kupiec_p_value": "baseline_kupiec_p_value",
            "independence_p_value": "baseline_independence_p_value",
            "conditional_coverage_p_value": "baseline_conditional_coverage_p_value",
        }
    )

    regime = regime.rename(
        columns={
            "observations": "regime_observations",
            "expected_violations": "regime_expected_violations",
            "violation_count": "regime_violation_count",
            "violation_rate": "regime_violation_rate",
            "kupiec_p_value": "regime_kupiec_p_value",
            "independence_p_value": "regime_independence_p_value",
            "conditional_coverage_p_value": "regime_conditional_coverage_p_value",
        }
    )

    comparison = baseline.merge(
        regime,
        on=key_columns,
        how="inner",
    )

    return comparison


def compare_es_scores(
    baseline_es_scores: pd.DataFrame,
    regime_es_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare baseline and regime-specific ES scoring results.

    Parameters
    ----------
    baseline_es_scores:
        Baseline ES scoring summary.
    regime_es_scores:
        Regime-specific ES scoring summary.

    Returns
    -------
    pd.DataFrame
        Merged ES scoring comparison table.
    """
    key_columns = ["asset", "probability_level"]

    selected_columns = [
        "asset",
        "probability_level",
        "mean_score",
        "median_score",
        "std_score",
        "violation_count",
        "violation_rate",
        "mean_var",
        "mean_es",
    ]

    baseline = baseline_es_scores[selected_columns].copy()
    regime = regime_es_scores[selected_columns].copy()

    baseline = baseline.rename(
        columns={
            "mean_score": "baseline_mean_es_score",
            "median_score": "baseline_median_es_score",
            "std_score": "baseline_std_es_score",
            "violation_count": "baseline_es_violation_count",
            "violation_rate": "baseline_es_violation_rate",
            "mean_var": "baseline_mean_var",
            "mean_es": "baseline_mean_es",
        }
    )

    regime = regime.rename(
        columns={
            "mean_score": "regime_mean_es_score",
            "median_score": "regime_median_es_score",
            "std_score": "regime_std_es_score",
            "violation_count": "regime_es_violation_count",
            "violation_rate": "regime_es_violation_rate",
            "mean_var": "regime_mean_var",
            "mean_es": "regime_mean_es",
        }
    )

    comparison = baseline.merge(
        regime,
        on=key_columns,
        how="inner",
    )

    comparison["es_score_difference_regime_minus_baseline"] = (
        comparison["regime_mean_es_score"]
        - comparison["baseline_mean_es_score"]
    )

    comparison["es_score_improvement_percent"] = (
        100.0
        * (
            comparison["baseline_mean_es_score"]
            - comparison["regime_mean_es_score"]
        )
        / comparison["baseline_mean_es_score"]
    )

    comparison["regime_score_improves"] = (
        comparison["regime_mean_es_score"]
        < comparison["baseline_mean_es_score"]
    )

    return comparison


def build_model_evaluation_comparison(
    baseline_backtest: pd.DataFrame,
    regime_backtest: pd.DataFrame,
    baseline_es_scores: pd.DataFrame,
    regime_es_scores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build combined model evaluation comparison table.

    Parameters
    ----------
    baseline_backtest:
        Baseline VaR backtesting results.
    regime_backtest:
        Regime-specific VaR backtesting results.
    baseline_es_scores:
        Baseline ES scoring summary.
    regime_es_scores:
        Regime-specific ES scoring summary.

    Returns
    -------
    pd.DataFrame
        Combined model comparison table.
    """
    var_comparison = compare_var_backtests(
        baseline_backtest=baseline_backtest,
        regime_backtest=regime_backtest,
    )

    es_comparison = compare_es_scores(
        baseline_es_scores=baseline_es_scores,
        regime_es_scores=regime_es_scores,
    )

    comparison = var_comparison.merge(
        es_comparison,
        on=["asset", "probability_level"],
        how="inner",
    )

    comparison["baseline_passes_kupiec_5pct"] = (
        comparison["baseline_kupiec_p_value"] >= 0.05
    )
    comparison["regime_passes_kupiec_5pct"] = (
        comparison["regime_kupiec_p_value"] >= 0.05
    )

    comparison["baseline_passes_independence_5pct"] = (
        comparison["baseline_independence_p_value"] >= 0.05
    )
    comparison["regime_passes_independence_5pct"] = (
        comparison["regime_independence_p_value"] >= 0.05
    )

    comparison["baseline_passes_conditional_coverage_5pct"] = (
        comparison["baseline_conditional_coverage_p_value"] >= 0.05
    )
    comparison["regime_passes_conditional_coverage_5pct"] = (
        comparison["regime_conditional_coverage_p_value"] >= 0.05
    )

    comparison = comparison.sort_values(
        ["asset", "probability_level"]
    ).reset_index(drop=True)

    return comparison


def build_compact_model_evaluation_table(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a compact thesis-friendly model evaluation table.

    Parameters
    ----------
    comparison:
        Full model evaluation comparison table.

    Returns
    -------
    pd.DataFrame
        Compact table for reporting.
    """
    compact_columns = [
        "asset",
        "probability_level",
        "baseline_violation_rate",
        "regime_violation_rate",
        "baseline_kupiec_p_value",
        "regime_kupiec_p_value",
        "baseline_independence_p_value",
        "regime_independence_p_value",
        "baseline_conditional_coverage_p_value",
        "regime_conditional_coverage_p_value",
        "baseline_mean_es_score",
        "regime_mean_es_score",
        "es_score_difference_regime_minus_baseline",
        "es_score_improvement_percent",
        "regime_score_improves",
    ]

    return comparison[compact_columns].copy()