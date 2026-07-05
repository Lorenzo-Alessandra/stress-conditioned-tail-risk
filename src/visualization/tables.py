from __future__ import annotations

from pathlib import Path

import pandas as pd


def make_descriptive_stats_table(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Create a descriptive statistics table for a return panel.

    Parameters
    ----------
    returns:
        Wide dataframe of returns with dates as index and assets as columns.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics table with one row per asset.
    """
    table = pd.DataFrame(index=returns.columns)

    table["mean"] = returns.mean()
    table["std"] = returns.std()
    table["min"] = returns.min()
    table["q01"] = returns.quantile(0.01)
    table["q05"] = returns.quantile(0.05)
    table["median"] = returns.median()
    table["q95"] = returns.quantile(0.95)
    table["q99"] = returns.quantile(0.99)
    table["max"] = returns.max()
    table["skewness"] = returns.skew()
    table["kurtosis"] = returns.kurtosis()
    table["observations"] = returns.count()

    return table


def format_descriptive_stats_table(table: pd.DataFrame) -> pd.DataFrame:
    """
    Format descriptive statistics for readable output.

    Parameters
    ----------
    table:
        Raw descriptive statistics table.

    Returns
    -------
    pd.DataFrame
        Formatted descriptive statistics table.
    """
    formatted = table.copy()

    numeric_columns = [
        "mean",
        "std",
        "min",
        "q01",
        "q05",
        "median",
        "q95",
        "q99",
        "max",
        "skewness",
        "kurtosis",
    ]

    for column in numeric_columns:
        formatted[column] = formatted[column].round(4)

    formatted["observations"] = formatted["observations"].astype(int)

    return formatted


def save_table_csv(table: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)


def save_table_latex(
    table: pd.DataFrame,
    path: str | Path,
    caption: str,
    label: str,
) -> None:
    """
    Save a table as a LaTeX file.

    Parameters
    ----------
    table:
        Table to save.
    path:
        Output LaTeX path.
    caption:
        LaTeX table caption.
    label:
        LaTeX table label.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    latex = table.to_latex(
        index=True,
        caption=caption,
        label=label,
        float_format="%.4f",
    )

    path.write_text(latex, encoding="utf-8")

    

def make_extreme_returns_audit_table(
    returns: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Create a table of the largest positive and negative returns for each asset.

    Parameters
    ----------
    returns:
        Wide dataframe of daily returns with dates as index and assets as columns.
    top_n:
        Number of largest positive and negative observations to report per asset.

    Returns
    -------
    pd.DataFrame
        Audit table containing extreme return observations.
    """
    records = []

    for asset in returns.columns:
        series = returns[asset].dropna()

        largest_losses = series.nsmallest(top_n)
        largest_gains = series.nlargest(top_n)

        for date, value in largest_losses.items():
            records.append(
                {
                    "asset": asset,
                    "date": date,
                    "type": "largest_negative_return",
                    "return": value,
                    "loss": -value,
                }
            )

        for date, value in largest_gains.items():
            records.append(
                {
                    "asset": asset,
                    "date": date,
                    "type": "largest_positive_return",
                    "return": value,
                    "loss": -value,
                }
            )

    audit_table = pd.DataFrame(records)
    audit_table = audit_table.sort_values(["asset", "type", "return"])

    return audit_table