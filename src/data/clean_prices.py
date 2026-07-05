from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def read_yahoo_price_csv(path: str | Path) -> pd.DataFrame:
    """
    Read a raw Yahoo Finance CSV saved from yfinance with two header rows.

    Parameters
    ----------
    path:
        Path to the raw Yahoo CSV.

    Returns
    -------
    pd.DataFrame
        Dataframe with a DatetimeIndex and MultiIndex columns:
        level 0 = price field, level 1 = ticker.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Yahoo price file not found: {path}")

    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(f"Yahoo price file is empty: {path}")

    df.index.name = "date"

    return df


def select_price_field(
    raw_prices: pd.DataFrame,
    field_preference: Sequence[str],
) -> pd.DataFrame:
    """
    Select a clean price panel from raw Yahoo prices.

    The function first tries the preferred price field, usually 'Adj Close'.
    If a ticker has missing adjusted close data but close data are available,
    it falls back to 'Close' for that ticker.

    Parameters
    ----------
    raw_prices:
        Raw Yahoo dataframe with MultiIndex columns.
    field_preference:
        Ordered list of price fields to prefer.

    Returns
    -------
    pd.DataFrame
        Wide price panel with dates as index and tickers as columns.
    """
    if not isinstance(raw_prices.columns, pd.MultiIndex):
        raise ValueError("Expected raw_prices to have MultiIndex columns.")

    available_fields = list(raw_prices.columns.get_level_values(0).unique())
    available_tickers = list(raw_prices.columns.get_level_values(1).unique())

    selected_columns: dict[str, pd.Series] = {}

    for ticker in available_tickers:
        chosen_series = None
        chosen_field = None

        for field in field_preference:
            if field not in available_fields:
                continue

            if (field, ticker) not in raw_prices.columns:
                continue

            candidate = raw_prices[(field, ticker)]

            if candidate.notna().sum() == 0:
                continue

            chosen_series = candidate
            chosen_field = field
            break

        if chosen_series is None:
            raise ValueError(
                f"No usable price field found for ticker {ticker}. "
                f"Tried fields: {list(field_preference)}"
            )

        selected_columns[ticker] = chosen_series.rename(ticker)

        if chosen_field != field_preference[0]:
            print(
                f"Warning: using {chosen_field} for {ticker} "
                f"because {field_preference[0]} was unavailable."
            )

    prices = pd.DataFrame(selected_columns)
    prices = prices.sort_index()

    return prices


def clean_price_panel(
    raw_prices: pd.DataFrame,
    field_preference: Sequence[str],
    drop_all_missing_rows: bool = True,
) -> pd.DataFrame:
    """
    Clean a raw Yahoo price dataframe into a usable wide price panel.

    Parameters
    ----------
    raw_prices:
        Raw Yahoo dataframe with MultiIndex columns.
    field_preference:
        Ordered list of price fields to prefer.
    drop_all_missing_rows:
        Whether to drop dates where all tickers are missing.

    Returns
    -------
    pd.DataFrame
        Clean wide price panel.
    """
    prices = select_price_field(
        raw_prices=raw_prices,
        field_preference=field_preference,
    )

    prices = prices.apply(pd.to_numeric, errors="coerce")

    if drop_all_missing_rows:
        prices = prices.dropna(how="all")

    prices = prices[~prices.index.duplicated(keep="first")]
    prices = prices.sort_index()

    return prices


def read_stress_csv(path: str | Path) -> pd.DataFrame:
    """
    Read raw stress-variable CSV.

    Parameters
    ----------
    path:
        Path to raw stress-variable CSV.

    Returns
    -------
    pd.DataFrame
        Stress-variable dataframe with DatetimeIndex.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Stress-variable file not found: {path}")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    if df.empty:
        raise ValueError(f"Stress-variable file is empty: {path}")

    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.sort_index()

    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save dataframe to Parquet.

    Parameters
    ----------
    df:
        Dataframe to save.
    path:
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def summarize_panel(df: pd.DataFrame, name: str) -> None:
    """
    Print a compact summary of a dataframe.

    Parameters
    ----------
    df:
        Dataframe to summarize.
    name:
        Human-readable name for the dataframe.
    """
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Shape: {df.shape}")
    print(f"Start date: {df.index.min().date()}")
    print(f"End date: {df.index.max().date()}")
    print("Missing values by column:")
    print(df.isna().sum())