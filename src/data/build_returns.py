from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_parquet(path: str | Path) -> pd.DataFrame:
    """
    Read a Parquet file into a dataframe.

    Parameters
    ----------
    path:
        Path to the Parquet file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(f"Parquet file is empty: {path}")

    return df


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from a price panel.

    Parameters
    ----------
    prices:
        Wide price panel with dates as index and assets as columns.

    Returns
    -------
    pd.DataFrame
        Wide return panel.
    """
    if (prices <= 0).any().any():
        raise ValueError("Price panel contains non-positive prices.")

    log_prices = np.log(prices)
    returns = log_prices.diff()

    returns = returns.replace([np.inf, -np.inf], np.nan)
    returns = returns.sort_index()

    return returns


def compute_losses(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute losses as negative returns.

    Parameters
    ----------
    returns:
        Wide return panel.

    Returns
    -------
    pd.DataFrame
        Wide loss panel.
    """
    losses = -returns
    losses = losses.sort_index()

    return losses


def drop_missing_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Drop rows containing any missing values.

    Parameters
    ----------
    df:
        Input dataframe.
    name:
        Human-readable dataframe name for reporting.

    Returns
    -------
    pd.DataFrame
        Dataframe with complete rows only.
    """
    before = len(df)
    cleaned = df.dropna(how="any")
    after = len(cleaned)

    print(f"{name}: dropped {before - after} rows with missing values.")
    print(f"{name}: remaining rows = {after}")

    return cleaned


def align_to_reference_index(
    df: pd.DataFrame,
    reference_index: pd.DatetimeIndex,
    name: str,
) -> pd.DataFrame:
    """
    Align a dataframe to a reference DatetimeIndex.

    Parameters
    ----------
    df:
        Dataframe to align.
    reference_index:
        Target index.
    name:
        Human-readable dataframe name for reporting.

    Returns
    -------
    pd.DataFrame
        Aligned dataframe.
    """
    aligned = df.reindex(reference_index)

    print(f"{name}: aligned to reference index with {len(reference_index)} dates.")
    print(f"{name}: missing values after alignment:")
    print(aligned.isna().sum())

    return aligned


def compute_equal_weighted_system_loss(losses: pd.DataFrame) -> pd.Series:
    """
    Compute equal-weighted system loss.

    Parameters
    ----------
    losses:
        Wide loss panel with bank losses.

    Returns
    -------
    pd.Series
        Equal-weighted system loss.
    """
    system_loss = losses.mean(axis=1)
    system_loss.name = "system_loss_equal_weighted"

    return system_loss


def save_parquet(df: pd.DataFrame | pd.Series, path: str | Path) -> None:
    """
    Save a dataframe or series to Parquet.

    Parameters
    ----------
    df:
        Dataframe or series to save.
    path:
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df.to_parquet(path)


def summarize_returns(returns: pd.DataFrame, name: str) -> None:
    """
    Print a compact summary of returns or losses.

    Parameters
    ----------
    returns:
        Return or loss dataframe.
    name:
        Human-readable name.
    """
    print(f"\n{name}")
    print("-" * len(name))
    print(f"Shape: {returns.shape}")
    print(f"Start date: {returns.index.min().date()}")
    print(f"End date: {returns.index.max().date()}")
    print("Mean:")
    print(returns.mean())
    print("Standard deviation:")
    print(returns.std())
    print("Minimum:")
    print(returns.min())
    print("Maximum:")
    print(returns.max())