from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_csv_with_dates(path: str | Path, date_column: str = "date") -> pd.DataFrame:
    """
    Read a CSV file and parse one column as dates.

    Parameters
    ----------
    path:
        Path to the CSV file.
    date_column:
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Dataframe with parsed dates.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)

    if date_column not in df.columns:
        raise ValueError(f"Date column not found: {date_column}")

    df[date_column] = pd.to_datetime(df[date_column])

    return df


def get_event_dates(
    audit_table: pd.DataFrame,
    assets: list[str],
    event_type: str | None = None,
) -> pd.DataFrame:
    """
    Filter an extreme-return audit table to selected assets and event types.

    Parameters
    ----------
    audit_table:
        Extreme-return audit table.
    assets:
        Assets to keep.
    event_type:
        Optional event type to keep.

    Returns
    -------
    pd.DataFrame
        Filtered audit table.
    """
    filtered = audit_table[audit_table["asset"].isin(assets)].copy()

    if event_type is not None:
        filtered = filtered[filtered["type"] == event_type].copy()

    filtered = filtered.sort_values(["asset", "date"])

    return filtered


def make_price_window_table(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    events: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """
    Create price and return windows around selected event dates.

    The window is measured in trading days, not calendar days.

    Parameters
    ----------
    prices:
        Wide price panel with dates as index and assets as columns.
    returns:
        Wide return panel with dates as index and assets as columns.
    events:
        Event table with columns asset, date, type, return, loss.
    window:
        Number of trading days before and after each event.

    Returns
    -------
    pd.DataFrame
        Table containing prices and returns around each event.
    """
    records = []

    for _, event in events.iterrows():
        asset = event["asset"]
        event_date = event["date"]

        if asset not in prices.columns:
            raise ValueError(f"Asset not found in price panel: {asset}")

        if asset not in returns.columns:
            raise ValueError(f"Asset not found in return panel: {asset}")

        if event_date not in prices.index:
            print(f"Warning: event date not found in prices index: {asset}, {event_date}")
            continue

        event_position = prices.index.get_loc(event_date)

        start_position = max(event_position - window, 0)
        end_position = min(event_position + window + 1, len(prices.index))

        window_dates = prices.index[start_position:end_position]

        for date in window_dates:
            records.append(
                {
                    "asset": asset,
                    "event_date": event_date,
                    "window_date": date,
                    "days_from_event": prices.index.get_loc(date) - event_position,
                    "event_type": event["type"],
                    "event_return": event["return"],
                    "event_loss": event["loss"],
                    "price": prices.loc[date, asset],
                    "return": returns.loc[date, asset] if date in returns.index else pd.NA,
                }
            )

    window_table = pd.DataFrame(records)

    if not window_table.empty:
        window_table = window_table.sort_values(
            ["asset", "event_date", "days_from_event"]
        )

    return window_table


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    """
    Save dataframe to CSV.

    Parameters
    ----------
    df:
        Dataframe to save.
    path:
        Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)