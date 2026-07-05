from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr


def ensure_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path:
        Directory path.

    Returns
    -------
    Path
        The created or existing directory path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_bank_tickers(ticker_config: dict[str, Any]) -> list[str]:
    """
    Extract bank tickers from the ticker configuration.

    Parameters
    ----------
    ticker_config:
        Dictionary loaded from config/tickers_free.yaml.

    Returns
    -------
    list[str]
        List of bank tickers.
    """
    return list(ticker_config["banks"].keys())


def get_market_tickers(ticker_config: dict[str, Any]) -> dict[str, str]:
    """
    Extract Yahoo market tickers from the ticker configuration.

    Parameters
    ----------
    ticker_config:
        Dictionary loaded from config/tickers_free.yaml.

    Returns
    -------
    dict[str, str]
        Mapping from internal market name to Yahoo ticker.
    """
    market_tickers: dict[str, str] = {}

    for internal_name, info in ticker_config["market_indices"].items():
        market_tickers[internal_name] = info["yahoo_ticker"]

    return market_tickers


def download_yahoo_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download daily price data from Yahoo Finance using yfinance.

    Parameters
    ----------
    tickers:
        List of Yahoo tickers.
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Raw yfinance price dataframe.
    """
    if not tickers:
        raise ValueError("Ticker list is empty.")

    prices = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if prices.empty:
        raise ValueError("Yahoo download returned an empty dataframe.")

    return prices


def download_fred_series(
    fred_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Download a single time series from FRED.

    Parameters
    ----------
    fred_code:
        FRED series code.
    start_date:
        Start date in YYYY-MM-DD format.
    end_date:
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the FRED series.
    """
    series = pdr.DataReader(fred_code, "fred", start_date, end_date)

    if series.empty:
        raise ValueError(f"FRED download returned an empty dataframe for {fred_code}.")

    return series


def save_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Save a dataframe to CSV.

    Parameters
    ----------
    df:
        Dataframe to save.
    output_path:
        Output CSV path.
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path)


def download_all_free_data(
    ticker_config: dict[str, Any],
    model_config: dict[str, Any],
    paths_config: dict[str, Any],
) -> None:
    """
    Download all free raw data needed for the first pipeline version.

    This includes:
    - bank prices from Yahoo Finance
    - market benchmark prices from Yahoo Finance
    - VIX from FRED

    Parameters
    ----------
    ticker_config:
        Dictionary loaded from config/tickers_free.yaml.
    model_config:
        Dictionary loaded from config/model_config.yaml.
    paths_config:
        Dictionary loaded from config/paths.yaml.
    """
    start_date = model_config["sample"]["start_date"]
    end_date = model_config["sample"]["end_date"]

    raw_free_dir = Path(paths_config["data"]["raw_free"])
    ensure_directory(raw_free_dir)

    bank_tickers = get_bank_tickers(ticker_config)
    market_tickers = get_market_tickers(ticker_config)

    print("Downloading bank prices from Yahoo Finance...")
    print(f"Bank tickers: {bank_tickers}")

    bank_prices = download_yahoo_prices(
        tickers=bank_tickers,
        start_date=start_date,
        end_date=end_date,
    )

    save_dataframe(bank_prices, raw_free_dir / "bank_prices_yahoo.csv")
    print(f"Saved bank prices to {raw_free_dir / 'bank_prices_yahoo.csv'}")

    print("Downloading market prices from Yahoo Finance...")
    print(f"Market tickers: {market_tickers}")

    market_prices = download_yahoo_prices(
        tickers=list(market_tickers.values()),
        start_date=start_date,
        end_date=end_date,
    )

    save_dataframe(market_prices, raw_free_dir / "market_prices_yahoo.csv")
    print(f"Saved market prices to {raw_free_dir / 'market_prices_yahoo.csv'}")

    print("Downloading stress variables from FRED...")

    stress_frames: list[pd.DataFrame] = []

    for internal_name, info in ticker_config["stress_variables"].items():
        fred_code = info["fred_code"]

        print(f"Downloading {internal_name} from FRED code {fred_code}...")

        fred_series = download_fred_series(
            fred_code=fred_code,
            start_date=start_date,
            end_date=end_date,
        )

        fred_series = fred_series.rename(columns={fred_code: internal_name})
        stress_frames.append(fred_series)

    stress_variables = pd.concat(stress_frames, axis=1)

    save_dataframe(stress_variables, raw_free_dir / "stress_variables_fred.csv")
    print(f"Saved stress variables to {raw_free_dir / 'stress_variables_fred.csv'}")