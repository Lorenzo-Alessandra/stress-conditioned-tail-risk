import pandas as pd

from src.data.audit import (
    get_event_dates,
    make_price_window_table,
    read_csv_with_dates,
    save_csv,
)
from src.utils.config import load_yaml


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")

    bank_prices_path = paths_config["processed_files"]["bank_prices"]
    returns_path = paths_config["processed_files"]["returns"]
    audit_path = "outputs/tables/extreme_returns_audit.csv"

    print("Reading bank prices...")
    bank_prices = pd.read_parquet(bank_prices_path)

    print("Reading bank returns...")
    returns = pd.read_parquet(returns_path)

    print("Reading extreme-return audit table...")
    audit_table = read_csv_with_dates(audit_path)

    assets_to_inspect = ["UCG.MI", "INGA.AS"]

    print(f"Filtering events for: {assets_to_inspect}")
    events = get_event_dates(
        audit_table=audit_table,
        assets=assets_to_inspect,
        event_type=None,
    )

    print("Creating price-window audit table...")
    price_windows = make_price_window_table(
        prices=bank_prices,
        returns=returns,
        events=events,
        window=3,
    )

    output_path = "outputs/tables/price_window_audit_ucg_inga.csv"

    price_windows_for_output = price_windows.copy()
    price_windows_for_output["event_date"] = price_windows_for_output[
        "event_date"
    ].dt.strftime("%Y-%m-%d")
    price_windows_for_output["window_date"] = price_windows_for_output[
        "window_date"
    ].dt.strftime("%Y-%m-%d")

    numeric_columns = ["event_return", "event_loss", "price", "return"]

    for column in numeric_columns:
        price_windows_for_output[column] = pd.to_numeric(
            price_windows_for_output[column],
            errors="coerce",
        ).round(6)

    save_csv(price_windows_for_output, output_path)

    print("\nPrice-window audit table created.")
    print(f"Saved to: {output_path}")

    print("\nPreview:")
    print(price_windows_for_output.head(40))


if __name__ == "__main__":
    main()