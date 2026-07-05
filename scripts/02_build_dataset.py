from src.data.clean_prices import (
    clean_price_panel,
    read_stress_csv,
    read_yahoo_price_csv,
    save_parquet,
    summarize_panel,
)
from src.utils.config import load_yaml


def main() -> None:
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    field_preference = model_config["returns"]["price_field_preference"]

    raw_free_dir = paths_config["data"]["raw_free"]

    bank_raw_path = f"{raw_free_dir}/bank_prices_yahoo.csv"
    market_raw_path = f"{raw_free_dir}/market_prices_yahoo.csv"
    stress_raw_path = f"{raw_free_dir}/stress_variables_fred.csv"

    print("Reading raw bank prices...")
    raw_bank_prices = read_yahoo_price_csv(bank_raw_path)

    print("Cleaning bank price panel...")
    bank_prices = clean_price_panel(
        raw_prices=raw_bank_prices,
        field_preference=field_preference,
    )

    print("Reading raw market prices...")
    raw_market_prices = read_yahoo_price_csv(market_raw_path)

    print("Cleaning market price panel...")
    market_prices = clean_price_panel(
        raw_prices=raw_market_prices,
        field_preference=field_preference,
    )

    print("Reading stress variables...")
    stress_variables = read_stress_csv(stress_raw_path)

    summarize_panel(bank_prices, "Bank price panel")
    summarize_panel(market_prices, "Market price panel")
    summarize_panel(stress_variables, "Stress variables")

    save_parquet(
        bank_prices,
        paths_config["processed_files"]["bank_prices"],
    )

    save_parquet(
        market_prices,
        paths_config["processed_files"]["market_prices"],
    )

    save_parquet(
        stress_variables,
        paths_config["processed_files"]["stress_variables"],
    )

    print("\nProcessed files saved successfully.")
    print(f"Bank prices: {paths_config['processed_files']['bank_prices']}")
    print(f"Market prices: {paths_config['processed_files']['market_prices']}")
    print(f"Stress variables: {paths_config['processed_files']['stress_variables']}")


if __name__ == "__main__":
    main()