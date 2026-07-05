from src.data.build_returns import (
    align_to_reference_index,
    compute_equal_weighted_system_loss,
    compute_log_returns,
    compute_losses,
    drop_missing_rows,
    read_parquet,
    save_parquet,
    summarize_returns,
)
from src.utils.config import load_yaml


def main() -> None:
    paths_config = load_yaml("config/paths.yaml")
    model_config = load_yaml("config/model_config.yaml")

    bank_prices_path = paths_config["processed_files"]["bank_prices"]
    market_prices_path = paths_config["processed_files"]["market_prices"]
    stress_variables_path = paths_config["processed_files"]["stress_variables"]

    print("Reading processed bank prices...")
    bank_prices = read_parquet(bank_prices_path)

    model_config = load_yaml("config/model_config.yaml")
    included_banks = model_config["modeling_universe"]["included_banks"]

    missing_banks = [bank for bank in included_banks if bank not in bank_prices.columns]

    if missing_banks:
        raise ValueError(f"Included banks not found in price panel: {missing_banks}")

    bank_prices = bank_prices[included_banks]

    print(f"Using modeling universe with {len(included_banks)} banks:")
    print(included_banks)

    print("Reading processed market prices...")
    market_prices = read_parquet(market_prices_path)

    print("Reading processed stress variables...")
    stress_variables = read_parquet(stress_variables_path)

    print("\nComputing bank log returns...")
    bank_returns_raw = compute_log_returns(bank_prices)

    print("Computing bank losses...")
    bank_losses_raw = compute_losses(bank_returns_raw)

    print("\nDropping missing rows from bank returns...")
    bank_returns = drop_missing_rows(bank_returns_raw, "Bank returns")

    print("\nAligning bank losses to cleaned bank return index...")
    bank_losses = bank_losses_raw.reindex(bank_returns.index)

    print("\nComputing market log returns...")
    market_returns_raw = compute_log_returns(market_prices)

    print("\nAligning market returns to bank return index...")
    market_returns = align_to_reference_index(
        df=market_returns_raw,
        reference_index=bank_returns.index,
        name="Market returns",
    )

    print("\nAligning stress variables to bank return index...")
    stress_variables_aligned = align_to_reference_index(
        df=stress_variables,
        reference_index=bank_returns.index,
        name="Stress variables",
    )

    print("\nComputing equal-weighted system loss...")
    system_loss = compute_equal_weighted_system_loss(bank_losses)

    summarize_returns(bank_returns, "Bank returns")
    summarize_returns(bank_losses, "Bank losses")
    summarize_returns(market_returns, "Market returns")
    summarize_returns(stress_variables_aligned, "Stress variables aligned")
    summarize_returns(system_loss.to_frame(), "Equal-weighted system loss")

    save_parquet(bank_returns, paths_config["processed_files"]["returns"])
    save_parquet(bank_losses, paths_config["processed_files"]["losses"])
    save_parquet(market_returns, "data/processed/market_returns.parquet")
    save_parquet(stress_variables_aligned, paths_config["processed_files"]["stress_variables"])
    save_parquet(system_loss, paths_config["processed_files"]["system_loss"])

    print("\nReturns and losses saved successfully.")
    print(f"Bank returns: {paths_config['processed_files']['returns']}")
    print(f"Bank losses: {paths_config['processed_files']['losses']}")
    print("Market returns: data/processed/market_returns.parquet")
    print(f"Stress variables: {paths_config['processed_files']['stress_variables']}")
    print(f"System loss: {paths_config['processed_files']['system_loss']}")


if __name__ == "__main__":
    main()