from src.utils.config import load_yaml


def main() -> None:
    tickers = load_yaml("config/tickers_free.yaml")
    model_config = load_yaml("config/model_config.yaml")
    paths = load_yaml("config/paths.yaml")

    print("Configuration files loaded successfully.")
    print(f"Number of banks: {len(tickers['banks'])}")
    print(f"Sample start date: {model_config['sample']['start_date']}")
    print(f"Processed returns path: {paths['processed_files']['returns']}")


if __name__ == "__main__":
    main()