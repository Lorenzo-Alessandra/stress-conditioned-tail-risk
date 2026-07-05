from src.data.download_free import download_all_free_data
from src.utils.config import load_yaml


def main() -> None:
    ticker_config = load_yaml("config/tickers_free.yaml")
    model_config = load_yaml("config/model_config.yaml")
    paths_config = load_yaml("config/paths.yaml")

    download_all_free_data(
        ticker_config=ticker_config,
        model_config=model_config,
        paths_config=paths_config,
    )


if __name__ == "__main__":
    main()