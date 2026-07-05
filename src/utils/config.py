from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict
        Dictionary containing the YAML contents.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML file is empty.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file is empty: {path}")

    return config