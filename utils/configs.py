import pathlib
from typing import Any

import tomllib


def load_config(config_file: pathlib.Path) -> dict[str, Any]:
    """Loads a configuration from a specified file.

    Args:
        config_file (pathlib.Path): The path to the configuration file.

    Returns:
        dict: The loaded configuration.

    Examples:
        >>> config_file = pathlib.Path("/path/to/config.toml")
        >>> load_config(config_file)
        {'key': 'value', 'foo': 'bar'}
    """
    with config_file.open("rb") as f:
        config: dict[str, Any] = tomllib.load(f)

    return config
