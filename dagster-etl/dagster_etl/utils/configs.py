import pathlib
from typing import Any

import tomllib

MODEL_PARAMETER_PATH: pathlib.Path = pathlib.Path("dagster-etl/dagster_etl/configs/models_parameter.toml")
DATA_PREPROCESSING_PATH: pathlib.Path = pathlib.Path("dagster-etl/dagster_etl/configs/data.toml")


def get_root_path() -> pathlib.Path:
    """Get the root path of the project.

    Returns:
        pathlib.Path: The root path of the project.

    Examples:
        >>> root_path = get_root_path()
        >>> # Use the root path for file operations or configuration loading
    """
    current_directory: pathlib.Path = pathlib.Path.cwd()
    while not list(current_directory.glob("*.git")):
        current_directory = current_directory.parent
    return current_directory


def load_model_config() -> dict[str, Any]:
    """Load the model configuration from a file.

    Returns:
        dict[str, Any]: A dictionary containing the model configuration.

    Examples:
        >>> model_config = load_model_config()
        >>> model = create_model(**model_config)
    """
    with (get_root_path() / MODEL_PARAMETER_PATH).open("rb") as f:
        return tomllib.load(f)


def load_data_preprocessing_config() -> dict[str, Any]:
    """Load the data preprocessing configuration from a file.

    Returns:
        dict[str, Any]: A dictionary containing the data preprocessing configuration.

    Examples:
        >>> data_preprocessing_config = load_data_preprocessing_config()
        >>> preprocessed_data = preprocess_data(data, **data_preprocessing_config)
    """
    with (get_root_path() / DATA_PREPROCESSING_PATH).open("rb") as f:
        return tomllib.load(f)
