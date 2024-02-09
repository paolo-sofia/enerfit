import datetime
import logging
import pathlib

import lightgbm as lgb
import pandas as pd

from src.config_loader import load_model_config
from src.data.preprocessing import (
    create_dataset,
)
from src.utils import check_model_type

config = load_model_config()


def load_and_prepare_data_for_inference(
    model_type: str,
    is_business: bool = True,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> pd.DataFrame:
    """Load and prepare data for inference based on the given model type, business flag, start date, and end date.

    Args:
        model_type: The type of the model.
        is_business: Whether the data is for business or not. Defaults to True.
        start_date: Optional start date to filter the data. Defaults to None.
        end_date: Optional end date to filter the data. Defaults to None.

    Returns:
        pd.DataFrame: The prepared data for inference.
    """
    model_type: str = check_model_type(model_type)

    columns: list[str] = (
        config.get(model_type, {}).get("business" if is_business else "not_business", {}).get("columns", [])
    )

    data = create_dataset(
        model_type=model_type,
        is_business=is_business,
        columns=columns,
        dataset_type="inference",
        start_date=start_date,
        end_date=end_date,
    )

    return data.drop(columns=["date"]) if "date" in data.columns else data


def load_model(model_path: pathlib.Path) -> lgb.Booster | None:
    """Load a LightGBM model from the given file path and make predictions on the provided data.

    Args:
        model_path: The path to the LightGBM model file.

    Returns:
        list | None: A list of predicted values if the model is successfully loaded, None otherwise.
    """
    try:
        return lgb.Booster(model_file=model_path)
    except FileNotFoundError as e:
        logging.error(e)
        return None


def inference_pipeline(
    model_path: pathlib.Path, model_type: str, is_business: bool, start_date: datetime.date, end_date: datetime.date
) -> list | None:
    """Run the inference pipeline using the given model path, model type, business flag, start date, and end date.

    Args:
        model_path: The path to the model file.
        model_type: The type of the model.
        is_business: Whether the data is for business or not.
        start_date: The start date for data filtering.
        end_date: The end date for data filtering.

    Returns:
        list | None: A list of predicted values if the model is successfully loaded, None otherwise.
    """
    model: lgb.Booster = load_model(model_path)
    if not model:
        return None

    data = load_and_prepare_data_for_inference(
        model_type=model_type, is_business=is_business, start_date=start_date, end_date=end_date
    )

    return model.predict(data)
