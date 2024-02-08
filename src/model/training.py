import datetime
import pathlib
from typing import Any

import lightgbm as lgb
import pandas as pd
import polars as pl

from src.config_loader import get_root_path
from src.data.preprocessing import add_noise_feature_for_training, convert_objects_columns_to_category, feature_engineer
from src.loader import create_dataset
from src.model.model_selection import cross_validation_month_and_year, split_train_test
from src.utils import MAP_MODEL_TYPE, load_model_config, set_seed

config = load_model_config()

SEED: int = config.get("general", {}).get("seed", 42)
set_seed(seed=SEED)


def train_model_cross_validation(dataframe: pd.DataFrame) -> list[lgb.LGBMModel]:
    """Train multiple models using cross-validation based on month and year.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        list[lgb.LGBMModel]: A list of trained LGBMModel objects.

    Examples:
        >>> dataframe = pd.DataFrame(...)
        >>> models = train_model_cross_validation(dataframe)
        >>> for model in models:
        ...     # Use the trained model for prediction or evaluation
    """
    models = []
    for i, (train_index, test_index, start_train_date, end_train_date, start_test_date, end_test_date) in enumerate(
        cross_validation_month_and_year(dataframe, train_months=6, test_months=3, debug=False)
    ):
        print(
            f"Split {i + 1} - train from {start_train_date} to {end_train_date} --- test from {start_test_date} to {end_test_date}"
        )
        models.append(train_model(dataframe, train_index, test_index))
        break
    return models


def load_data_and_train_model(model_type: str, is_business: bool = True) -> lgb.LGBMModel:
    is_business: int = int(is_business)
    model_type = model_type.lower()

    if model_type not in ["producer", "consumer"]:
        raise ValueError(f"Model type must be either 'producer' or 'consumer', given model type is: {model_type}")

    parameters = config.get(model_type, {}).get("business" if is_business else "not_business", {})

    data = create_dataset()
    data = data.filter(
        (pl.col("is_consumption") == MAP_MODEL_TYPE.get(model_type, 0)) & (pl.col("is_business") == is_business)
    ).drop(["is_consumption", "is_business"])
    data = feature_engineer(data)
    data = data.drop_nulls()
    data = data.select(parameters.get("columns", {}))
    data = add_noise_feature_for_training(data).to_pandas()
    data = convert_objects_columns_to_category(data)

    train_index, test_index = split_train_test(data=data)
    data = data.drop(columns=["date"])

    return train_model(dataframe=data, train_indexes=train_index, test_indexes=test_index, parameters=parameters)


def train_model(
    dataframe: pd.DataFrame, train_indexes: list[int], test_indexes: list[int], parameters: dict[str, Any]
) -> lgb.LGBMModel:
    """Train a LightGBM model using the specified data and parameters.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the training data.
        train_indexes (list[int]): The indices of the training data.
        test_indexes (list[int]): The indices of the test data.
        parameters (dict[str, Any]): The parameters for the LightGBM model.

    Returns:
        lgb.LGBMModel: The trained LightGBM model.

    Examples:
        >>> dataframe = pd.DataFrame(...)
        >>> train_indexes = [0, 1, 2, 3, 4]
        >>> test_indexes = [5, 6, 7]
        >>> parameters = {"objective": "l2", "num_leaves": 10, "learning_rate": 0.1}
        >>> model = train_model(dataframe, train_indexes, test_indexes, parameters)
        >>> # Use the trained model for prediction or evaluation
    """
    x_train, x_test = dataframe.loc[train_indexes], dataframe.loc[test_indexes]
    y_train, y_test = x_train.pop("target"), x_test.pop("target")
    # eval_results = {}
    model = lgb.LGBMRegressor(random_state=SEED, **parameters)

    model.fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="mae",
        callbacks=[
            lgb.log_evaluation(),
            # lgb.record_evaluation(eval_results),
            lgb.early_stopping(stopping_rounds=100),
        ],
    )
    return model


def save_model(model: lgb.LGBMModel, model_type: str, is_business: bool = True) -> bool:
    model_name: str = f"""{model_type}_{"business" if is_business else "not_business"}"""
    year, month, day = datetime.datetime.now(tz=datetime.UTC).date().strftime("%Y-%m-%d").split("-")
    output_dir: pathlib.Path = (pathlib.Path(get_root_path()) / "data" / "model" / str(year) / str(month)) / str(day)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.booster_.save_model((output_dir / model_name).with_suffix(".joblib"))
        return True
    except Exception:
        return False
