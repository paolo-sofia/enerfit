import datetime
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytz
from dagster import asset
from dagster_etl.resources.model_dataset_config_resource import ModelDatasetConfigResource
from dagster_etl.resources.model_parameters_resource import ModelParametersResource
from dateutil.relativedelta import relativedelta

MAP_MODEL_TYPE: dict[str, int] = {"producer": 0, "consumer": 1}
TIMEZONE: pytz.timezone = pytz.timezone("Europe/Rome")


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None

    Examples:
        >>> set_seed(42)
        >>> # Perform operations that require random number generation
    """
    np.random.default_rng(seed)
    random.seed(seed)


def split_train_test(data: pd.DataFrame, test_months: int = 3) -> tuple[list[int], ...]:
    """Split the data into train and test sets based on a specified number of test months.

    Args:
        data (pd.DataFrame): The input DataFrame containing a 'date' column.
        test_months (int, optional): The number of months to include in the test set. Defaults to 6.

    Returns:
        tuple[list[int], ...]: A tuple containing the train and test indices.

    Examples:
        >>> data = pd.DataFrame(...)
        >>> train_index, test_index = split_train_test(data)
        >>> train_data = data.loc[train_index]
        >>> test_data = data.loc[test_index]
    """
    today: datetime.date = datetime.datetime.now(tz=TIMEZONE).today()
    start_test_date: datetime.date = today - relativedelta(months=test_months)

    train_index = data[data["date"] < start_test_date].index
    test_index = data[data["date"] >= start_test_date].index
    return train_index, test_index


def train_model(
    dataset: pd.DataFrame,
    model_data_resource: ModelDatasetConfigResource,
    model_config: dict,
    seed: int = 42,
) -> lgb.LGBMModel:
    """Train a model using the provided dataset, model data resource, and model configuration.

    Args:
        dataset (pd.DataFrame): The dataset for training the model.
        model_data_resource (ModelDatasetConfigResource): The configuration for the model dataset.
        model_config (ModelParametersResource): The configuration for the model parameters.

    Returns:
        lgb.LGBMModel: The trained model.
    """
    set_seed(seed=seed)

    train_index, test_index = split_train_test(dataset, model_data_resource.test_months)

    dataset = dataset.drop(columns=["date"])

    if model_data_resource.columns:
        dataset = dataset[model_data_resource.columns]

    x_train, y_train = dataset.drop(columns=["target"]).loc[train_index], dataset.pop("target").loc[train_index]
    x_test, y_test = dataset.drop(columns=["target"]).loc[test_index], dataset.pop("target").loc[test_index]

    model_config |= {"seed": seed}
    model = lgb.LGBMRegressor(**model_config)

    model.fit(
        x_train,
        y_train,
        eval_names=["train", "valid"],
        eval_set=[(x_train, y_train), (x_test, y_test)],
        eval_metric="mae",
        callbacks=[
            lgb.callback.early_stopping(first_metric_only=True, stopping_rounds=100),
            lgb.callback.log_evaluation(),
        ],
    )
    return model


@asset(
    name="producer_model",
    io_manager_key="io_manager",
    key_prefix=["model", "models"],
    compute_kind="LightGBM",
)
def train_producer_model(
    training_dataset: pd.DataFrame,
    model_data_resource: ModelDatasetConfigResource,
    model_config: ModelParametersResource,
) -> lgb.LGBMModel:
    """Train a producer model using the provided dataset, model data resource, and model configuration.

    Args:
        training_dataset (pd.DataFrame): The dataset for training the model.
        model_data_resource (ModelDatasetConfigResource): The configuration for the model dataset.
        model_config (ModelParametersResource): The configuration for the model parameters.

    Returns:
        lgb.LGBMModel: The trained producer model.

    Note:
        This function is decorated with `@asset` to define the asset properties.
    """
    today: str = datetime.datetime.now(tz=TIMEZONE).today().strftime("%Y_%m_%d")

    training_dataset = training_dataset.query("is_consumption == 0").drop(columns=["is_consumption"])
    model = train_model(
        training_dataset,
        model_data_resource=model_data_resource,
        model_config=model_config.producer_parameters,
        seed=model_config.seed,
    )

    model.booster_.save_model(filename=f"../../data/model/models/producer/{today}.joblib")
    return model


@asset(
    name="consumer_model",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["model", "dataset"],
    compute_kind="polars",
)
def train_consumer_model(
    training_dataset: pd.DataFrame,
    model_data_resource: ModelDatasetConfigResource,
    model_config: ModelParametersResource,
) -> lgb.LGBMModel:
    """Train a consumer model using the provided dataset, model data resource, and model configuration.

    Args:
        training_dataset (pd.DataFrame): The dataset for training the model.
        model_data_resource (ModelDatasetConfigResource): The configuration for the model dataset.
        model_config (ModelParametersResource): The configuration for the model parameters.

    Returns:
        lgb.LGBMModel: The trained consumer model.

    Note:
        This function is decorated with `@asset` to define the asset properties.
    """
    today: str = datetime.datetime.now(tz=TIMEZONE).today().strftime("%Y_%m_%d")

    training_dataset = training_dataset.query("is_consumption == 1").drop(columns=["is_consumption"])
    model = train_model(
        training_dataset,
        model_data_resource=model_data_resource,
        model_config=model_config.consumer_parameters,
        seed=model_config.seed,
    )

    model.booster_.save_model(filename=f"../../data/model/models/consumer/{today}.joblib")
    return model
