import datetime
import random

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytz
from dagster import asset
from dagster_etl.resources.model_dataset_config_resource import ModelDatasetConfigResource
from dagster_etl.resources.model_parameters_resource import ModelParametersResource
from dateutil.relativedelta import relativedelta

MAP_MODEL_TYPE: dict[str, int] = {"producer": 0, "consumer": 1}
TIMEZONE: pytz.timezone = pytz.timezone("Europe/Rome")
DATE_FORMAT: str = "%Y-%m-%d"

today: datetime.date = datetime.datetime.now(tz=TIMEZONE).today()
year: str = str(today.year)
month: str = str(today.month)
day: str = str(today.day)


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
    max_dataset_date: datetime.date = data["date"].max()
    start_test_date: datetime.date = max_dataset_date - relativedelta(months=test_months)

    train_index = data[data["date"] < start_test_date].index
    test_index = data[data["date"] >= start_test_date].index
    return train_index, test_index


def convert_objects_columns_to_category(dataset: pd.DataFrame) -> pd.DataFrame:
    """Converts object-type columns in the given DataFrame to the category type.

    Args:
        dataset: The DataFrame containing the columns to be converted.

    Returns:
        The DataFrame with the object-type columns converted to the category type.
    """
    for col in dataset.columns:
        if dataset[col].dtype == "object":
            dataset[col] = dataset[col].astype("category")
    return dataset


def create_train_test(
    dataset: pl.LazyFrame,
    model_data_resource: ModelDatasetConfigResource,
    model_config: dict,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create train and test datasets for model training.

    Args:
        dataset: The dataset to create train and test datasets from.
        model_data_resource: The resource containing the model dataset configuration.
        model_config: The configuration parameters for the model.

    Returns:
        x_train: The training features dataset.
        y_train: The training target dataset.
        x_test: The testing features dataset.
        y_test: The testing target dataset.
    """
    dataset = dataset.drop(
        [
            "client_id",
            # "date",
            "datetime",
            "data_block_id",
            "county",
            "county_name_forecast",
            "date_client",
            "date_gas",
            "latitude_max_forecast",
            "latitude_min_forecast",
            "longitude_max_forecast",
            "longitude_min_forecast",
        ]
    )

    dataset: pd.DataFrame = dataset.collect().to_pandas()
    dataset = convert_objects_columns_to_category(dataset)

    train_index, test_index = split_train_test(dataset, model_data_resource.test_months)

    if model_config.get("columns", []):
        dataset = dataset[model_data_resource.columns]

    dataset = dataset.drop(columns=["date"])
    x_train: pd.DataFrame = dataset.loc[train_index].copy()
    x_test: pd.DataFrame = dataset.loc[test_index].copy()

    del dataset

    y_train: pd.Series = x_train.pop("target")
    y_test: pd.Series = x_test.pop("target")

    return x_train, y_train, x_test, y_test


def train_model(
    dataset: pl.LazyFrame,
    model_data_resource: ModelDatasetConfigResource,
    model_config: dict,
    seed: int = 42,
) -> lgb.LGBMModel:
    """Train a LightGBM model using the provided dataset and configuration.

    Args:
        dataset: The dataset to train the model on.
        model_data_resource: The resource containing the model dataset configuration.
        model_config: The configuration parameters for the model.
        seed (optional): The random seed for reproducibility. Defaults to 42.

    Returns:
        The trained LightGBM model.
    """
    set_seed(seed=seed)

    x_train, y_train, x_test, y_test = create_train_test(
        dataset=dataset, model_config=model_config, model_data_resource=model_data_resource
    )

    model_config |= {"seed": seed}
    del model_config["columns"]

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
    key_prefix=["model", "models", "producer", year, month, day],
    compute_kind="LightGBM",
)
def train_producer_model(
    training_dataset: pl.LazyFrame,
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
    training_dataset = training_dataset.filter(pl.col("is_consumption") == 0).drop("is_consumption")

    return train_model(
        dataset=training_dataset,
        model_data_resource=model_data_resource,
        model_config=model_config.consumer_parameters,
        seed=model_config.seed,
    )


@asset(
    name="consumer_model",
    key_prefix=["model", "models", "consumer", year, month, day],
    compute_kind="LightGBM",
)
def train_consumer_model(
    training_dataset: pl.LazyFrame,
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
    training_dataset = training_dataset.filter(pl.col("is_consumption") == 1).drop("is_consumption")

    return train_model(
        dataset=training_dataset,
        model_data_resource=model_data_resource,
        model_config=model_config.consumer_parameters,
        seed=model_config.seed,
    )
