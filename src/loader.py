#!/usr/bin/env python
# coding: utf-8

# ## Imports


import datetime
import pathlib
import random
from typing import Generator

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import tomllib
from dateutil.relativedelta import relativedelta

# ## Environment variables and Constants definition


DEBUG = False
SEED = 42
DATE_FORMAT: str = "%Y-%m-%d"
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
TIMEZONE: str = "Europe/Tallinn"
ESTONIAN_HOLIDAYS = list(holidays.country_holidays("EE", years=range(2021, 2026)).keys())

MAP_MODEL_TYPE: dict[str, int] = {"producer": 0, "consumer": 1}


# ## Helper functions


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    pl.set_random_seed(seed)


set_seed(seed=SEED)

# ### Paths

# ### Data path


BASE_DATA_PATH: pathlib.Path = pathlib.Path().absolute().parent.parent / "data" / "predict-energy-behavior-of-prosumers"

CLIENTS_PATH: pathlib.Path = BASE_DATA_PATH / "client.csv"
ELECTRICITY_PATH: pathlib.Path = BASE_DATA_PATH / "electricity_prices.csv"
GAS_PATH: pathlib.Path = BASE_DATA_PATH / "gas_prices.csv"
HISTORICAL_WEATHER_PATH: pathlib.Path = BASE_DATA_PATH / "historical_weather.csv"
WEATHER_FORECAST_PATH: pathlib.Path = BASE_DATA_PATH / "forecast_weather.csv"
TRAIN_PATH: pathlib.Path = BASE_DATA_PATH / "train.csv"
WEATHER_STATION_COUNTY_PATH: pathlib.Path = BASE_DATA_PATH / "weather_station_to_county_mapping.csv"
COUNTY_MAP_PATH: pathlib.Path = BASE_DATA_PATH / "county_id_to_name_map.json"

# ### Test data path


TEST_DATA_PATH: pathlib.Path = BASE_DATA_PATH / "example_test_files"

CLIENTS_TEST_PATH: pathlib.Path = TEST_DATA_PATH / "client.csv"
ELECTRICITY_TEST_PATH: pathlib.Path = TEST_DATA_PATH / "electricity_prices.csv"
GAS_TEST_PATH: pathlib.Path = TEST_DATA_PATH / "gas_prices.csv"
HISTORICAL_TEST_WEATHER_PATH: pathlib.Path = TEST_DATA_PATH / "historical_weather.csv"
WEATHER_FORECAST_TEST_PATH: pathlib.Path = TEST_DATA_PATH / "forecast_weather.csv"
TRAIN_TEST_PATH: pathlib.Path = TEST_DATA_PATH / "train.csv"
REVEALED_TARGET_PATH: pathlib.Path = TEST_DATA_PATH / "revealed_targets.csv"
SAMPLE_SUBMISSION_PATH: pathlib.Path = TEST_DATA_PATH / "sample_submission.csv"

# ### Other


MODEL_PARAMETER_PATH: pathlib.Path = pathlib.Path("/src/models_parameter.toml")


# ## Helper functions


def assert_null_counts(dataframe: pl.LazyFrame) -> bool:
    if not DEBUG:
        return False
    length_dataframe: int = dataframe.collect().shape[0]

    return np.any(
        dataframe.null_count()
        .with_columns(*[pl.col(col) == length_dataframe for col in dataframe.columns])
        .collect()
        .to_numpy()
    )


def load_parameters(path: pathlib.Path) -> dict[str, str]:
    with path.open("rb") as f:
        return tomllib.load(f)


config = load_parameters(pathlib.Path("/src/models_parameter.toml"))
config


# ## Data load

# ## Load client and convert columns


def load_clients() -> pl.LazyFrame:
    clients: pl.LazyFrame = pl.scan_csv(CLIENTS_PATH)
    clients = clients.with_columns(
        [
            pl.col("product_type").cast(pl.Int8),
            pl.col("county").cast(pl.Int8),
            pl.col("eic_count").cast(pl.Int16),
            pl.col("installed_capacity").cast(pl.Float32),
            pl.col("is_business").cast(pl.Int8),
            pl.col("date").str.to_date(DATE_FORMAT),
            pl.col("data_block_id").cast(pl.Int16),
        ]
    )
    return clients


def load_electricity() -> pl.LazyFrame:
    electricity: pl.LazyFrame = pl.scan_csv(ELECTRICITY_PATH).drop(["origin_date"])
    electricity = electricity.with_columns(
        [
            pl.col("forecast_date").str.to_datetime(DATETIME_FORMAT) + pl.duration(days=1),
            pl.col("euros_per_mwh").cast(pl.Float32),
            pl.col("data_block_id").cast(pl.Int16),
        ]
    ).rename({"forecast_date": "datetime", "euros_per_mwh": "electricity_euros_per_mwh"})
    return electricity


def load_gas() -> pl.LazyFrame:
    gas: pl.LazyFrame = pl.scan_csv(GAS_PATH).drop(["origin_date"])
    gas = gas.with_columns(
        [
            pl.col("forecast_date").str.to_date(DATE_FORMAT),
            pl.col("lowest_price_per_mwh").cast(pl.Float32),
            pl.col("highest_price_per_mwh").cast(pl.Float32),
            ((pl.col("lowest_price_per_mwh") + pl.col("highest_price_per_mwh")) / 2).alias("gas_mean_price_per_mhw"),
            pl.col("data_block_id").cast(pl.Int16),
        ]
    ).rename(
        {
            "forecast_date": "date",
            "lowest_price_per_mwh": "gas_lowest_price_per_mwh",
            "highest_price_per_mwh": "gas_highest_price_per_mwh",
        }
    )
    return gas


def load_weather_station_mapping() -> pl.LazyFrame:
    weather_station_county_mapping: pl.LazyFrame = pl.scan_csv(WEATHER_STATION_COUNTY_PATH)
    weather_station_county_mapping = weather_station_county_mapping.with_columns(
        [
            pl.col("longitude").cast(pl.Float32).round(decimals=2),
            pl.col("latitude").cast(pl.Float32).round(decimals=2),
            pl.col("county").cast(pl.Int8).fill_null(-1),
            pl.col("county_name").fill_null("Unknown"),
        ]
    )

    weather_station_county_mapping = weather_station_county_mapping.join(
        other=weather_station_county_mapping.group_by("county").agg(
            [
                pl.col("longitude").min().alias("longitude_min"),
                pl.col("longitude").max().alias("longitude_max"),
                pl.col("latitude").min().alias("latitude_min"),
                pl.col("latitude").max().alias("latitude_max"),
            ]
        ),
        on=["county"],
        how="inner",
    )

    return weather_station_county_mapping


# ## Load weather forecast and convert columns


def load_weather_forecast(weather_station_county_mapping: pl.LazyFrame) -> pl.LazyFrame:
    weather_forecast: pl.LazyFrame = (
        pl.scan_csv(WEATHER_FORECAST_PATH).drop(["origin_datetime"]).rename({"forecast_datetime": "datetime"})
    )
    # weather_forecast = weather_forecast.filter(pl.col("hours_ahead") >= 24)  # we don't need to forecast for today
    weather_forecast = weather_forecast.with_columns(
        [
            pl.col("datetime").str.to_datetime(DATETIME_FORMAT),
            pl.col("latitude").cast(pl.Float32).round(decimals=2),
            pl.col("longitude").cast(pl.Float32).round(decimals=2),
            pl.col("data_block_id").cast(pl.Int16),
        ]
    )

    weather_forecast = weather_forecast.join(
        other=weather_station_county_mapping, how="left", on=["latitude", "longitude"]
    ).drop(["latitude", "longitude"])

    weather_forecast = weather_forecast.group_by("county", "datetime", "data_block_id").agg(
        pl.col("hours_ahead").mean(),
        pl.col("temperature").mean(),
        pl.col("dewpoint").mean(),
        pl.col("cloudcover_high").mean(),
        pl.col("cloudcover_low").mean(),
        pl.col("cloudcover_mid").mean(),
        pl.col("cloudcover_total").mean(),
        pl.col("10_metre_u_wind_component").mean(),
        pl.col("10_metre_v_wind_component").mean(),
        pl.col("direct_solar_radiation").mean(),
        pl.col("surface_solar_radiation_downwards").mean(),
        pl.col("snowfall").mean(),
        pl.col("total_precipitation").mean(),
        pl.col("latitude_min").first(),
        pl.col("latitude_max").first(),
        pl.col("longitude_min").first(),
        pl.col("longitude_max").first(),
        pl.col("county_name").first(),
    )

    return weather_forecast


# ## Load historical weather and convert columns


def load_historical_weather(weather_station_county_mapping: pl.LazyFrame) -> pl.LazyFrame:
    historical_weather: pl.LazyFrame = pl.scan_csv(HISTORICAL_WEATHER_PATH)
    historical_weather = historical_weather.with_columns(
        [
            pl.col("datetime").str.to_datetime(DATETIME_FORMAT),
            pl.col("latitude").cast(pl.Float32).round(decimals=2),
            pl.col("longitude").cast(pl.Float32).round(decimals=2),
            pl.col("data_block_id").cast(pl.Int16),
        ]
    )

    historical_weather = historical_weather.join(
        other=weather_station_county_mapping, how="left", on=["latitude", "longitude"]
    ).drop(["latitude", "longitude"])

    historical_weather = historical_weather.group_by("county", "datetime", "data_block_id").agg(
        pl.col("temperature").mean(),
        pl.col("dewpoint").mean(),
        pl.col("rain").mean(),
        pl.col("snowfall").mean(),
        pl.col("surface_pressure").mean(),
        pl.col("cloudcover_total").mean(),
        pl.col("cloudcover_low").mean(),
        pl.col("cloudcover_mid").mean(),
        pl.col("cloudcover_high").mean(),
        pl.col("windspeed_10m").mean(),
        pl.col("winddirection_10m").mean(),
        pl.col("shortwave_radiation").mean(),
        pl.col("direct_solar_radiation").mean(),
        pl.col("diffuse_radiation").mean(),
        pl.col("latitude_min").first(),
        pl.col("latitude_max").first(),
        pl.col("longitude_min").first(),
        pl.col("longitude_max").first(),
        pl.col("county_name").first(),
    )

    # Test set has 1 day offset for hour<11 and 2 day offset for hour>11
    historical_weather = historical_weather.with_columns(
        pl.when(pl.col("datetime").dt.hour() < 11)
        .then(pl.col("datetime") + pl.duration(days=1))
        .otherwise(pl.col("datetime") + pl.duration(days=2))
    )

    return historical_weather


# ## Load Train and convert columns


def load_train() -> pl.LazyFrame:
    train: pl.LazyFrame = pl.scan_csv(TRAIN_PATH)

    train = train.drop(["prediction_unit_id", "row_id"]).with_columns(
        pl.col("datetime").str.to_datetime(DATETIME_FORMAT),
        pl.col("is_business").cast(pl.Int8),
        pl.col("product_type").cast(pl.Int8),
        pl.col("target").cast(pl.Float32),
        pl.col("is_consumption").cast(pl.Int8),
        pl.col("county").cast(pl.Int8),
        pl.col("data_block_id").cast(pl.Int16),
    )
    train = train.with_columns(
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").cast(pl.Date).alias("date"),
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        pl.col("datetime").dt.day().alias("day"),
        pl.col("datetime").dt.ordinal_day().alias("day_of_year"),
        pl.col("datetime").dt.hour().alias("hour"),
    )
    return train


# ### Create lagged weather features


def create_lagged_weather_forecast(weather_forecast: pl.LazyFrame) -> pl.LazyFrame:
    return (
        weather_forecast.sort("county", "datetime", "data_block_id")
        .rolling(index_column="datetime", period="1d", by=["county", "data_block_id"])
        .agg(
            [
                pl.col("temperature").mean().alias("temperature_forecast_last_day"),
                pl.col("dewpoint").mean().alias("dewpoint_forecast_last_day"),
                pl.col("snowfall").mean().alias("snowfall_forecast_last_day"),
                pl.col("cloudcover_total").mean().alias("cloudcover_total_forecast_last_day"),
                pl.col("cloudcover_low").mean().alias("cloudcover_low_forecast_last_day"),
                pl.col("cloudcover_mid").mean().alias("cloudcover_mid_forecast_last_day"),
                pl.col("cloudcover_high").mean().alias("cloudcover_high_forecast_last_day"),
                pl.col("10_metre_u_wind_component").mean().alias("10_metre_u_wind_component_forecast_last_day"),
                pl.col("10_metre_v_wind_component").mean().alias("10_metre_v_wind_component_forecast_last_day"),
                pl.col("surface_solar_radiation_downwards")
                .mean()
                .alias("surface_solar_radiation_downwards_forecast_last_day"),
                pl.col("direct_solar_radiation").mean().alias("direct_solar_radiation_forecast_last_day"),
                pl.col("total_precipitation").mean().alias("total_precipitation_forecast_last_day"),
            ]
        )
    )


def create_lagged_historical_weather_last_week(historical_weather: pl.LazyFrame) -> pl.LazyFrame:
    return (
        historical_weather.sort("county", "datetime", "data_block_id")
        .rolling(index_column="datetime", period="1w", by=["county"])
        .agg(
            [
                pl.col("temperature").mean().alias("temperature_last_7_days"),
                pl.col("dewpoint").mean().alias("dewpoint_last_7_days"),
                pl.col("rain").mean().alias("rain_last_7_days"),
                pl.col("snowfall").mean().alias("snowfall_last_7_days"),
                pl.col("cloudcover_total").mean().alias("cloudcover_total_last_7_days"),
                pl.col("cloudcover_low").mean().alias("cloudcover_low_last_7_days"),
                pl.col("cloudcover_mid").mean().alias("cloudcover_mid_last_7_days"),
                pl.col("cloudcover_high").mean().alias("cloudcover_high_last_7_days"),
                pl.col("windspeed_10m").mean().alias("windspeed_10m_last_7_days"),
                pl.col("winddirection_10m").mean().alias("winddirection_10m_last_7_days"),
                pl.col("shortwave_radiation").mean().alias("shortwave_radiation_last_7_days"),
                pl.col("direct_solar_radiation").mean().alias("direct_solar_radiation_last_7_days"),
                pl.col("diffuse_radiation").mean().alias("diffuse_radiation_last_7_days"),
            ]
        )
    )


def create_lagged_historical_weather_last_day(historical_weather: pl.LazyFrame) -> pl.LazyFrame:
    return (
        historical_weather.with_columns(pl.col("datetime").dt.hour().alias("hour"))
        .sort("county", "datetime")
        .rolling(index_column="datetime", period="1d", by=["county"])
        .agg(
            [
                pl.col("temperature").mean().alias("temperature_last_24_hours"),
                pl.col("dewpoint").mean().alias("dewpoint_last_24_hours"),
                pl.col("rain").mean().alias("rain_last_24_hours"),
                pl.col("snowfall").mean().alias("snowfall_last_24_hours"),
                pl.col("cloudcover_total").mean().alias("cloudcover_total_last_24_hours"),
                pl.col("cloudcover_low").mean().alias("cloudcover_low_last_24_hours"),
                pl.col("cloudcover_mid").mean().alias("cloudcover_mid_last_24_hours"),
                pl.col("cloudcover_high").mean().alias("cloudcover_high_last_24_hours"),
                pl.col("windspeed_10m").mean().alias("windspeed_10m_last_24_hours"),
                pl.col("winddirection_10m").mean().alias("winddirection_10m_last_24_hours"),
                pl.col("shortwave_radiation").mean().alias("shortwave_radiation_last_24_hours"),
                pl.col("direct_solar_radiation").mean().alias("direct_solar_radiation_last_24_hours"),
                pl.col("diffuse_radiation").mean().alias("diffuse_radiation_last_24_hours"),
            ]
        )
    )


def add_clients_id(data: pl.LazyFrame) -> pl.LazyFrame:
    client_ids_columns = ["county", "is_business", "product_type", "is_consumption"]

    data = (
        data.group_by(client_ids_columns)
        .len()
        .drop("len")
        .sort(client_ids_columns)
        .with_row_index(name="client_id")
        .join(other=data, how="inner", on=client_ids_columns)
    )
    return data


def load_data() -> tuple[pl.LazyFrame, ...]:
    train_data: pl.LazyFrame = load_train()
    gas_data: pl.LazyFrame = load_gas()
    electricity_data: pl.LazyFrame = load_electricity()
    clients_data: pl.LazyFrame = load_clients()
    weather_county_map: pl.LazyFrame = load_weather_station_mapping()
    weather_forecast_data: pl.LazyFrame = load_weather_forecast(weather_county_map)
    historical_weather_data: pl.LazyFrame = load_historical_weather(weather_county_map)
    return train_data, gas_data, electricity_data, clients_data, weather_forecast_data, historical_weather_data


def add_monthly_historical_data(data: pl.LazyFrame, historical_weather_data: pl.LazyFrame) -> pl.LazyFrame:
    columns_to_avg = list(
        set(historical_weather_data.columns).difference(
            [
                "county",
                "datetime",
                "data_block_id",
                "latitude_min",
                "latitude_max",
                "longitude_min",
                "longitude_max",
                "county_name",
            ]
        )
    )
    monthly_average_data: list[pl.LazyFrame] = []

    historical_weather_data = historical_weather_data.with_columns(
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").dt.month().alias("month"),
    )

    for year, month in historical_weather_data.select("year", "month").unique().collect().iter_rows():
        temp_df = historical_weather_data.filter((pl.col("year") < year) & (pl.col("month") == month))

        if temp_df.collect().is_empty():
            continue

        # get historical monthly average
        temp_df = temp_df.group_by("month").agg(
            *[pl.col(column).mean().alias(f"{column}_monthly_historical") for column in columns_to_avg]
        )

        # add year column
        monthly_average_data.append(temp_df.with_columns(pl.lit(year).alias("year")))

        # print(year, month)

    monthly_average_data: pl.LazyFrame = pl.concat(monthly_average_data)

    # join with train data
    data = data.join(monthly_average_data, on=["year", "month"], how="left")

    # fill null values with original values
    return data.with_columns(
        *[pl.col(f"{column}_monthly_historical").fill_null(pl.col(column)) for column in columns_to_avg]
    )


def create_dataset() -> pl.LazyFrame:
    train_data, gas_data, electricity_data, clients_data, weather_forecast_data, historical_weather_data = load_data()

    data: pl.LazyFrame = (
        train_data.join(
            other=clients_data,
            how="left",
            on=["county", "is_business", "product_type", "data_block_id"],
            suffix="_client",
        )
        .join(other=gas_data, on="data_block_id", how="left", suffix="_gas")
        .join(other=electricity_data, on=["datetime", "data_block_id"], how="left", suffix="_electricity")
    )

    data = add_clients_id(data)

    data = data.join(
        other=historical_weather_data, how="left", on=["county", "datetime", "data_block_id"], suffix="_measured"
    ).join(other=weather_forecast_data, how="left", on=["county", "datetime", "data_block_id"], suffix="_forecast")

    data = (
        data.join(
            other=create_lagged_weather_forecast(weather_forecast_data),
            on=["county", "datetime", "data_block_id"],
            how="left",
        )
        .join(
            other=create_lagged_historical_weather_last_week(historical_weather_data),
            on=["county", "datetime"],
            how="left",
        )
        .join(
            other=create_lagged_historical_weather_last_day(historical_weather_data),
            on=["county", "datetime"],
            how="left",
        )
    )

    # data = add_monthly_historical_data(data, historical_weather_data)
    return data


# ## Feature engineering


def create_revealed_target_features(data: pl.LazyFrame, lag_days: int = 7) -> pl.LazyFrame:
    revealed_targets = data.select("datetime", "client_id", "target")

    # Create revealed targets for all day lags
    for day_lag in range(2, lag_days + 1):
        data = data.join(
            revealed_targets.with_columns(pl.col("datetime") + pl.duration(days=day_lag)),
            how="left",
            on=["datetime", "client_id"],
            suffix=f"_{day_lag}_days_ago",
        )
    return data


def create_time_based_features(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.with_columns(
        (2 * np.pi * pl.col("hour") / 24).sin().cast(pl.Float32).alias("sin(hour)"),
        (2 * np.pi * pl.col("hour") / 24).cos().cast(pl.Float32).alias("cos(hour)"),
        (2 * np.pi * pl.col("weekday") / 7).sin().cast(pl.Float32).alias("sin(weekday)"),
        (2 * np.pi * pl.col("weekday") / 7).cos().cast(pl.Float32).alias("cos(weekday)"),
        (2 * np.pi * pl.col("month") / 12).sin().cast(pl.Float32).alias("sin(month)"),
        (2 * np.pi * pl.col("month") / 12).cos().cast(pl.Float32).alias("cos(month)"),
        pl.when(pl.col("datetime").dt.is_leap_year())
        .then(np.pi * pl.col("day_of_year") / 366)
        .otherwise(np.pi * pl.col("day_of_year") / 365)
        .sin()
        .cast(pl.Float32)
        .alias("sin(day_of_year)"),
        pl.when(pl.col("datetime").dt.is_leap_year())
        .then(np.pi * pl.col("day_of_year") / 366)
        .otherwise(np.pi * pl.col("day_of_year") / 365)
        .cos()
        .cast(pl.Float32)
        .alias("cos(day_of_year)"),
        # pl.col("datetime").dt.quarter().alias("quarter"),
        pl.col("date")
        .dt.strftime("%Y-%m-%d")
        .is_in([x.strftime("%Y-%m-%d") for x in ESTONIAN_HOLIDAYS])
        .alias("is_holiday"),
    ).drop(["hour", "day_of_year", "weekday"])


def map_product_type_to_string_values(data: pl.LazyFrame) -> pl.LazyFrame:
    product_type_map: dict[int, str] = {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}

    return data.with_columns(pl.col("product_type").replace(product_type_map, default="Unknown"))


def cast_data_to_32_bits(data: pl.LazyFrame) -> pl.LazyFrame:
    return data.with_columns(
        pl.col(pl.Int64).cast(pl.Int32),
        pl.col(pl.Float64).cast(pl.Float32),
    )


def add_noise_feature_for_training(data: pl.LazyFrame) -> pl.DataFrame:
    data = data.collect()
    return data.with_columns(pl.lit(np.random.normal(0, 1, size=data.shape[0])).alias("noise"))


def feature_engineer(data: pl.LazyFrame) -> pl.DataFrame:
    data = create_revealed_target_features(data, lag_days=7)
    data = create_time_based_features(data)
    data = map_product_type_to_string_values(data)
    data = cast_data_to_32_bits(data)
    return data


def convert_objects_columns_to_category(dataset: pd.DataFrame) -> pd.DataFrame:
    for col in dataset.columns:
        if dataset[col].dtype == "object":
            dataset[col] = dataset[col].astype("category")
    return dataset


# ## Train model function

# ## Training loop


def cross_validation_month_and_year(
    dataframe: pd.DataFrame, train_months: int = 3, test_months: int = 1, debug: bool = False
) -> Generator:
    for _, row in dataframe[["year", "month"]].drop_duplicates().sort_values(["year", "month"]).iterrows():
        current_date = datetime.date(row.year, row.month, 1) + relativedelta(months=1)
        start_train_date = current_date - relativedelta(months=train_months)
        end_train_date = current_date
        start_test_date = current_date
        end_test_date = current_date + relativedelta(months=test_months)

        try:
            train_index = dataframe.query(
                "(date >= @start_train_date & date < @end_train_date) | (year < @row.year & month == @row.month)"
            ).index
            test_index = dataframe.query("(date >= @start_test_date & date < @end_test_date)").index
        except KeyError:
            continue

        if debug:
            print(
                f"train date: {start_train_date} - {end_train_date - relativedelta(months=1)}test date: {start_test_date} - {end_test_date - relativedelta(months=1)}"
            )

        if len(train_index) == 0 or len(test_index) == 0:
            continue

        yield train_index, test_index, start_train_date, end_train_date, start_test_date, end_test_date


def split_train_test(data: pd.DataFrame, test_months: int = 6) -> tuple[list[int], ...]:
    max_dataset_date: datetime.date = data["date"].max()
    start_test_date: datetime.date = max_dataset_date - relativedelta(months=test_months)
    train_index = data.query("date < @start_test_date").index
    test_index = data.query("date >= @start_test_date").index
    return train_index, test_index


def train_model_cross_validation(dataframe: pd.DataFrame) -> list[lgb.LGBMRegressor]:
    models = []
    for i, (train_index, test_index, start_train_date, end_train_date, start_test_date, end_test_date) in enumerate(
        cross_validation_month_and_year(dataframe, train_months=15, test_months=6, debug=False)
    ):
        print(
            f"Split {i + 1} - train from {start_train_date} to {end_train_date} --- test from {start_test_date} to {end_test_date}"
        )
        models.append(train_model(dataframe, train_index, test_index, debug=False))
        break
    return models


def load_data_and_train_model(columns_to_drop: list[str], model_type: str) -> lgb.LGBMModel:
    model_type = model_type.lower()

    if model_type not in ["producer", "consumer"]:
        raise ValueError(f"Model type must be either 'producer' or 'consumer', given model type is: {model_type}")

    data = create_dataset()
    data = data.filter(pl.col("is_consumption") == MAP_MODEL_TYPE.get(model_type, 0)).drop(["is_consumption"])
    data = feature_engineer(data)
    data = data.drop_nulls()
    data = data.drop(columns_to_drop)
    data = add_noise_feature_for_training(data).to_pandas()
    data = convert_objects_columns_to_category(data)

    train_index, test_index = split_train_test(data=data)
    data = data.drop(columns=["date"])

    return train_model(dataframe=data, train_indexes=train_index, test_indexes=test_index)


def get_feature_importances_and_print_useless_columns(model: lgb.LGBMRegressor) -> pd.DataFrame:
    feature_importances = pd.DataFrame(
        {"feature": model.feature_name_, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importances["importance_perc"] = (
        feature_importances["importance"] / feature_importances["importance"].sum()
    ) * 100
    feature_importances = feature_importances.sort_values("importance_perc", ascending=False).reset_index(drop=True)
    feature_importances["importance_perc_cumulative"] = feature_importances["importance_perc"].cumsum()

    noise_importance = feature_importances.query("feature == 'noise'").importance.item()

    print(feature_importances[feature_importances["importance"] < noise_importance].feature.tolist())

    return feature_importances


def train_model(dataframe: pd.DataFrame, train_indexes: list[int], test_indexes: list[int]) -> lgb.LGBMModel:
    x_train, x_test = dataframe.loc[train_indexes], dataframe.loc[test_indexes]
    y_train, y_test = x_train.pop("target"), x_test.pop("target")
    eval_results = {}
    model = lgb.LGBMRegressor(
        random_state=SEED,
        num_leaves=31,
        n_estimators=20_000,
        subsample_for_bin=200_000,
        objective="huber",
        subsample=1.0,
        colsample_bytree=1.0,
        n_jobs=-1,
        linear_tree=True,
        verbosity=1,
        device="cpu",
        alpha=0.7,
    )

    model.fit(
        X=x_train,
        y=y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="mae",
        callbacks=[lgb.log_evaluation(), lgb.record_evaluation(eval_results), lgb.early_stopping(stopping_rounds=100)],
    )
    return model


# Train consumer model

columns_to_drop_producer = [
    "client_id",
    "data_block_id",
    # "date",
    "date_client",
    "date_gas",
    "datetime",
    "county",
    "county_name_forecast",
    "latitude_min_forecast",
    "latitude_max_forecast",
    "longitude_min_forecast",
    "longitude_max_forecast",
    # "hour",
    # "quarter",
    "year",
    "month",
]

columns_to_drop = columns_to_drop + [
    "direct_solar_radiation_last_24_hours",
    "cloudcover_low_forecast_last_day",
    "winddirection_10m_last_24_hours",
    "snowfall_forecast_last_day",
    "10_metre_u_wind_component",
    "product_type",
    "sin(day_of_year)",
    "rain_last_7_days",
    "surface_solar_radiation_downwards_forecast_last_day",
    "cloudcover_high_forecast_last_day",
    "gas_lowest_price_per_mwh",
    "cloudcover_high_last_7_days",
    "temperature_last_24_hours",
    "cloudcover_low",
    "dewpoint_forecast",
    "shortwave_radiation",
    "cos(day_of_year)",
    "dewpoint_forecast_last_day",
    "electricity_euros_per_mwh",
    "10_metre_v_wind_component",
    "cos(hour)",
    "winddirection_10m",
    "longitude_max",
    "longitude_min",
    "is_holiday",
    "shortwave_radiation_last_24_hours",
    "cloudcover_low_last_24_hours",
    "cloudcover_mid_last_24_hours",
    "shortwave_radiation_last_7_days",
    "temperature",
    "snowfall_last_7_days",
    "day",
    "sin(hour)",
    "temperature_forecast_last_day",
    "dewpoint_last_7_days",
    "surface_pressure",
    "temperature_forecast",
    "cloudcover_total_last_24_hours",
    "snowfall_last_24_hours",
    "rain_last_24_hours",
    "diffuse_radiation_last_7_days",
    "cloudcover_high",
    "winddirection_10m_last_7_days",
    "windspeed_10m_last_7_days",
    "windspeed_10m",
    "direct_solar_radiation",
    "temperature_last_7_days",
    "diffuse_radiation",
    "latitude_min",
    "latitude_max",
    "dewpoint",
    "cloudcover_high_forecast",
    "cloudcover_mid_forecast",
    "cloudcover_total_forecast",
    "cloudcover_high_last_24_hours",
    "windspeed_10m_last_24_hours",
    "dewpoint_last_24_hours",
    "snowfall_forecast",
    "gas_mean_price_per_mhw",
    "sin(month)",
    "rain",
    "cos(month)",
    "snowfall",
    "diffuse_radiation_last_24_hours",
    "cloudcover_mid",
]

columns_to_drop_consumer = [
    "client_id",
    "data_block_id",
    "date",
    "date_client",
    "date_gas",
    "datetime",
    "county",
    "county_name_forecast",
    "latitude_min_forecast",
    "latitude_max_forecast",
    "longitude_min_forecast",
    "longitude_max_forecast",
    "hour",
    "quarter",
    "year",
    "month",
]
consumer_model = load_data_and_train_model(model_type="consumer", columns_to_drop=columns_to_drop_consumer)

feature_importances_consumer = get_feature_importances_and_print_useless_columns(consumer_model)
