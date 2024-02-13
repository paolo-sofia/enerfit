import datetime
import pathlib
from typing import Any

import holidays
import numpy as np
import pandas as pd
import polars as pl
from dagster import asset
from dagster_etl.resources.model_dataset_config_resource import ModelDatasetConfigResource
from dagster_etl.utils.configs import load_data_preprocessing_config
from dagster_etl.utils.preprocessing import get_start_and_end_date_from_config

ESTONIAN_HOLIDAYS = list(holidays.country_holidays("EE", years=range(2021, 2026)).keys())

data_config: dict[str, Any] = load_data_preprocessing_config()


def create_lagged_weather_forecast(weather_forecast: pl.LazyFrame) -> pl.LazyFrame:
    """Creates lagged weather forecast features for the last day by aggregating the weather forecast data.

    Args:
        weather_forecast: The LazyFrame containing the weather forecast data.

    Returns:
        The LazyFrame with the lagged weather forecast features for the last day.
    """
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
    """Creates lagged historical weather features for the last week by aggregating the historical weather data.

    Args:
        historical_weather: The LazyFrame containing the historical weather data.

    Returns:
        The LazyFrame with the lagged historical weather features for the last week.
    """
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
    """Creates lagged historical weather features for the last day by aggregating the historical weather data.

    Args:
        historical_weather: The LazyFrame containing the historical weather data.

    Returns:
        The LazyFrame with the lagged historical weather features for the last day.

    Examples:
        >>> historical_weather = pl.DataFrame({
        ...     "datetime": ["2022-01-01 12:00:00", "2022-01-01 15:00:00", "2022-01-02 12:00:00"],
        ...     "county": ["A", "A", "B"],
        ...     "temperature": [10, 15, 20],
        ...     "dewpoint": [5, 10, 15],
        ...     "rain": [0, 0, 1],
        ...     "snowfall": [0, 0, 0],
        ...     "cloudcover_total": [50, 60, 70],
        ...     "cloudcover_low": [20, 30, 40],
        ...     "cloudcover_mid": [20, 30, 40],
        ...     "cloudcover_high": [10, 20, 30],
        ...     "windspeed_10m": [5, 10, 15],
        ...     "winddirection_10m": [180, 200, 220],
        ...     "shortwave_radiation": [100, 200, 300],
        ...     "direct_solar_radiation": [50, 100, 150],
        ...     "diffuse_radiation": [50, 100, 150]
        ... })
        >>> lagged_weather = create_lagged_historical_weather_last_day(historical_weather)
        >>> lagged_weather
                    datetime  county  temperature_last_24_hours  dewpoint_last_24_hours  rain_last_24_hours  snowfall_last_24_hours  cloudcover_total_last_24_hours  cloudcover_low_last_24_hours  cloudcover_mid_last_24_hours  cloudcover_high_last_24_hours  windspeed_10m_last_24_hours  winddirection_10m_last_24_hours  shortwave_radiation_last_24_hours  direct_solar_radiation_last_24_hours  diffuse_radiation_last_24_hours
        -------------------  ------  -------------------------  ----------------------  ------------------  -----------------------  -----------------------------  ----------------------------  ----------------------------  -----------------------------  ---------------------------  --------------------------------  ----------------------------------  -------------------------------------  --------------------------------
        2022-01-01 12:00:00  A                               10                       5                   0                        0                             50                            20                            20                             10                             5                              180                                 100                                    50
        2022-01-01 15:00:00  A                               12.5                     7.5                 0                        0                             55                            25                            25                             15                             7.5                            190                                 150                                    75
        2022-01-02 12:00:00  B                               20                       15                  1                        0                             70                            40                            40                             30                             15                             220                                 300                                   150
    """
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
    """Adds client IDs to the given LazyFrame based on specified columns.

    Args:
        data: The LazyFrame to which client IDs will be added.

    Returns:
        The LazyFrame with client IDs added.

    Examples:
        >>> data = pl.DataFrame({
        ...     "county": ["A", "B", "A", "B"],
        ...     "is_business": [True, False, True, False],
        ...     "product_type": ["X", "Y", "X", "Y"],
        ...     "is_consumption": [True, False, True, False],
        ...     "value": [10, 20, 30, 40]
        ... })
        >>> data_with_ids = add_clients_id(data)
        >>> data_with_ids
          client_id  county  is_business  product_type  is_consumption  value
        -----------  ------  -----------  ------------  --------------  -----
                  0  A       True         X             True               10
                  1  A       True         X             True               30
                  2  B       False        Y             False              20
                  3  B       False        Y             False              40
    """
    client_ids_columns = ["county", "is_business", "product_type", "is_consumption"]

    return (
        data.group_by(client_ids_columns)
        .len()
        .drop("len")
        .sort(client_ids_columns)
        .with_row_index(name="client_id")
        .join(other=data, how="inner", on=client_ids_columns)
    )


def load_data(
    start_date: datetime.date | None = None, end_date: datetime.date | None = None
) -> tuple[pl.LazyFrame, ...]:
    """Loads and returns multiple data sources as LazyFrames.

    Returns:
        A tuple of LazyFrames containing the loaded data from various sources.
    """
    root_dir: pathlib.Path = get_root_path()
    paths: dict[str, str] = data_config.get("paths", {})

    paths: dict[str, pathlib.Path] = {name: root_dir / path for name, path in paths.items()}

    train_data: pl.LazyFrame = load_train(pathlib.Path(paths.get("train")), start_date=start_date, end_date=end_date)
    gas_data: pl.LazyFrame = load_gas(pathlib.Path(paths.get("gas")), start_date=start_date, end_date=end_date)
    electricity_data: pl.LazyFrame = load_electricity(
        pathlib.Path(paths.get("electricity")), start_date=start_date, end_date=end_date
    )
    clients_data: pl.LazyFrame = load_clients(
        pathlib.Path(paths.get("clients")), start_date=start_date, end_date=end_date
    )
    weather_county_map: pl.LazyFrame = load_weather_station_mapping(pathlib.Path(paths.get("weather_station_map")))
    weather_forecast_data: pl.LazyFrame = load_weather_forecast(
        weather_county_map, pathlib.Path(paths.get("weather_forecast"), start_date=start_date, end_date=end_date)
    )
    historical_weather_data: pl.LazyFrame = load_historical_weather(
        weather_county_map, pathlib.Path(paths.get("historical_weather"), start_date=start_date, end_date=end_date)
    )
    return train_data, gas_data, electricity_data, clients_data, weather_forecast_data, historical_weather_data


def filter_dataframe_by_date(
    dataframe: pl.LazyFrame,
    start_date: datetime.date | None = None,
    end_date: datetime.date | None = None,
) -> pl.LazyFrame:
    filter_column = list(filter(lambda column: "date" in column, dataframe.columns))
    if not filter_column:
        print("No columns containing a date")


def load_and_join_data(
    train_data: pl.LazyFrame,
    gas_data: pl.LazyFrame,
    electricity_data: pl.LazyFrame,
    clients_data: pl.LazyFrame,
    weather_forecast_data: pl.LazyFrame,
    historical_weather_data: pl.LazyFrame,
) -> pl.LazyFrame:
    """Loads and joins multiple data sources to create a LazyFrame containing the combined data.

    Returns:
        The LazyFrame with the loaded and joined data.
    """
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

    return (
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


def create_revealed_target_features(data: pl.LazyFrame, lag_days: int = 7) -> pl.LazyFrame:
    """Creates revealed target features by joining the given LazyFrame with itself for a specified number of lag days.

    Args:
        data: The LazyFrame containing the data to create revealed target features from.
        lag_days: An optional integer representing the number of lag days. Default is 7.

    Returns:
        The LazyFrame with the revealed target features added.

    Examples:
        >>> data = pl.DataFrame({
        ...     "datetime": ["2022-01-01", "2022-01-02", "2022-01-03"],
        ...     "client_id": [1, 2, 3],
        ...     "target": [10, 20, 30]
        ... })
        >>> revealed_features = create_revealed_target_features(data, lag_days=2)
        >>> revealed_features
          datetime  client_id  target    target_2_days_ago
        ----------  ----------  ------  -------------------
        2022-01-01           1      10                 null
        2022-01-02           2      20                 null
        2022-01-03           3      30                   10
    """
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
    """Creates time-based features from the given LazyFrame.

    Args:
        data: The LazyFrame containing the data to create time-based features from.

    Returns:
        The LazyFrame with the time-based features added.

    Examples:
        >>> data = pl.DataFrame({
        ...     "datetime": ["2022-01-01 12:00:00", "2022-01-02 08:00:00", "2022-01-03 16:00:00"],
        ...     "hour": [12, 8, 16],
        ...     "weekday": [5, 6, 0],
        ...     "month": [1, 1, 1],
        ...     "day_of_year": [1, 2, 3],
        ...     "date": ["2022-01-01", "2022-01-02", "2022-01-03"]
        ... })
        >>> time_features = create_time_based_features(data)
        >>> time_features
                    datetime  sin(hour)  cos(hour)  sin(weekday)  cos(weekday)  sin(month)  cos(month)  sin(day_of_year)  cos(day_of_year)  is_holiday
        -------------------  ----------  ----------  ------------  ------------  -----------  -----------  ----------------  ----------------  -----------
        2022-01-01 12:00:00    0.000000    1.000000      0.974928     -0.222521          0.0          1.0               0.0               1.0        False
        2022-01-02 08:00:00    0.866025    0.500000      0.433884     -0.900969          0.0          1.0               0.0               1.0        False
        2022-01-03 16:00:00   -0.866025   -0.500000      0.781831      0.623490          0.0          1.0               0.0               1.0        False
    """
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
    """Maps integer values in the 'product_type' column of the given LazyFrame to their corresponding string values.

    Args:
        data: The LazyFrame containing the data to be processed.

    Returns:
        The LazyFrame with the 'product_type' column replaced by string values according to the mapping.
    """
    product_type_map: dict[int, str] = {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}

    return data.with_columns(pl.col("product_type").replace(product_type_map, default="Unknown"))


def cast_data_to_32_bits(data: pl.LazyFrame) -> pl.LazyFrame:
    """Casts the data in the given LazyFrame to 32-bit integer and float types.

    Args:
        data: The LazyFrame containing the data to be cast.

    Returns:
        The LazyFrame with the data cast to 32-bit integer and float types.
    """
    return data.with_columns(
        pl.col(pl.Int64).cast(pl.Int32),
        pl.col(pl.Float64).cast(pl.Float32),
    )


def add_noise_feature_for_training(data: pl.LazyFrame, seed: int = 42) -> pl.DataFrame:
    """Adds a noise feature to the given LazyFrame for training purposes.

    Args:
        data: The LazyFrame containing the data to which the noise feature will be added.
        seed: An optional integer seed value for the random number generator. Default is 42.

    Returns:
        The DataFrame with the noise feature added.

    Examples:
        >>> data = pl.DataFrame({
        ...     "feature1": [1, 2, 3, 4],
        ...     "feature2": [5, 6, 7, 8]
        ... })
        >>> noisy_data = add_noise_feature_for_training(data, seed=123)
        >>> noisy_data
          feature1  feature2     noise
        ---------  ---------  --------
                1          5  0.392938
                2          6 -0.427721
                3          7  0.865408
                4          8 -0.303711
    """
    data: pl.DataFrame = data.collect()
    return data.with_columns(pl.lit(np.random.default_rng(seed=seed).normal(0, 1, size=data.shape[0])).alias("noise"))


def feature_engineer(data: pl.LazyFrame) -> pl.LazyFrame:
    """Performs feature engineering on the given LazyFrame.

     It creates revealed target features, time-based features, mapping product types to string values,
        and casting data to 32-bit integer and float types.

    Args:
        data: The LazyFrame containing the data to be feature engineered.

    Returns:
        The feature engineered LazyFrame.

    Examples:
        >>> data = pl.DataFrame({
        ...     "feature1": [1, 2, 3, 4],
        ...     "feature2": [5, 6, 7, 8]
        ... })
        >>> engineered_data = feature_engineer(data)
        >>> engineered_data
          feature1  feature2  product_type
        ---------  ---------  -------------
                1          5  Combined
                2          6  Fixed
                3          7  General service
                4          8  Spot
    """
    data = create_revealed_target_features(data, lag_days=data_config.get("preprocessing", {}).get("lag_days", 7))
    data = create_time_based_features(data)
    data = map_product_type_to_string_values(data)
    return cast_data_to_32_bits(data)


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


@asset(
    name="training_dataset",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["model", "dataset"],
    compute_kind="polars",
)
def create_train_dataset(
    train: pl.LazyFrame,
    clients: pl.LazyFrame,
    electricity: pl.LazyFrame,
    gas: pl.LazyFrame,
    historical_weather: pl.LazyFrame,
    weather_forecast: pl.LazyFrame,
    model_data_resource: ModelDatasetConfigResource,
) -> pd.DataFrame:
    """Create the training dataset for the model.

    Args:
        train (pl.LazyFrame): The lazy frame containing the training data.
        clients (pl.LazyFrame): The lazy frame containing the clients data.
        electricity (pl.LazyFrame): The lazy frame containing the electricity data.
        gas (pl.LazyFrame): The lazy frame containing the gas data.
        historical_weather (pl.LazyFrame): The lazy frame containing the historical weather data.
        weather_forecast (pl.LazyFrame): The lazy frame containing the weather forecast data.
        model_data_resource (ModelDatasetConfigResource): The configuration for the model dataset.

    Returns:
        pd.DataFrame: The created training dataset.

    Note:
        This function is decorated with `@asset` to define the asset properties.

    Examples:
        >>> create_train_dataset(train, clients, electricity, gas, historical_weather, weather_forecast, config)
        pd.DataFrame(...)
    """
    start_date, end_date = get_start_and_end_date_from_config(
        model_data_resource.train_months, model_data_resource.test_months
    )

    train = filter_dataframe_by_date(train, start_date, end_date)
    clients = filter_dataframe_by_date(clients, start_date, end_date)
    gas = filter_dataframe_by_date(gas, start_date, end_date)
    electricity = filter_dataframe_by_date(electricity, start_date, end_date)
    weather_forecast = filter_dataframe_by_date(weather_forecast, start_date, end_date)
    historical_weather = filter_dataframe_by_date(historical_weather, start_date, end_date)

    data = load_and_join_data(
        train_data=train,
        clients_data=clients,
        gas_data=gas,
        electricity_data=electricity,
        weather_forecast_data=weather_forecast,
        historical_weather_data=historical_weather,
    )
    # data = data.filter(
    #     (pl.col("is_consumption") == MAP_MODEL_TYPE.get(model_type, 0)) & (pl.col("is_business") == int(is_business))
    # ).drop(["is_consumption", "is_business"])
    data = feature_engineer(data)
    data = data.drop_nulls()

    # if columns:
    #     data = data.select(columns)

    if model_data_resource.add_noise_column:
        data = add_noise_feature_for_training(data).to_pandas()
    else:
        data.collect().to_pandas()

    return convert_objects_columns_to_category(data)
