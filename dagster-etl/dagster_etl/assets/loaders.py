import datetime

import polars as pl
import pytz
from dagster import asset

from ..resources.data_path_resource import DataPathResource

DATE_FORMAT: str = "%Y-%m-%d"
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"
today: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Rome")).today()
year: str = str(today.year)
month: str = str(today.month)
day: str = str(today.day)


def add_data_block_id(dataframe: pl.LazyFrame) -> pl.LazyFrame:
    """Add a data block ID column to the given lazy frame if it does not already exist.

    Args:
        dataframe: The input lazy frame.

    Returns:
        pl.LazyFrame: The lazy frame with the data block ID column added.
    """
    if "data_block_id" not in dataframe.columns:
        dataframe = dataframe.with_columns(pl.lit(0).alias("data_block_id"))

    return dataframe


@asset(
    name="clients",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "clients", year, month, day],
    compute_kind="polars",
)
def load_clients(data_path_resource: DataPathResource) -> pl.LazyFrame:
    """Load client data from the given data path resource.

    This function loads client data from the specified data path resource, applies transformations to the data,
    and returns a lazy frame containing the transformed client data.

    Args:
        data_path_resource (DataPathResource): The data path resource representing the location of the client data.

    Returns:
        pl.LazyFrame: A lazy frame containing the transformed client data.
    """
    clients: pl.LazyFrame = pl.scan_csv(data_path_resource.clients)
    clients = add_data_block_id(clients)
    return clients.with_columns(
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


@asset(
    name="electricity",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "electricity", year, month, day],
    compute_kind="polars",
)
def load_electricity(data_path_resource: DataPathResource) -> pl.LazyFrame:
    """Load electricity data from the given data path resource.

    This function loads electricity data from the specified data path resource, applies transformations to the data,
    and returns a lazy frame containing the transformed electricity data.

    Args:
        data_path_resource (DataPathResource): The data path resource representing the location of the client data.

    Returns:
        pl.LazyFrame: A lazy frame containing the transformed electricity data.
    """
    electricity: pl.LazyFrame = pl.scan_csv(data_path_resource.electricity).drop(["origin_date"])
    electricity = add_data_block_id(electricity)
    return electricity.with_columns(
        [
            pl.col("forecast_date").str.to_datetime(DATETIME_FORMAT) + pl.duration(days=1),
            pl.col("euros_per_mwh").cast(pl.Float32),
            pl.col("data_block_id").cast(pl.Int16),
        ]
    ).rename({"forecast_date": "datetime", "euros_per_mwh": "electricity_euros_per_mwh"})


@asset(
    name="gas",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "gas", year, month, day],
    compute_kind="polars",
)
def load_gas(data_path_resource: DataPathResource) -> pl.LazyFrame:
    gas: pl.LazyFrame = pl.scan_csv(data_path_resource.gas).drop(["origin_date"])
    gas = add_data_block_id(gas)
    return gas.with_columns(
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


@asset(
    name="weather_station_county_mapping",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw"],
    compute_kind="polars",
)
def load_weather_station_mapping(data_path_resource: DataPathResource) -> pl.DataFrame:
    """Load the weather station to county mapping data from a CSV file and perform data transformations.

    Args:
        data_path_resource (DataPathResource): The data path resource representing the location of the client data.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded weather station to county mapping data.

    Examples:
        >>> weather_station_county_map_path = pathlib.Path("weather_station_mapping.csv")
        >>> weather_station_mapping_data = load_weather_station_mapping(weather_station_county_map_path)
        >>> # Perform operations on the loaded weather station mapping data
    """
    weather_station_county_mapping: pl.LazyFrame = pl.scan_csv(data_path_resource.weather_station_map)
    weather_station_county_mapping = weather_station_county_mapping.with_columns(
        [
            pl.col("longitude").cast(pl.Float32).round(decimals=2),
            pl.col("latitude").cast(pl.Float32).round(decimals=2),
            pl.col("county").cast(pl.Int8).fill_null(-1),
            pl.col("county_name").fill_null("Unknown"),
        ]
    )

    return weather_station_county_mapping.join(
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
    ).collect()


@asset(
    name="weather_forecast",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "weather_forecast", year, month, day],
    compute_kind="polars",
)
def load_weather_forecast(
    weather_station_county_mapping: pl.LazyFrame, data_path_resource: DataPathResource
) -> pl.DataFrame:
    """Load weather forecast data from the given CSV file path and apply optional filters based on start and end dates.

    Args:
        weather_station_county_mapping (pl.LazyFrame): A lazy frame containing the mapping between weather stations
            and counties.
        data_path_resource (DataPathResource): The data path resource representing the location of the client data.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded weather forecast data.
    """
    weather_forecast: pl.LazyFrame = (
        pl.scan_csv(data_path_resource.weather_forecast)
        .drop(["origin_datetime"])
        .rename({"forecast_datetime": "datetime"})
    )

    weather_forecast = add_data_block_id(weather_forecast)
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

    return (
        weather_forecast.group_by("county", "datetime", "data_block_id")
        .agg(
            pl.col("hours_ahead").mean().cast(pl.Float32),
            pl.col("temperature").mean().cast(pl.Float32),
            pl.col("dewpoint").mean().cast(pl.Float32),
            pl.col("cloudcover_high").mean().cast(pl.Float32),
            pl.col("cloudcover_low").mean().cast(pl.Float32),
            pl.col("cloudcover_mid").mean().cast(pl.Float32),
            pl.col("cloudcover_total").mean().cast(pl.Float32),
            pl.col("10_metre_u_wind_component").mean().cast(pl.Float32),
            pl.col("10_metre_v_wind_component").mean().cast(pl.Float32),
            pl.col("direct_solar_radiation").mean().cast(pl.Float32),
            pl.col("surface_solar_radiation_downwards").mean().cast(pl.Float32),
            pl.col("snowfall").mean().cast(pl.Float32),
            pl.col("total_precipitation").mean().cast(pl.Float32),
            pl.col("latitude_min").first(),
            pl.col("latitude_max").first(),
            pl.col("longitude_min").first(),
            pl.col("longitude_max").first(),
            pl.col("county_name").first(),
        )
        .collect()
    )


@asset(
    name="historical_weather",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "historical_weather", year, month, day],
    compute_kind="polars",
)
def load_historical_weather(
    weather_station_county_mapping: pl.LazyFrame, data_path_resource: DataPathResource
) -> pl.DataFrame:
    """Load and preprocess historical weather data from the given CSV file.

    Args:
        data_path_resource (DataPathResource): The data path resource representing the location of the client data.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded historical weather data.
    """
    historical_weather: pl.LazyFrame = pl.scan_csv(data_path_resource.historical_weather)

    historical_weather = add_data_block_id(historical_weather)

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
        pl.col("temperature").mean().cast(pl.Float32),
        pl.col("dewpoint").mean().cast(pl.Float32),
        pl.col("rain").mean().cast(pl.Float32),
        pl.col("snowfall").mean().cast(pl.Float32),
        pl.col("surface_pressure").mean().cast(pl.Float32),
        pl.col("cloudcover_total").mean().cast(pl.Float32),
        pl.col("cloudcover_low").mean().cast(pl.Float32),
        pl.col("cloudcover_mid").mean().cast(pl.Float32),
        pl.col("cloudcover_high").mean().cast(pl.Float32),
        pl.col("windspeed_10m").mean().cast(pl.Float32),
        pl.col("winddirection_10m").mean().cast(pl.Float32),
        pl.col("shortwave_radiation").mean().cast(pl.Float32),
        pl.col("direct_solar_radiation").mean().cast(pl.Float32),
        pl.col("diffuse_radiation").mean().cast(pl.Float32),
        pl.col("latitude_min").first(),
        pl.col("latitude_max").first(),
        pl.col("longitude_min").first(),
        pl.col("longitude_max").first(),
        pl.col("county_name").first(),
    )

    # Test set has 1 day offset for hour<11 and 2 day offset for hour>11
    return historical_weather.with_columns(
        pl.when(pl.col("datetime").dt.hour() < 11)
        .then(pl.col("datetime") + pl.duration(days=1))
        .otherwise(pl.col("datetime") + pl.duration(days=2))
    ).collect()


@asset(
    name="train",
    io_manager_key="polars_parquet_io_manager",
    key_prefix=["raw", "train", year, month, day],
    compute_kind="polars",
)
def load_train(data_path_resource: DataPathResource) -> pl.LazyFrame:
    """Load training data from the given CSV file path and apply optional filters based on start and end dates.

    Args:
        data_path_resource (DataPathResource): The data path resource representing the location of the client data.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded training data.
    """
    train: pl.LazyFrame = pl.scan_csv(data_path_resource.train)

    train = add_data_block_id(train)

    train = train.drop(["prediction_unit_id", "row_id"]).with_columns(
        pl.col("datetime").str.to_datetime(DATETIME_FORMAT),
        pl.col("is_business").cast(pl.Int8),
        pl.col("product_type").cast(pl.Int8),
        pl.col("target").cast(pl.Float32),
        pl.col("is_consumption").cast(pl.Int8),
        pl.col("county").cast(pl.Int8),
        pl.col("data_block_id").cast(pl.Int16),
    )

    return train.with_columns(
        pl.col("datetime").cast(pl.Date).alias("date"),
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.day().alias("day"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        pl.col("datetime").dt.ordinal_day().alias("day_of_year"),
        pl.col("datetime").dt.hour().alias("hour"),
    )
