import pathlib

import polars as pl

DATE_FORMAT: str = "%Y-%m-%d"
DATETIME_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def load_clients(clients_path: pathlib.Path) -> pl.LazyFrame:
    """Load the clients data from a CSV file and perform column type conversions.

    Args:
        clients_path (pathlib.Path): The path to the clients CSV file.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded clients data with converted column types.

    Examples:
        >>> clients_path = pathlib.Path("clients.csv")
        >>> clients_data = load_clients(clients_path)
        >>> # Perform operations on the loaded clients data
    """
    clients: pl.LazyFrame = pl.scan_csv(clients_path)
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


def load_electricity(electricity_path: pathlib.Path) -> pl.LazyFrame:
    """Load the electricity data from a CSV file and perform column type conversions.

    Args:
        electricity_path (pathlib.Path): The path to the electricity CSV file.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded electricity data with converted column types.

    Examples:
        >>> electricity_path = pathlib.Path("electricity.csv")
        >>> electricity_data = load_electricity(electricity_path)
        >>> # Perform operations on the loaded electricity data
    """
    electricity: pl.LazyFrame = pl.scan_csv(electricity_path).drop(["origin_date"])
    return electricity.with_columns(
        [
            pl.col("forecast_date").str.to_datetime(DATETIME_FORMAT) + pl.duration(days=1),
            pl.col("euros_per_mwh").cast(pl.Float32),
            pl.col("data_block_id").cast(pl.Int16),
        ]
    ).rename({"forecast_date": "datetime", "euros_per_mwh": "electricity_euros_per_mwh"})


def load_gas(gas_path: pathlib.Path) -> pl.LazyFrame:
    """Load the gas data from a CSV file and perform column type conversions.

    Args:
        gas_path (pathlib.Path): The path to the gas CSV file.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded gas data with converted column types.

    Examples:
        >>> gas_path = pathlib.Path("gas.csv")
        >>> gas_data = load_gas(gas_path)
        >>> # Perform operations on the loaded gas data
    """
    gas: pl.LazyFrame = pl.scan_csv(gas_path).drop(["origin_date"])
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


def load_weather_station_mapping(weather_station_county_map_path: pathlib.Path) -> pl.LazyFrame:
    """Load the weather station to county mapping data from a CSV file and perform data transformations.

    Args:
        weather_station_county_map_path (pathlib.Path): The path to the weather station to county mapping CSV file.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded weather station to county mapping data.

    Examples:
        >>> weather_station_county_map_path = pathlib.Path("weather_station_mapping.csv")
        >>> weather_station_mapping_data = load_weather_station_mapping(weather_station_county_map_path)
        >>> # Perform operations on the loaded weather station mapping data
    """
    weather_station_county_mapping: pl.LazyFrame = pl.scan_csv(weather_station_county_map_path)
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
    )


def load_weather_forecast(
    weather_station_county_mapping: pl.LazyFrame, weather_forecast_path: pathlib.Path
) -> pl.LazyFrame:
    """Load the weather forecast data from a CSV file and perform data transformations.

    Args:
        weather_station_county_mapping (pl.LazyFrame): The weather station to county mapping data.
        weather_forecast_path (pathlib.Path): The path to the weather forecast CSV file.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded weather forecast data with transformed columns.

    Examples:
        >>> weather_station_county_mapping = load_weather_station_mapping(weather_station_county_map_path)
        >>> weather_forecast_path = pathlib.Path("weather_forecast.csv")
        >>> weather_forecast_data = load_weather_forecast(weather_station_county_mapping, weather_forecast_path)
        >>> # Perform operations on the loaded weather forecast data
    """
    weather_forecast: pl.LazyFrame = (
        pl.scan_csv(weather_forecast_path).drop(["origin_datetime"]).rename({"forecast_datetime": "datetime"})
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

    return weather_forecast.group_by("county", "datetime", "data_block_id").agg(
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


def load_historical_weather(
    weather_station_county_mapping: pl.LazyFrame, historical_weather_path: pathlib.Path
) -> pl.LazyFrame:
    """Load the historical weather data from a CSV file and perform data transformations.

    Args:
        weather_station_county_mapping (pl.LazyFrame): The weather station to county mapping data.
        historical_weather_path (pathlib.Path): The path to the historical weather CSV file.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded historical weather data with transformed columns.

    Examples:
        >>> weather_station_county_mapping = load_weather_station_mapping(weather_station_county_map_path)
        >>> historical_weather_path = pathlib.Path("historical_weather.csv")
        >>> historical_weather_data = load_historical_weather(weather_station_county_mapping, historical_weather_path)
        >>> # Perform operations on the loaded historical weather data
    """
    historical_weather: pl.LazyFrame = pl.scan_csv(historical_weather_path)
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
    return historical_weather.with_columns(
        pl.when(pl.col("datetime").dt.hour() < 11)
        .then(pl.col("datetime") + pl.duration(days=1))
        .otherwise(pl.col("datetime") + pl.duration(days=2))
    )


def load_train(train_path: pathlib.Path) -> pl.LazyFrame:
    """Load the training data from a CSV file and perform data transformations.

    Args:
        train_path (pathlib.Path): The path to the training data CSV file.

    Returns:
        pl.LazyFrame: A lazy frame containing the loaded training data with transformed columns.

    Examples:
        >>> train_path = pathlib.Path("train.csv")
        >>> train_data = load_train(train_path)
        >>> # Perform operations on the loaded training data
    """
    train: pl.LazyFrame = pl.scan_csv(train_path)

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
        pl.col("datetime").dt.year().alias("year"),
        pl.col("datetime").cast(pl.Date).alias("date"),
        pl.col("datetime").dt.month().alias("month"),
        pl.col("datetime").dt.weekday().alias("weekday"),
        pl.col("datetime").dt.day().alias("day"),
        pl.col("datetime").dt.ordinal_day().alias("day_of_year"),
        pl.col("datetime").dt.hour().alias("hour"),
    )
