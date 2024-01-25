# Deve essere un cron job che parte a mezzanotte e salva tutti i dati giornalieri sul datalake, quindi farlo con airflow o qualcosa di simile
import datetime
import pathlib
from dataclasses import dataclass
from typing import Any

import polars as pl
import pytz

from consumers.utils import (
    commit_processed_messages_from_topic_from_day,
    fetch_data_at_day,
    load_config_and_get_date_to_process,
    save_to_datalake_partition_by_date,
)
from utils.polars import (
    cast_column_to_16_bits_numeric,
    cast_column_to_32_bits_numeric,
    cast_column_to_datetime,
)

ENCODING: str = "utf-8"
BASE_PATH: pathlib.Path = pathlib.Path()


@dataclass
class WeatherColumns:
    """A dataclass representing the columns of weather data.

    Args:
        datetime (str): The column name for datetime.
        temperature (str): The column name for temperature.
        dewpoint (str): The column name for dewpoint.
        rain (str): The column name for rain.
        snowfall (str): The column name for snowfall.
        surface_pressure (str): The column name for surface pressure.
        cloudcover_total (str): The column name for total cloud cover.
        cloudcover_low (str): The column name for low cloud cover.
        cloudcover_mid (str): The column name for mid-level cloud cover.
        cloudcover_high (str): The column name for high cloud cover.
        windspeed_10m (str): The column name for wind speed at 10m.
        winddirection_10m (str): The column name for wind direction at 10m.
        shortwave_radiation (str): The column name for shortwave radiation.
        direct_solar_radiation (str): The column name for direct solar radiation.
        diffuse_radiation (str): The column name for diffuse radiation.
        latitude (str): The column name for latitude.
        longitude (str): The column name for longitude.
    """

    datetime: str = "datetime"
    temperature: str = "temperature"
    dewpoint: str = "dewpoint"
    rain: str = "rain"
    snowfall: str = "snowfall"
    surface_pressure: str = "surface_pressure"
    cloudcover_total: str = "cloudcover_total"
    cloudcover_low: str = "cloudcover_low"
    cloudcover_mid: str = "cloudcover_mid"
    cloudcover_high: str = "cloudcover_high"
    windspeed_10m: str = "windspeed_10m"
    winddirection_10m: str = "winddirection_10m"
    shortwave_radiation: str = "shortwave_radiation"
    direct_solar_radiation: str = "direct_solar_radiation"
    diffuse_radiation: str = "diffuse_radiation"
    latitude: str = "latitude"
    longitude: str = "longitude"
    data_block_id: str = "data_block_id"


def update_processed_gas_table(is_processed: bool, day: datetime.date) -> bool:
    """Updates the processed gas table for a specific day.

    Args:
        is_processed (bool): The flag indicating if the gas are processed or not.
        day (datetime.date): The specific day to update the processed gas table for.

    Returns:
        bool: True if the processed gas table was successfully updated, False otherwise.
    """
    pass


def preprocess_gas_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses gas data by casting columns to specific data types.

    Args:
        data (pl.DataFrame): The gas data to be preprocessed.

    Returns:
        pl.DataFrame: The preprocessed gas data.

    Examples:
        >>> data = pl.DataFrame(...)
        >>> preprocess_gas_data(data)
        pl.DataFrame(...)
    """
    data = cast_column_to_datetime(
        data=data,
        column_names=[WeatherColumns.datetime],
        datetime_format="%Y-%m-%d %H:%M:%S",
        timezone=pytz.timezone("Europe/Tallin"),
    )
    data = cast_column_to_32_bits_numeric(data)
    return cast_column_to_16_bits_numeric(data)


def main() -> None:
    """Runs the main function to process gas data.

    Returns:
        None
    """
    config: dict[str, Any]
    yesterday: datetime.date
    config, yesterday = load_config_and_get_date_to_process()

    # Prenditi tutti i dati di ieri da kafka
    historical_weather: pl.DataFrame = fetch_data_at_day(
        config=config,
        day=yesterday,
        filter_col=WeatherColumns.datetime,
    )

    if historical_weather.is_empty():
        print(f"No data from {yesterday}")
        update_processed_gas_table(True, yesterday)
        return

    historical_weather = preprocess_gas_data(historical_weather)

    if data_saved := save_to_datalake_partition_by_date(
        data=historical_weather, day=yesterday, base_path=BASE_PATH, filename="historical_weather"
    ):
        commit_processed_messages_from_topic_from_day(day=yesterday, filter_col=WeatherColumns.datetime)

    update_processed_gas_table(data_saved, yesterday)


if __name__ == "__main__":
    main()
