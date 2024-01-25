# Deve essere un cron job che parte a mezzanotte e salva tutti i dati giornalieri sul datalake, quindi farlo con airflow o qualcosa di simile
import datetime
import pathlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl
import pytz

from consumers.utils import (
    commit_processed_messages_from_topic_from_day,
    fetch_data_at_day,
    filter_data_by_date,
    load_config_and_get_date_to_process,
    save_to_datalake_partition_by_date,
)
from utils.io import create_base_output_path_paritioned_by_date
from utils.polars import cast_column_to_16_bits_numeric, cast_column_to_32_bits_numeric, cast_column_to_datetime

ENCODING: str = "utf-8"
BASE_PATH: pathlib.Path = pathlib.Path()


@dataclass
class ElectricityColumns:
    """A dataclass representing the column names for the Electricity dataset.

    Attributes:
        forecast_date (str): The column name for the forecast date.
        euros_per_mwh (str): The column name for the euros per MWh.
        origin_date (str): The column name for the origin date.
        data_block_id (str): The column name for the data block ID.
    """

    forecast_date: str = "forecast_date"
    euros_per_mwh: str = "euros_per_mwh"
    origin_date: str = "origin_date"
    data_block_id: str = "data_block_id"


def save_electricity_to_datalake(data: pd.DataFrame, day: datetime.date) -> bool:
    """Saves electricity data to the datalake.

    This function takes a DataFrame of electricity data and a specific day, and saves the data to the datalake. The data is
        saved as a parquet file at the corresponding output path based on the given day.

    Args:
        data (pd.DataFrame): The DataFrame of electricity data to be saved.
        day (datetime.date): The specific day associated with the electricity data.

    Returns:
        bool: True if the data is successfully saved, False otherwise.
    """
    output_path: pathlib.Path = create_base_output_path_paritioned_by_date(
        base_path=BASE_PATH, date=day, file_name="electricity"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = pl.from_pandas(data)
        data.write_parquet(output_path)
        return True
    except Exception as e:
        print(e)
        return False


def update_processed_electricity_table(is_processed: bool, day: datetime.date) -> bool:
    """Updates the processed electricity table for a specific day.

    Args:
        is_processed (bool): The flag indicating if the electricity are processed or not.
        day (datetime.date): The specific day to update the processed electricity table for.

    Returns:
        bool: True if the processed electricity table was successfully updated, False otherwise.
    """
    pass


def preprocess_electricity_data(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses electricity data by casting columns to specific data types.

    Args:
        data (pl.DataFrame): The electricity data to be preprocessed.

    Returns:
        pl.DataFrame: The preprocessed electricity data.
    """
    data = cast_column_to_datetime(
        data=data,
        column_names=[ElectricityColumns.forecast_date, ElectricityColumns.origin_date],
        datetime_format="%Y-%m-%d %H:%M:%S",
        timezone=pytz.timezone("Europe/Tallin"),
    )
    data = cast_column_to_32_bits_numeric(data)
    return cast_column_to_16_bits_numeric(data)


def main() -> None:
    """Runs the main function to process electricity data.

    Returns:
        None
    """
    config: dict[str, Any]
    yesterday: datetime.date
    config, yesterday = load_config_and_get_date_to_process()

    electricity: pl.DataFrame = fetch_data_at_day(
        config=config, day=yesterday, filter_col=ElectricityColumns.forecast_date
    )

    if electricity.is_empty():
        print(f"No data from {yesterday}")
        update_processed_electricity_table(True, yesterday)
        return

    electricity = preprocess_electricity_data(electricity)
    electricity = filter_data_by_date(
        data=electricity, filter_col=ElectricityColumns.forecast_date, filter_date=yesterday
    )

    if data_saved := save_to_datalake_partition_by_date(
        data=electricity, day=yesterday, base_path=BASE_PATH, filename="electricity"
    ):
        commit_processed_messages_from_topic_from_day(day=yesterday, filter_col=ElectricityColumns.forecast_date)

    update_processed_electricity_table(data_saved, yesterday)


if __name__ == "__main__":
    main()
