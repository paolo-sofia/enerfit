# Deve essere un cron job che parte a mezzanotte e salva tutti i dati giornalieri sul datalake, quindi farlo con airflow o qualcosa di simile
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl
import pytz
from kafka3 import KafkaConsumer

from utils.configs import load_config
from utils.io import create_base_output_path_paritioned_by_date
from utils.kafka import (
    create_consumer_and_seek_to_last_committed_message,
    create_kafka_consumer,
    kafka_consumer_seek_to_last_committed_message,
)
from utils.polars import cast_column_to_16_bits_numeric, cast_column_to_datetime

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


def fetch_electricity_at_day(config: dict[str, Any], day: datetime.date) -> pl.DataFrame:
    """Fetches electricity data for a specific day from a Kafka topic.

    Args:
        config (dict): A dictionary containing the Kafka consumer configuration.
        day (datetime.date): The specific day to fetch electricity data for.

    Returns:
        pl.DataFrame: The electricity data for the specified day.
    """
    consumer: KafkaConsumer = create_consumer_and_seek_to_last_committed_message(config)

    day_messages: list[dict] = []
    for message in consumer:
        message_dict: dict = json.loads(message.decode(ENCODING))
        if message_dict.get(ElectricityColumns.forecast_date) == day:
            day_messages.append(message_dict)

    data = pl.from_dicts(day_messages)
    data = cast_column_to_datetime(
        data,
        column_names=[ElectricityColumns.forecast_date, ElectricityColumns.origin_date],
        datetime_format="%Y-%m-%d %H:%M:%S",
        timezone=pytz.timezone("Europe/Tallin"),
    )
    return data.filter(pl.col(ElectricityColumns.forecast_date) == day)


def commit_processed_messages_from_topic(day: datetime.date, encoding: str = "utf-8") -> None:
    """Commits processed messages from a Kafka topic for a specific day.

    Args:
        day (datetime.date): The specific day to filter the messages.
        encoding (str, optional): The encoding used for deserializing messages. Defaults to "utf-8".

    Returns:
        None
    """
    consumer: KafkaConsumer = create_kafka_consumer({})
    consumer = kafka_consumer_seek_to_last_committed_message(consumer)

    for message in consumer:
        if json.loads(message.decode(encoding)).get("date") == day:
            consumer.commit()


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


def main() -> None:
    """Runs the main function to process electricity data.

    Returns:
        None
    """
    config: dict[str, Any] = load_config(pathlib.Path(__file__).parent / "config.toml")

    yesterday: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Tallin")).date() - datetime.timedelta(
        days=1
    )

    # Prenditi tutti i dati di ieri da kafka
    electricity: pl.DataFrame = fetch_electricity_at_day(config=config, day=yesterday)
    if electricity.is_empty():
        print(f"No data from {yesterday}")
        update_processed_electricity_table(True, yesterday)

    electricity = cast_column_to_16_bits_numeric(electricity)

    if data_saved := save_electricity_to_datalake(electricity, yesterday):
        commit_processed_messages_from_topic(yesterday)

    update_processed_electricity_table(data_saved, yesterday)


if __name__ == "__main__":
    main()
