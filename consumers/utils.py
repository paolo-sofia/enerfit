import datetime
import json
import pathlib
from typing import Any

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


def fetch_data_at_day(
    config: dict[str, Any], day: datetime.date, filter_col: str, encoding: str = "utf-8"
) -> pl.DataFrame:
    """Fetches energy data for a specific day from a Kafka topic.

    Args:
        config (dict): A dictionary containing the Kafka consumer configuration.
        day (datetime.date): The specific day to fetch energy data for.
        filter_col (str): The column name to filter the data by.
        encoding (str, optional): The encoding used for decoding messages. Defaults to "utf-8".

    Returns:
        pl.DataFrame: The energy data for the specified day.

    Examples:
        >>> config = {
        ...     "bootstrap_servers": "localhost:9092",
        ...     "auto_offset_reset": "earliest",
        ...     "enable_auto_commit": True,
        ...     "group_id": "my-group"
        ... }
        >>> day = datetime.date(2022, 1, 1)
        >>> filter_col = "date1"
        >>> fetch_energy_at_day(config, day, filter_col)
        pl.DataFrame(...)
    """
    consumer: KafkaConsumer = create_consumer_and_seek_to_last_committed_message(config)

    day_messages: list[dict] = []
    for message in consumer:
        message_dict: dict = json.loads(message.decode(encoding))
        if message_dict.get(filter_col) == day:
            day_messages.append(message_dict)

    return pl.from_dicts(day_messages)


def filter_data_by_date(
    data: pl.DataFrame, filter_col: str, filter_date: datetime.datetime | datetime.date
) -> pl.DataFrame:
    """Filters the given DataFrame based on a specified date.

    Args:
        data (pl.DataFrame): The DataFrame to be filtered.
        filter_col (str): The column name to filter on.
        filter_date (datetime.datetime | datetime.date): The date or datetime object to filter by.

    Returns:
        pl.DataFrame: The filtered DataFrame.

    Raises:
        TypeError: If the filter_date is not of type datetime.date or datetime.datetime.
    """
    if isinstance(filter_date, datetime.datetime):
        filter_date = filter_date.date()
    if not isinstance(filter_date, datetime.date):
        raise TypeError(f"filter_dates must be either date or datetime, given type is {type(filter_date)}")

    return data.filter(pl.col(filter_col) == filter_date)


def commit_processed_messages_from_topic_from_day(day: datetime.date, filter_col: str, encoding: str = "utf-8") -> None:
    """Commits processed messages from a Kafka topic for a specific day.

    Args:
        day (datetime.date): The specific day to filter the messages.
        filter_col (str): The column name to filter the messages by.
        encoding (str, optional): The encoding used for deserializing messages. Defaults to "utf-8".

    Returns:
        None
    """
    consumer: KafkaConsumer = create_kafka_consumer({})
    consumer = kafka_consumer_seek_to_last_committed_message(consumer)

    for message in consumer:
        if json.loads(message.decode(encoding)).get(filter_col) == day:
            consumer.commit()


def save_to_datalake_partition_by_date(
    data: pl.DataFrame, day: datetime.date, base_path: pathlib.Path, filename: str
) -> bool:
    """Saves data to a datalake partitioned by date.

    Args:
        data (pl.DataFrame): The data to be saved.
        day (datetime.date): The date used for partitioning.
        base_path (pathlib.Path): The base path for the output.
        filename (str): The name of the file.

    Returns:
        bool: True if the data was successfully saved, False otherwise.
    """
    output_path: pathlib.Path = create_base_output_path_paritioned_by_date(
        base_path=base_path, date=day, file_name=filename
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = pl.from_pandas(data)
        data.write_parquet(output_path)
        return True
    except Exception as e:
        print(e)
        return False


def load_config_and_get_date_to_process() -> tuple[dict[str, Any], datetime.date]:
    """Loads the configuration and retrieves the date to process.

    Returns:
        tuple[dict[str, Any], datetime.date]: A tuple containing the loaded configuration as a dictionary and the date
            to process as a `datetime.date` object.
    """
    config: dict[str, Any] = load_config(pathlib.Path(__file__).parent / "config.toml")

    yesterday: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Tallinn")).date() - datetime.timedelta(
        days=1
    )
    return config, yesterday
