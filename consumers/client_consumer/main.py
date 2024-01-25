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
from utils.kafka import create_kafka_consumer, kafka_consumer_seek_to_last_committed_message
from utils.polars import cast_column_to_32_bits_numeric, cast_column_to_date

ENCODING: str = "utf-8"
COUNTY_MAP_PATH: pathlib.Path = pathlib.Path() / "data" / "county_id_to_name_map.json"
BASE_PATH: pathlib.Path = pathlib.Path()

product_type_map: dict[int, str] = {0: "Combined", 1: "Fixed", 2: "General service", 3: "Spot"}


@dataclass
class ClientsColumns:
    """A dataclass representing the column names for the Clients dataset.

    Attributes:
        product_type (str): The column name for the product type.
        county (str): The column name for the county.
        eic_count (str): The column name for the EIC count.
        installed_capacity (str): The column name for the installed capacity.
        is_business (str): The column name for the business indicator.
        date (str): The column name for the date.
        data_block_id (str): The column name for the data block ID.
    """

    product_type: str = "product_type"
    county: str = "county"
    eic_count: str = "eic_count"
    installed_capacity: str = "installed_capacity"
    is_business: str = "is_business"
    date: str = "date"
    data_block_id: str = "data_block_id"


def preprocess_product_type(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses the product type column in the given DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the product type column.

    Returns:
        pd.DataFrame: The DataFrame with the product type column preprocessed.

    Examples:
        >>> data = pd.DataFrame({"product_type": ["A", "B", "C"]})
        >>> preprocess_product_type(data)
           product_type
        0             A
        1             B
        2             C
    """
    # data["product_type"] = data["product_type"].apply(lambda x: product_type_map.get(x, x))
    return data.with_columns([pl.col(ClientsColumns.product_type).apply(lambda x: product_type_map.get(x, x))])


def preprocess_county(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses the county column in the given DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the county column.

    Returns:
        pd.DataFrame: The DataFrame with the county column preprocessed.

    Examples:
        >>> data = pd.DataFrame({"county": [1, 2, 3]})
        >>> preprocess_county(data)
           county
        0  County1
        1  County2
        2  County3
    """
    with COUNTY_MAP_PATH.open("r") as f:
        county_id_to_name_map: dict[int, str] = json.load(f)

    return data.with_columns([pl.col(ClientsColumns.county).apply(lambda x: county_id_to_name_map.get(x, "Unknown"))])


def preprocess_is_business(data: pl.DataFrame) -> pl.DataFrame:
    """Preprocesses the is_business column in the given DataFrame by replacing 0 with False and 1 with True.

    Args:
        data (pd.DataFrame): The DataFrame containing the is_business column.

    Returns:
        pd.DataFrame: The DataFrame with the is_business column preprocessed.

    Examples:
        >>> data = pd.DataFrame({"is_business": [0, 1, 0]})
        >>> preprocess_is_business(data)
           is_business
        0        False
        1         True
        2        False
    """
    return data.with_columns([pl.col(ClientsColumns.is_business).cast(pl.Boolean)])


def preprocess_clients_columns(data: pl.DataFrame) -> pd.DataFrame:
    """Preprocesses multiple columns in the given DataFrame.

    This function applies a series of preprocessing steps to the product_type, county, date, and is_business columns in
        the DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.

    Examples:
        >>> data = pd.DataFrame({"product_type": ["A", "B", "C"], "county": [1, 2, 3],
            "date": ["2022-01-01", "2022-01-02", "2022-01-03"], "is_business": [0, 1, 0]})
        >>> preprocess_clients_columns(data)
           product_type   county       date  is_business
        0             A  County1 2022-01-01        False
        1             B  County2 2022-01-02         True
        2             C  County3 2022-01-03        False
    """
    data = preprocess_product_type(data)
    data = preprocess_county(data)
    return preprocess_is_business(data)


def fetch_clients_at_day(config: dict[str, Any], day: datetime.date) -> pl.DataFrame:
    """Fetches client data for a specific day from a Kafka topic.

    Args:
        config (dict): A dictionary containing the Kafka consumer configuration.
        day (datetime.date): The specific day to fetch client data for.

    Returns:
        pl.DataFrame: The client data for the specified day.

    Examples:
        >>> config = {
        ...     "bootstrap_servers": "localhost:9092",
        ...     "auto_offset_reset": "earliest",
        ...     "enable_auto_commit": True,
        ...     "group_id": "my-group"
        ... }
        >>> day = datetime.date(2022, 1, 1)
        >>> fetch_clients_at_day(config, day)
        pl.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.5, 5.6, 6.7]
        })
    """
    consumer: KafkaConsumer = create_kafka_consumer(config)
    consumer = kafka_consumer_seek_to_last_committed_message(consumer)

    day_messages: list[dict] = []
    for message in consumer:
        message_dict: dict = json.loads(message.decode(ENCODING))
        if message_dict.get(ClientsColumns.date) == day:
            day_messages.append(message_dict)

    data = pl.from_dicts(day_messages)
    return cast_column_to_date(data, ClientsColumns.date).filter(pl.col(ClientsColumns.date) == day)


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


def save_clients_to_datalake(data: pd.DataFrame, day: datetime.date) -> bool:
    """Saves client data to the datalake.

    This function takes a DataFrame of client data and a specific day, and saves the data to the datalake. The data is
        saved as a parquet file at the corresponding output path based on the given day.

    Args:
        data (pd.DataFrame): The DataFrame of client data to be saved.
        day (datetime.date): The specific day associated with the client data.

    Returns:
        bool: True if the data is successfully saved, False otherwise.
    """
    output_path: pathlib.Path = create_base_output_path_paritioned_by_date(
        base_path=BASE_PATH, date=day, file_name="clients"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = pl.from_pandas(data)
        data.write_parquet(output_path)
        return True
    except Exception as e:
        print(e)
        return False


def update_processed_clients_table(is_processed: bool, day: datetime.date) -> bool:
    """Updates the processed clients table for a specific day.

    Args:
        is_processed (bool): The flag indicating if the clients are processed or not.
        day (datetime.date): The specific day to update the processed clients table for.

    Returns:
        bool: True if the processed clients table was successfully updated, False otherwise.
    """
    pass


def main() -> None:
    """Runs the main function to process client data.

    Returns:
        None
    """
    config: dict[str, Any] = load_config(pathlib.Path(__file__).parent / "config.toml")

    yesterday: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Tallin")).date() - datetime.timedelta(
        days=1
    )

    # Prenditi tutti i dati di ieri da kafka
    clients: pl.DataFrame = fetch_clients_at_day(config=config, day=yesterday)
    if clients.is_empty():
        print(f"No data from {yesterday}")
        update_processed_clients_table(True, yesterday)

    clients = cast_column_to_32_bits_numeric(clients)
    clients = preprocess_clients_columns(clients)

    if data_saved := save_clients_to_datalake(clients, yesterday):
        commit_processed_messages_from_topic(yesterday)

    update_processed_clients_table(data_saved, yesterday)


if __name__ == "__main__":
    main()
