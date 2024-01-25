# Deve essere un cron job che parte a mezzanotte e salva tutti i dati giornalieri sul datalake, quindi farlo con airflow o qualcosa di simile
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from consumers.utils import (
    commit_processed_messages_from_topic_from_day,
    fetch_data_at_day,
    load_config_and_get_date_to_process,
    save_to_datalake_partition_by_date,
)
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
    data = cast_column_to_32_bits_numeric(data)
    data = preprocess_product_type(data)
    data = preprocess_county(data)
    return preprocess_is_business(data)


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
    config: dict[str, Any]
    yesterday: datetime.date
    config, yesterday = load_config_and_get_date_to_process()

    # Prenditi tutti i dati di ieri da kafka
    clients: pl.DataFrame = fetch_data_at_day(config=config, day=yesterday, filter_col=ClientsColumns.date)
    clients = cast_column_to_date(clients, ClientsColumns.date).filter(pl.col(ClientsColumns.date) == yesterday)

    if clients.is_empty():
        print(f"No data from {yesterday}")
        update_processed_clients_table(True, yesterday)

    clients = preprocess_clients_columns(clients)

    if data_saved := save_to_datalake_partition_by_date(
        data=clients, day=yesterday, base_path=BASE_PATH, filename="clients"
    ):
        commit_processed_messages_from_topic_from_day(day=yesterday, filter_col=ClientsColumns.date)

    update_processed_clients_table(data_saved, yesterday)


if __name__ == "__main__":
    main()
