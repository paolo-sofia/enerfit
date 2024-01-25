# Deve essere un cron job che parte a mezzanotte e salva tutti i dati giornalieri sul datalake, quindi farlo con airflow o qualcosa di simile
import datetime
import pathlib
from dataclasses import dataclass
from typing import Any

import polars as pl

from consumers.utils import (
    commit_processed_messages_from_topic_from_day,
    fetch_data_at_day,
    filter_data_by_date,
    load_config_and_get_date_to_process,
    save_to_datalake_partition_by_date,
)
from utils.polars import (
    cast_column_to_16_bits_numeric,
    cast_column_to_32_bits_numeric,
    cast_column_to_date,
)

ENCODING: str = "utf-8"
BASE_PATH: pathlib.Path = pathlib.Path()


@dataclass
class GasColumns:
    """A dataclass representing the column names for the Gas dataset.

    Attributes:
        forecast_date (str): The column name for the forecast date.
        lowest_price_per_mwh (str): The column name for the lowest price per MWh.
        highest_price_per_mwh (str): The column name for the highest price per MWh.
        origin_date (str): The column name for the origin date.
        data_block_id (str): The column name for the data block ID.
    """

    forecast_date: str = "forecast_date"
    lowest_price_per_mwh: str = "lowest_price_per_mwh"
    highest_price_per_mwh: str = "highest_price_per_mwh"
    origin_date: str = "origin_date"
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
    data = cast_column_to_date(data=data, column_names=[GasColumns.forecast_date, GasColumns.origin_date])
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
    gas: pl.DataFrame = fetch_data_at_day(
        config=config,
        day=yesterday,
        filter_col=GasColumns.forecast_date,
    )

    if gas.is_empty():
        print(f"No data from {yesterday}")
        update_processed_gas_table(True, yesterday)
        return

    gas = preprocess_gas_data(gas)
    gas = filter_data_by_date(data=gas, filter_col=GasColumns.forecast_date, filter_date=yesterday)

    if data_saved := save_to_datalake_partition_by_date(data=gas, day=yesterday, base_path=BASE_PATH, filename="gas"):
        commit_processed_messages_from_topic_from_day(day=yesterday, filter_col=GasColumns.forecast_date)

    update_processed_gas_table(data_saved, yesterday)


if __name__ == "__main__":
    main()
