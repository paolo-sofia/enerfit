import datetime

import polars as pl
import pytz
from dateutil.relativedelta import relativedelta


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


def get_start_and_end_date_from_config(
    train_months: int = 6, test_months: int = 3
) -> tuple[datetime.date, datetime.date]:
    """Get the start and end date based on the specified number of training and testing months.

    Args:
        train_months (int): The number of training months. Default is 6.
        test_months (int): The number of testing months. Default is 3.

    Returns:
        tuple[datetime.date, datetime.date]: A tuple containing the start date and today's date.
    """
    today: datetime.date = datetime.datetime.now(tz=pytz.timezone("Europe/Rome")).today()
    start_date = today - relativedelta(months=train_months + test_months)
    return start_date, today
