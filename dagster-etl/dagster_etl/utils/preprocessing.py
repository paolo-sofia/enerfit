import polars as pl


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
