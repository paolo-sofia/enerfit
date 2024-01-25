from typing import Type

import polars as pl


def cast_column_to_32_bits_numeric(data: pl.DataFrame) -> pl.DataFrame:
    """Casts the columns of a DataFrame to 32-bit numeric types.

    Args:
        data: The DataFrame to be cast.

    Returns:
        The DataFrame with columns cast to 32-bit numeric types.
    """
    column_mapping: dict[Type[pl.Int64 | pl.Float64], Type[pl.Int32 | pl.Float32]] = {
        pl.Int64: pl.Int32,
        pl.Float64: pl.Float32,
    }
    return data.with_columns(
        [
            pl.col(column).cast(column_mapping.get(col_dtype, col_dtype))
            for column, col_dtype in zip(data.columns, data.dtypes)
        ]
    )


def cast_column_to_date(data: pl.DataFrame, column_names: str | list[str]) -> pl.DataFrame:
    """Casts specified columns of a DataFrame to date type.

    Args:
        data: The DataFrame to be modified.
        column_names: A string or a list of strings representing the column names to be cast.

    Returns:
        The DataFrame with specified columns cast to date type.

    Raises:
        TypeError: If column_names is not a string or a list of strings.

    Examples:
        >>> data = pl.DataFrame({
        ...     'date1': ['2022-01-01', '2022-01-02'],
        ...     'date2': ['2022-01-03', '2022-01-04']
        ... })
        >>> cast_column_to_date(data, 'date1')
        pl.DataFrame({
            'date1': [datetime.date(2022, 1, 1), datetime.date(2022, 1, 2)],
            'date2': ['2022-01-03', '2022-01-04']
        })
    """
    if isinstance(column_names, str):
        column_names = [column_names]

    if not isinstance(column_names, list):
        raise TypeError(f"column_names must be either a str or a list of string, given type is {type(column_names)}")

    return data.with_columns([pl.col(column).str.to_date("%Y-%m-%d") for column in column_names])