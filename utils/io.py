import datetime
import pathlib


def create_base_output_path_paritioned_by_date(
    base_path: pathlib.Path, date: datetime.date, file_name: str
) -> pathlib.Path:
    """Creates a base output path partitioned by date.

    Args:
        base_path (pathlib.Path): The base path for the output.
        date (datetime.date): The date used for partitioning.
        file_name (str): The name of the file.

    Returns:
        pathlib.Path: The created output path.

    Examples:
        >>> base_path = pathlib.Path("/data")
        >>> date = datetime.date(2022, 1, 1)
        >>> file_name = "output"
        >>> create_base_output_path_paritioned_by_date(base_path, date, file_name)
        PosixPath('/data/2022/1/1/output.parquet')
    """
    year, month, day = date.year, date.month, date.day
    return base_path / str(year) / str(month) / str(day) / f"{file_name}.parquet"
