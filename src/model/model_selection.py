import datetime
from typing import Generator

import pandas as pd
from dateutil.relativedelta import relativedelta


def cross_validation_month_and_year(
    dataframe: pd.DataFrame, train_months: int = 3, test_months: int = 1, debug: bool = False
) -> Generator:
    """Generate train and test indices for cross-validation based on month and year.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing a 'date', 'year', and 'month' column.
        train_months (int, optional): The number of months to include in the training set. Defaults to 3.
        test_months (int, optional): The number of months to include in the test set. Defaults to 1.
        debug (bool, optional): Whether to print debug information. Defaults to False.

    Yields:
        Generator: A generator that yields tuples of train and test indices, start and end dates for training and testing.

    Examples:
        >>> dataframe = pd.DataFrame(...)
        >>> for train_index, test_index, start_train_date, end_train_date, start_test_date, end_test_date in cross_validation_month_and_year(dataframe):
        ...     # Perform cross-validation
    """
    for _, row in dataframe[["year", "month"]].drop_duplicates().sort_values(["year", "month"]).iterrows():
        current_date = datetime.date(year=row.year, month=row.month, day=1) + relativedelta(months=1)
        start_train_date = current_date - relativedelta(months=train_months)
        end_train_date = current_date
        start_test_date = current_date
        end_test_date = current_date + relativedelta(months=test_months)

        try:
            train_index = dataframe.query(
                "(date >= @start_train_date & date < @end_train_date) | (year < @row.year & month == @row.month)"
            ).index
            test_index = dataframe.query("(date >= @start_test_date & date < @end_test_date)").index
        except KeyError:
            continue

        if debug:
            print(
                f"train date: {start_train_date} - {end_train_date - relativedelta(months=1)}"
                f"test date: {start_test_date} - {end_test_date - relativedelta(months=1)}"
            )

        if len(train_index) == 0 or len(test_index) == 0:
            continue

        yield train_index, test_index, start_train_date, end_train_date, start_test_date, end_test_date


def split_train_test(data: pd.DataFrame, test_months: int = 6) -> tuple[list[int], ...]:
    """Split the data into train and test sets based on a specified number of test months.

    Args:
        data (pd.DataFrame): The input DataFrame containing a 'date' column.
        test_months (int, optional): The number of months to include in the test set. Defaults to 6.

    Returns:
        tuple[list[int], ...]: A tuple containing the train and test indices.

    Examples:
        >>> data = pd.DataFrame(...)
        >>> train_index, test_index = split_train_test(data)
        >>> train_data = data.loc[train_index]
        >>> test_data = data.loc[test_index]
    """
    max_dataset_date: datetime.date = data["date"].max()
    start_test_date: datetime.date = max_dataset_date - relativedelta(months=test_months)
    train_index = data[data["date"] < start_test_date].index
    test_index = data[data["date"] >= start_test_date].index
    return train_index, test_index
