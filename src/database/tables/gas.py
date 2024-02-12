import datetime
from dataclasses import dataclass


@dataclass
class Gas:
    """Represents gas data.

    Args:
        datetime (datetime.datetime): The datetime of the gas data.
        lowest_price_per_mwh (float): The lowest price of gas per MWh.
        highest_price_per_mwh (float): The highest price of gas per MWh.
        data_block_id (int): The ID of the data block associated with the gas data.
    """

    datetime: datetime.datetime
    lowest_price_per_mwh: float
    highest_price_per_mwh: float
    data_block_id: int
