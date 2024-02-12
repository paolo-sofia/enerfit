import datetime
from dataclasses import dataclass


@dataclass
class Electricity:
    """Represents electricity data.

    Args:
        datetime (datetime.datetime): The datetime of the electricity data.
        euros_per_mwh (float): The price of electricity per MWh.
        data_block_id (int): The ID of the data block associated with the electricity data.
    """

    datetime: datetime.datetime
    euros_per_mwh: float
    data_block_id: int
