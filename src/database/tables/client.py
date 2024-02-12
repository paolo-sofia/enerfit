import datetime
from dataclasses import dataclass


@dataclass
class Client:
    """Represents a client.

    Args:
        id (int): The ID of the client.
        product_type (str): The type of product the client has.
        county (int): The county of the client.
        eic_count (float): The EIC count of the client.
        installed_capacity (float): The installed capacity of the client.
        is_business (bool): Indicates if the client is a business.
        date (datetime.date): The date associated with the client.
        data_block_id (int): The ID of the data block associated with the client.

    Attributes:
        id (int): The ID of the client.
        product_type (str): The type of product the client has.
        county (int): The county of the client.
        eic_count (float): The EIC count of the client.
        installed_capacity (float): The installed capacity of the client.
        is_business (bool): Indicates if the client is a business.
        date (datetime.date): The date associated with the client.
        data_block_id (int): The ID of the data block associated with the client.
    """

    id: int
    product_type: str
    county: int
    eic_count: float
    installed_capacity: float
    is_business: bool
    date: datetime.date
    data_block_id: int
