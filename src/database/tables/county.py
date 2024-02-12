from dataclasses import dataclass


@dataclass
class County:
    """Represents a county.

    Args:
        id (int): The ID of the county.
        name (str): The name of the county.
        latitude (str): The latitude of the county.
        longitude (str): The longitude of the county.
    """

    id: int
    name: str
    latitude: str
    longitude: str
