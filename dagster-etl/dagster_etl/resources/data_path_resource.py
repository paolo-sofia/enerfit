from dagster import ConfigurableResource


class DataPathResource(ConfigurableResource):
    """A resource for managing data paths.

    This resource represents a path to a data location, which can be either a `pathlib.Path` object or a string.

    Args:
    path: The path to the data location, which can be either a `pathlib.Path` object or a string.
    """

    train: str
    clients: str
    gas: str
    electricity: str
    weather_station_map: str
    weather_forecast: str
    historical_weather: str
