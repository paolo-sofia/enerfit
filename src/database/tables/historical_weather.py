import datetime
from dataclasses import dataclass


@dataclass
class HistoricalWeather:
    """Represents historical weather data.

    Args:
        datetime (datetime.datetime): The datetime of the weather data.
        temperature (float): The temperature in degrees Celsius.
        dewpoint (float): The dewpoint temperature in degrees Celsius.
        rain (float): The amount of rain in millimeters.
        snowfall (float): The amount of snowfall in millimeters.
        surface_pressure (float): The surface pressure in hPa.
        cloudcover_total (float): The total cloud cover in percentage.
        cloudcover_low (float): The low-level cloud cover in percentage.
        cloudcover_mid (float): The mid-level cloud cover in percentage.
        cloudcover_high (float): The high-level cloud cover in percentage.
        windspeed_10m (float): The wind speed at 10 meters above ground level in meters per second.
        winddirection_10m (float): The wind direction at 10 meters above ground level in degrees.
        shortwave_radiation (float): The shortwave radiation in watts per square meter.
        direct_solar_radiation (float): The direct solar radiation in watts per square meter.
        diffuse_radiation (float): The diffuse solar radiation in watts per square meter.
        latitude (float): The latitude of the weather location.
        longitude (float): The longitude of the weather location.
        data_block_id (int): The ID of the data block associated with the weather data.
    """

    datetime: datetime.datetime
    temperature: float
    dewpoint: float
    rain: float
    snowfall: float
    surface_pressure: float
    cloudcover_total: float
    cloudcover_low: float
    cloudcover_mid: float
    cloudcover_high: float
    windspeed_10m: float
    winddirection_10m: float
    shortwave_radiation: float
    direct_solar_radiation: float
    diffuse_radiation: float
    latitude: float
    longitude: float
    data_block_id: int
