import datetime
from dataclasses import dataclass


@dataclass
class WeatherForecast:
    """Represents weather forecast data.

    Args:
        latitude (float): The latitude of the location.
        longitude (float): The longitude of the location.
        hours_ahead (int): The number of hours ahead the weather forecast is for.
        temperature (float): The temperature in degrees Celsius.
        dewpoint (float): The dewpoint temperature in degrees Celsius.
        cloudcover_total (float): The total cloud cover in percentage.
        cloudcover_low (float): The low-level cloud cover in percentage.
        cloudcover_mid (float): The mid-level cloud cover in percentage.
        cloudcover_high (float): The high-level cloud cover in percentage.
        u_wind_component_10_metre (float): The u-component of wind at 10 meters above ground level.
        v_wind_component_10_metre (float): The v-component of wind at 10 meters above ground level.
        data_block_id (int): The ID of the data block.
        datetime (datetime.datetime): The date and time of the weather data.
        direct_solar_radiation (float): The direct solar radiation in watts per square meter.
        snowfall (float): The snowfall in millimeters.
        surface_solar_radiation_downwards (float): The surface solar radiation downwards in watts per square meter.
        total_precipitation (float): The total precipitation in millimeters.

    Returns:
        None
    """

    latitude: float
    longitude: float
    hours_ahead: int
    temperature: float
    dewpoint: float
    cloudcover_total: float
    cloudcover_low: float
    cloudcover_mid: float
    cloudcover_high: float
    u_wind_component_10_metre: float
    v_wind_component_10_metre: float
    data_block_id: int
    datetime: datetime.datetime
    direct_solar_radiation: float
    snowfall: float
    surface_solar_radiation_downwards: float
    total_precipitation: float
