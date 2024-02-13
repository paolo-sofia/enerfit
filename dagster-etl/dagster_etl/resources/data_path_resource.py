from dagster import ConfigurableResource


class DataPathResource(ConfigurableResource):
    """A resource for managing data paths.

    This resource represents a path to a data location, which can be either a `pathlib.Path` object or a string.

    Args:
    path: The path to the data location, which can be either a `pathlib.Path` object or a string.
    """

    path: str
