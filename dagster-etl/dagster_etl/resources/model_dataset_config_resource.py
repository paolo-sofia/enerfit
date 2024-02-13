from dagster import ConfigurableResource


class ModelDatasetConfigResource(ConfigurableResource):
    """A resource for managing data paths.

    This resource represents a path to a data location, which can be either a `pathlib.Path` object or a string.

    Args:
    path: The path to the data location, which can be either a `pathlib.Path` object or a string.
    """

    train_months: int
    test_months: int
    offset_hours: int = 11
    lag_days: int = 7
    add_noise_column: bool = True
