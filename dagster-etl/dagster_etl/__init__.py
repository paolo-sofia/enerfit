import pathlib

from dagster import AssetSelection, Definitions, ScheduleDefinition, define_asset_job
from dagster_polars.io_managers import PolarsParquetIOManager

from .assets import data_loader_assets
from .resources.data_path_resource import DataPathResource
from .resources.model_dataset_config_resource import ModelDatasetConfigResource
from .utils.configs import load_data_preprocessing_config

# set the data directory
BASE_DATA_DIR: pathlib.Path = pathlib.Path("../data")

# get configs
config = load_data_preprocessing_config()
config["paths"] = {k: str(BASE_DATA_DIR / v) for k, v in config["paths"].items()}


# load assets
# all_assets = load_assets_from_modules([loaders])
all_assets = [*data_loader_assets]

# define jobs with schedules
data_loader_job = define_asset_job("loader_job", selection=AssetSelection.groups("loaders"))
data_loader_schedule = ScheduleDefinition(
    job=data_loader_job, cron_schedule="0 0 * * *", execution_timezone="Europe/Rome"
)

# Define Managers
polars_parquet_manager = PolarsParquetIOManager(base_dir="../data")


# Definitions a.k.a project
defs = Definitions(
    assets=all_assets,
    schedules=[data_loader_schedule],
    resources={
        "polars_parquet_io_manager": polars_parquet_manager,
        "data_path_resource": DataPathResource(**config["paths"]),
        "model_data_resource": ModelDatasetConfigResource(**config.get("preprocessing", {})),
    },
)
