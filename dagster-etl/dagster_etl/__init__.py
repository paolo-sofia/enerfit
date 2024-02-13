import pathlib

from dagster import AssetSelection, Definitions, ScheduleDefinition, define_asset_job, load_assets_from_modules
from dagster_polars.io_managers import PolarsParquetIOManager

from .assets import loaders
from .resources.data_path_resource import DataPathResource
from .utils.configs import load_data_preprocessing_config

# set the data directory
BASE_DATA_DIR: pathlib.Path = pathlib.Path("../data")

# get config
config = load_data_preprocessing_config()

all_assets = load_assets_from_modules([loaders])

clients_job = define_asset_job("clients_job", selection=AssetSelection.all())
clients_schedule = ScheduleDefinition(job=clients_job, cron_schedule="0 0 * * *")

polars_parquet_manager = PolarsParquetIOManager(base_dir="../data")

defs = Definitions(
    assets=all_assets,
    schedules=[clients_schedule],
    resources={
        "polars_parquet_io_manager": polars_parquet_manager,
        "data_path_resource": DataPathResource(path=str(BASE_DATA_DIR / config.get("paths", {}).get("clients", ""))),
    },
)
