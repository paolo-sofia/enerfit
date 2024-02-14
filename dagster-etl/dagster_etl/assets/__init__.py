from dagster import load_assets_from_package_module

from . import enefit_model

model_assets = load_assets_from_package_module(
    package_module=enefit_model,
    # group_name="enefit_model",
)
