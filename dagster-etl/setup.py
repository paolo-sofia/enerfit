from setuptools import find_packages, setup

setup(
    name="dagster_etl",
    packages=find_packages(exclude=["dagster_etl_tests"]),
    install_requires=["dagster", "dagster-cloud", "pandas", "polars", "dagster-polars", "lightgbm"],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
