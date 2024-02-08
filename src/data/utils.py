import random
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

MAP_MODEL_TYPE: dict[str, int] = {"producer": 0, "consumer": 1}


@dataclass
class FeatureImportanceColumns:
    """A data class that represents the column names for features importance's.

    Attributes:
        feature (str): The column name for the feature.
        importance (str): The column name for the importance.
        importance_perc (str): The column name for the percentage importance.
        importance_perc_cumulative (str): The column name for the cumulative percentage importance.
    """

    feature: str = "feature"
    importance: str = "importance"
    importance_perc: str = "importance_perc"
    importance_perc_cumulative: str = "importance_perc_cumulative"


def set_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.

    Returns:
        None

    Examples:
        >>> set_seed(42)
        >>> # Perform operations that require random number generation
    """
    np.random.default_rng(seed)
    random.seed(seed)
    pl.set_random_seed(seed)


def get_feature_importances_and_print_useless_columns(model: lgb.LGBMModel) -> pd.DataFrame:
    """Get the feature importances of a trained LGBMRegressor model and print the list of useless columns.

    Args:
        model: The trained LGBMRegressor model.

    Returns:
        pd.DataFrame: A DataFrame containing the feature importances, sorted by importance in descending order.

    Examples:
        >>> model = lgb.LGBMRegressor()
        >>> feature_importances = get_feature_importances_and_print_useless_columns(model)
    """
    # create the feature importance dataset and sort it by importance
    feature_importances: pd.DataFrame = (
        pd.DataFrame(
            {
                FeatureImportanceColumns.feature: model.feature_name_,
                FeatureImportanceColumns.importance: model.feature_importances_,
            }
        )
        .sort_values(FeatureImportanceColumns.importance, ascending=False)
        .reset_index(drop=True)
    )

    # compute the importance percentage of each feature
    feature_importances[FeatureImportanceColumns.importance_perc] = (
        feature_importances[FeatureImportanceColumns.importance]
        / feature_importances[FeatureImportanceColumns.importance].sum()
    ) * 100
    # feature_importances = feature_importances.sort_values(
    #     by=FeatureImportanceColumns.importance_perc, ascending=False
    # )

    # compute the cumulative percentage of importance
    feature_importances[FeatureImportanceColumns.importance_perc_cumulative] = feature_importances[
        FeatureImportanceColumns.importance_perc
    ].cumsum()

    # get the feature noise importance and plot all the features that have an importance value lower than the noise
    # feature.
    noise_importance = feature_importances.query("feature == 'noise'").importance.item()

    print(
        feature_importances[
            feature_importances[FeatureImportanceColumns.importance] < noise_importance
        ].feature.tolist()
    )

    return feature_importances
