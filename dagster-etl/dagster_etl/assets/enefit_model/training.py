import lightgbm as lgb
import pandas as pd

#
# from src.config_loader import load_model_config
# from src.data.preprocessing import (
#     create_dataset,
# )
# from src.data.utils import set_seed
# from src.model.model_selection import cross_validation_month_and_year, split_train_test
# from src.utils import check_model_type
#
# config = load_model_config()
#
# SEED: int = config.get("general", {}).get("seed", 42)
# set_seed(seed=SEED)
#
#
# def train_model(
#     dataframe: pd.DataFrame, train_indexes: list[int], test_indexes: list[int], parameters: dict[str, Any]
# ) -> lgb.LGBMModel:
#     """Train a LightGBM model using the specified data and parameters.
#
#     Args:
#         dataframe (pd.DataFrame): The input DataFrame containing the training data.
#         train_indexes (list[int]): The indices of the training data.
#         test_indexes (list[int]): The indices of the test data.
#         parameters (dict[str, Any]): The parameters for the LightGBM model.
#
#     Returns:
#         lgb.LGBMModel: The trained LightGBM model.
#
#     Examples:
#         >>> dataframe = pd.DataFrame(...)
#         >>> train_indexes = [0, 1, 2, 3, 4]
#         >>> test_indexes = [5, 6, 7]
#         >>> parameters = {"objective": "l2", "num_leaves": 10, "learning_rate": 0.1}
#         >>> model = train_model(dataframe, train_indexes, test_indexes, parameters)
#         >>> # Use the trained model for prediction or evaluation
#     """
#     x_train, x_test = dataframe.loc[train_indexes], dataframe.loc[test_indexes]
#     y_train, y_test = x_train.pop("target"), x_test.pop("target")
#     # eval_results = {}
#     model = lgb.LGBMRegressor(random_state=SEED, **parameters)
#
#     model.fit(
#         X=x_train,
#         y=y_train,
#         eval_set=[(x_test, y_test)],
#         eval_metric="mae",
#         callbacks=[
#             lgb.log_evaluation(),
#             # lgb.record_evaluation(eval_results),
#             lgb.early_stopping(stopping_rounds=100),
#         ],
#     )
#     return model
#
#
# def train_model_cross_validation(
#     dataframe: pd.DataFrame, params: dict[str, str | int | float | bool]
# ) -> list[lgb.LGBMModel]:
#     """Train multiple models using cross-validation based on month and year.
#
#     Args:
#         dataframe (pd.DataFrame): The input DataFrame containing the data.
#         params (dict[str, str | int | float]): the model's parameters.
#
#     Returns:
#         list[lgb.LGBMModel]: A list of trained LGBMModel objects.
#
#     Examples:
#         >>> dataframe = pd.DataFrame(...)
#         >>> models = train_model_cross_validation(dataframe)
#         >>> for model in models:
#         ...     # Use the trained model for prediction or evaluation
#     """
#     models = []
#     for i, (train_index, test_index, start_train_date, end_train_date, start_test_date, end_test_date) in enumerate(
#         cross_validation_month_and_year(dataframe, train_months=6, test_months=3, debug=False)
#     ):
#         print(
#             f"Split {i + 1} - train from {start_train_date} to {end_train_date} --- "
#             f"test from {start_test_date} to {end_test_date}"
#         )
#         models.append(train_model(dataframe, train_index, test_index, parameters=params))
#         break
#     return models
#
#
# def load_data_and_train_model(model_type: str, is_business: bool = True) -> lgb.LGBMModel:
#     """Load the data and train a LightGBM model based on the given model type.
#
#     Args:
#         model_type: The type of the model. Must be either 'producer' or 'consumer'.
#         is_business: Whether the model is for business or not. Defaults to True.
#
#     Returns:
#         lgb.LGBMModel: The trained LightGBM model.
#
#     Raises:
#         ValueError: If the model type is not 'producer' or 'consumer'.
#     """
#     model_type: str = check_model_type(model_type)
#
#     parameters: dict[str, str | float | int | bool] = config.get(model_type, {}).get(
#         "business" if is_business else "not_business", {}
#     )
#     columns: list[str] = parameters.pop("columns", [])
#
#     data = create_dataset(model_type=model_type, is_business=is_business, columns=columns)
#
#     train_index, test_index = split_train_test(data=data)
#
#     data = data.drop(columns=["date"])
#
#     return train_model(dataframe=data, train_indexes=train_index, test_indexes=test_index, parameters=parameters)


def train_producer_model(dataset: pd.DataFrame):
    pass


def train_consumer_model(dataset: pd.DataFrame) -> lgb.LightGBModel:
    pass
