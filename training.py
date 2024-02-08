import lightgbm as lgb
import mlflow
import pandas as pd

from src.data.preprocessing import create_dataset
from src.data.utils import get_feature_importances_and_print_useless_columns
from src.model.model_selection import split_train_test

SEED = 666


def train_model(x_train: pd.DataFrame, y_train: pd.Series, eval_set: tuple[pd.DataFrame, pd.Series]) -> lgb.LGBMModel:
    eval_results = {}
    model = lgb.LGBMRegressor(
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=10_000,
        subsample_for_bin=200_000,
        objective="l2",
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=SEED,
        n_jobs=-1,
        importance_type="split",
        linear_tree=True,
        verbosity=0,
        device="cpu",
    )

    model.fit(
        X=x_train,
        y=y_train,
        eval_set=[eval_set],
        eval_metric="mae",
        callbacks=[lgb.log_evaluation(), lgb.record_evaluation(eval_results), lgb.early_stopping(stopping_rounds=100)],
    )
    return model


columns_to_drop_consumer = [
    "client_id",
    "data_block_id",
    # "date",
    "date_client",
    "date_gas",
    "datetime",
    "county",
    "county_name_forecast",
    "latitude_min_forecast",
    "latitude_max_forecast",
    "longitude_min_forecast",
    "longitude_max_forecast",
    "year",
    "month",
]

dataset = create_dataset(model_type="consumer", columns=[], add_noise_column=True)
train_index, test_index = split_train_test(data=dataset)
dataset = dataset.drop(columns=["date"])
x_train, x_test = dataset.loc[train_index], dataset.loc[test_index]
y_train, y_test = x_train.pop("target"), x_test.pop("target")

mlflow.set_experiment("consumer_model")

mlflow.lightgbm.autolog(
    log_input_examples=False,
    log_model_signatures=True,
    log_models=True,
    log_datasets=False,
    disable=False,
    exclusive=False,
    disable_for_unsupported_versions=False,
    silent=False,
    registered_model_name=None,
    extra_tags=None,
)

print("starting run")
with mlflow.start_run() as run:
    consumer_model = train_model(x_train=x_train, y_train=y_train, eval_set=(x_test, y_test))

feature_importances_consumer: pd.DataFrame = get_feature_importances_and_print_useless_columns(consumer_model)
