import pathlib

import joblib
import lightgbm as lgb
import pandas as pd


def load_model(model_path: pathlib.Path) -> lgb.LGBMModel | None:
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict(model: lgb.LGBMModel, data: pd.DataFrame) -> float:
    return model.predict(data)
