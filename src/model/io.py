import datetime
import pathlib

import joblib
import lightgbm as lgb

from src.config_loader import get_root_path


def load_model(model_path: pathlib.Path) -> lgb.LGBMModel | None:
    """Load a LightGBM model from the given file path.

    Args:
        model_path: The path to the model file.

    Returns:
        lgb.LGBMModel | None: The loaded LightGBM model if successful, None otherwise.
    """
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def save_model(model: lgb.LGBMModel, model_type: str, is_business: bool = True) -> bool:
    """Save the given LightGBM model to a file.

    Args:
        model: The LightGBM model to be saved.
        model_type: The type of the model.
        is_business: Whether the model is for business or not. Defaults to True.

    Returns:
        bool: True if the model is successfully saved, False otherwise.
    """
    model_name: str = f"""{model_type}_{"business" if is_business else "not_business"}"""
    year, month, day = datetime.datetime.now(tz=datetime.UTC).date().strftime("%Y-%m-%d").split("-")
    output_dir: pathlib.Path = (pathlib.Path(get_root_path()) / "data" / "model" / str(year) / str(month)) / str(day)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.booster_.save_model((output_dir / model_name).with_suffix(".joblib"))
        return True
    except Exception:
        return False
