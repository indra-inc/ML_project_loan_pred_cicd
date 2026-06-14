"""
train.py

Responsibility:
- Build Random Forest model
- Build XGBoost model
- Create full sklearn pipeline: preprocessor + model
- Train model pipeline

Note:
- This file does not read config directly.
- This file does not use MLflow directly.
- Params will come from dev_config.yaml through training_pipeline.py.
"""

from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def build_random_forest_model(params: Dict[str, Any]) -> RandomForestClassifier:
    """
    Build Random Forest classifier using parameters from config.

    Args:
        params (dict): Random Forest hyperparameters.

    Returns:
        RandomForestClassifier: Model object.
    """

    try:
        model = RandomForestClassifier(**params)
        return model

    except Exception as e:
        raise RuntimeError(f"Error in build_random_forest_model: {e}")


def build_xgboost_model(params: Dict[str, Any]):
    """
    Build XGBoost classifier using parameters from config.

    Args:
        params (dict): XGBoost hyperparameters.

    Returns:
        XGBClassifier: Model object.
    """

    try:
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is not installed. Install it using: pip install xgboost"
            )

        model = XGBClassifier(**params)
        return model

    except Exception as e:
        raise RuntimeError(f"Error in build_xgboost_model: {e}")


def get_model(model_name: str, params: Dict[str, Any]):
    """
    Return model object based on model name.

    Args:
        model_name (str): Name of the model.
        params (dict): Model hyperparameters.

    Returns:
        sklearn-compatible model object.
    """

    try:
        model_name = model_name.lower()

        if model_name == "random_forest":
            return build_random_forest_model(params=params)

        elif model_name == "xgboost":
            return build_xgboost_model(params=params)

        else:
            raise ValueError(
                f"Unsupported model_name: {model_name}. "
                "Supported models: random_forest, xgboost"
            )

    except Exception as e:
        raise RuntimeError(f"Error in get_model: {e}")


def build_model_pipeline(preprocessor, model) -> Pipeline:
    """
    Build full sklearn pipeline with preprocessing and model.

    Args:
        preprocessor: sklearn ColumnTransformer or preprocessing pipeline.
        model: sklearn-compatible model.

    Returns:
        Pipeline: Full ML pipeline.
    """

    try:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        return pipeline

    except Exception as e:
        raise RuntimeError(f"Error in build_model_pipeline: {e}")


def train_model_pipeline(
    X_train,
    y_train,
    preprocessor,
    model_name: str,
    model_params: Dict[str, Any]
) -> Pipeline:
    """
    Build and train full model pipeline.

    Args:
        X_train: Training features.
        y_train: Training target.
        preprocessor: Preprocessing pipeline.
        model_name (str): random_forest or xgboost.
        model_params (dict): Model hyperparameters.

    Returns:
        Pipeline: Fitted sklearn pipeline.
    """

    try:
        model = get_model(
            model_name=model_name,
            params=model_params
        )

        model_pipeline = build_model_pipeline(
            preprocessor=preprocessor,
            model=model
        )

        model_pipeline.fit(X_train, y_train)

        return model_pipeline

    except Exception as e:
        raise RuntimeError(f"Error in train_model_pipeline: {e}")

