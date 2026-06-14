"""
services.py

Responsibility:
- Build MLflow model URI
- Load registered MLflow model
- Prepare input data for prediction
- Return business prediction label
"""

from typing import Optional

import pandas as pd
import mlflow.sklearn

from app.config import app_config
from loan_pred.processing.preprocessing_pipeline import prepare_prediction_data


EXPECTED_PREDICTION_COLUMNS = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

TARGET_COLUMN = "loan_status"
COLUMNS_TO_DROP = ["loan_id"]


# Based on your training output:
# LabelEncoder classes: ['Approved', 'Rejected']
# Therefore:
# 0 -> Approved
# 1 -> Rejected
TARGET_MAPPING = {
    0: app_config.target_negative_label,  # Approved
    1: app_config.target_positive_label,  # Rejected
}


def build_model_uri(
    registered_model_name: str,
    model_version: Optional[str] = None,
    model_alias: Optional[str] = None
) -> str:
    """
    Build MLflow model URI.

    Priority:
    - If model_alias is provided, load using alias.
    - Otherwise load using model_version.
    """

    try:
        if model_alias:
            return f"models:/{registered_model_name}@{model_alias}"

        if model_version:
            return f"models:/{registered_model_name}/{model_version}"

        raise ValueError("Either model_version or model_alias must be provided.")

    except Exception as e:
        raise RuntimeError(f"Error in build_model_uri: {e}")


def load_registered_model():
    """
    Load registered model from MLflow Model Registry.
    """

    try:
        mlflow.set_tracking_uri(app_config.mlflow_tracking_uri)

        model_uri = build_model_uri(
            registered_model_name=app_config.registered_model_name,
            model_version=app_config.model_version,
            model_alias=app_config.model_alias
        )

        model = mlflow.sklearn.load_model(model_uri) # type: ignore

        print(f"🤖 Model loaded successfully from URI: {model_uri}")

        return model

    except Exception as e:
        raise RuntimeError(f"Error in load_registered_model: {e}")


def predict_single_record(
    model,
    input_data: dict
) -> dict:
    """
    Predict loan approval for a single input record.
    """

    try:
        input_df = pd.DataFrame([input_data])

        prediction_df = prepare_prediction_data(
            df=input_df,
            columns_to_drop=COLUMNS_TO_DROP,
            target_column=TARGET_COLUMN,
            expected_columns=EXPECTED_PREDICTION_COLUMNS
        )

        encoded_prediction = model.predict(prediction_df)[0]
        encoded_prediction = int(encoded_prediction)

        prediction_label = TARGET_MAPPING.get(
            encoded_prediction,
            str(encoded_prediction)
        )

        return {
            "encoded_prediction": encoded_prediction,
            "prediction": prediction_label
        }

    except Exception as e:
        raise RuntimeError(f"Error in predict_single_record: {e}")


def predict_batch_records(
    model,
    input_records: list[dict]
) -> list[dict]:
    """
    Predict loan approval for multiple records.
    """

    try:
        input_df = pd.DataFrame(input_records)

        prediction_df = prepare_prediction_data(
            df=input_df,
            columns_to_drop=COLUMNS_TO_DROP,
            target_column=TARGET_COLUMN,
            expected_columns=EXPECTED_PREDICTION_COLUMNS
        )

        encoded_predictions = model.predict(prediction_df)

        results = []

        for pred in encoded_predictions:
            pred = int(pred)
            results.append(
                {
                    "encoded_prediction": pred,
                    "prediction": TARGET_MAPPING.get(pred, str(pred))
                }
            )

        return results

    except Exception as e:
        raise RuntimeError(f"Error in predict_batch_records: {e}")