"""
predict.py

Responsibility:
- Load registered MLflow model
- Prepare raw prediction input
- Run prediction
- Convert encoded prediction into business label:
    0/1 -> Approved/Rejected

Run examples:

Using model version:
    python -m loan_pred.inference.predict \
        --model_name Prediction_Model_RF \
        --model_version 1

Using model alias:
    python -m loan_pred.inference.predict \
        --model_name Prediction_Model_RF \
        --model_alias champion
"""

import argparse
from typing import Optional

import pandas as pd
import mlflow.sklearn

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


# Important:
# LabelEncoder sorted classes alphabetically during training.
# For your training data:
# label_encoder.classes_ = ['Approved', 'Rejected']
# So:
# 0 -> Approved
# 1 -> Rejected
TARGET_MAPPING = {
    0: "Approved",
    1: "Rejected"
}


def build_model_uri(
    model_name: str,
    model_version: Optional[str] = None,
    model_alias: Optional[str] = None
) -> str:
    """
    Build MLflow model URI from registered model name.

    Args:
        model_name (str): Registered model name.
        model_version (str): Model version, e.g. "1".
        model_alias (str): Model alias, e.g. "champion".

    Returns:
        str: MLflow model URI.
    """

    try:
        if model_alias:
            return f"models:/{model_name}@{model_alias}"

        if model_version:
            return f"models:/{model_name}/{model_version}"

        raise ValueError(
            "Either model_version or model_alias must be provided."
        )

    except Exception as e:
        raise RuntimeError(f"Error in build_model_uri: {e}")


def load_registered_model(
    tracking_uri: str,
    model_name: str,
    model_version: Optional[str] = None,
    model_alias: Optional[str] = None
):
    """
    Load registered model from MLflow Model Registry.

    Args:
        tracking_uri (str): MLflow tracking URI.
        model_name (str): Registered model name.
        model_version (str): Model version.
        model_alias (str): Model alias.

    Returns:
        Loaded MLflow sklearn model.
    """

    try:
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = build_model_uri(
            model_name=model_name,
            model_version=model_version,
            model_alias=model_alias
        )

        print(f"Loading model from URI: {model_uri}")

        model = mlflow.sklearn.load_model(model_uri) # type: ignore

        return model

    except Exception as e:
        raise RuntimeError(f"Error in load_registered_model: {e}")


def predict_single_record(
    model,
    input_data: dict
) -> dict:
    """
    Predict for a single input record.

    Args:
        model: Loaded MLflow model pipeline.
        input_data (dict): Raw input dictionary.

    Returns:
        dict: Prediction output.
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
    input_data: list[dict]
) -> list[dict]:
    """
    Predict for multiple input records.

    Args:
        model: Loaded MLflow model pipeline.
        input_data (list): List of raw input dictionaries.

    Returns:
        list: Prediction outputs.
    """

    try:
        input_df = pd.DataFrame(input_data)

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


def main():
    """
    Local smoke test for registered model prediction.
    """

    try:
        args = parse_args()

        model = load_registered_model(
            tracking_uri=args.tracking_uri,
            model_name=args.model_name,
            model_version=args.model_version,
            model_alias=args.model_alias
        )

        sample_input = {
            "loan_id": 1,
            "no_of_dependents": 2,
            "education": "Graduate",
            "self_employed": "No",
            "income_annum": 9600000,
            "loan_amount": 29900000,
            "loan_term": 12,
            "cibil_score": 778,
            "residential_assets_value": 2400000,
            "commercial_assets_value": 17600000,
            "luxury_assets_value": 22700000,
            "bank_asset_value": 8000000
        }

        result = predict_single_record(
            model=model,
            input_data=sample_input
        )

        print("Prediction result:")
        print(result)

    except Exception as e:
        print(f"Prediction failed: {e}")


def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Load registered MLflow model and run prediction"
    )

    parser.add_argument(
        "--tracking_uri",
        type=str,
        default="http://127.0.0.1:5000",
        help="MLflow tracking URI"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Registered model name"
    )

    parser.add_argument(
        "--model_version",
        type=str,
        default=None,
        help="Registered model version, e.g. 1"
    )

    parser.add_argument(
        "--model_alias",
        type=str,
        default=None,
        help="Registered model alias, e.g. champion"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()


# CLI: python pipelines/predict.py   --model_name Prediction_Model_RF   --
# model_version 1