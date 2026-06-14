"""
preprocessing_pipeline.py

Responsibility:
- Clean string values
- Drop unwanted columns
- Separate features and target
- Encode target column
- Build sklearn preprocessing pipeline

Note:
- This file does not train the model.
- This file does not read config directly.
- Config values will come through training_pipeline.py.
"""

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def clean_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean string values in object columns by removing extra spaces.
    """

    try:
        df = df.copy()

        object_columns = df.select_dtypes(include=["object"]).columns

        for col in object_columns:
            df[col] = df[col].astype(str).str.strip()

        return df

    except Exception as e:
        raise RuntimeError(f"Error in clean_string_values: {e}")


def drop_unwanted_columns(
    df: pd.DataFrame,
    columns_to_drop: list | None = None
) -> pd.DataFrame:
    """
    Drop unwanted columns if they exist.
    Example: loan_id
    """

    try:
        df = df.copy()

        if columns_to_drop is None:
            columns_to_drop = []

        existing_columns = [
            col for col in columns_to_drop
            if col in df.columns
        ]

        if existing_columns:
            df = df.drop(columns=existing_columns)

        return df

    except Exception as e:
        raise RuntimeError(f"Error in drop_unwanted_columns: {e}")


def separate_features_target(
    df: pd.DataFrame,
    target_column: str
):
    """
    Separate dataframe into X and y.
    """

    try:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        return X, y

    except Exception as e:
        raise RuntimeError(f"Error in separate_features_target: {e}")


def encode_target(y: pd.Series):
    """
    Encode target labels into numeric values.
    Example:
        Approved / Rejected -> 0 / 1
    """

    try:
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        return y_encoded, label_encoder

    except Exception as e:
        raise RuntimeError(f"Error in encode_target: {e}")


def get_feature_types(X: pd.DataFrame):
    """
    Identify numerical and categorical columns.
    """

    try:
        numerical_columns = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        return numerical_columns, categorical_columns

    except Exception as e:
        raise RuntimeError(f"Error in get_feature_types: {e}")


def build_preprocessing_pipeline(
    numerical_columns: list,
    categorical_columns: list
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline.

    Numerical pipeline:
    - Median imputation
    - Standard scaling

    Categorical pipeline:
    - Most frequent imputation
    - One-hot encoding
    """

    try:
        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_columns),
                ("cat", categorical_pipeline, categorical_columns)
            ]
        )

        return preprocessor

    except Exception as e:
        raise RuntimeError(f"Error in build_preprocessing_pipeline: {e}")


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    columns_to_drop: list | None = None
):
    """
    Complete preprocessing preparation function.

    Returns:
        X
        y_encoded
        preprocessor
        label_encoder
        numerical_columns
        categorical_columns
    """

    try:
        df = clean_string_values(df=df)

        df = drop_unwanted_columns(
            df=df,
            columns_to_drop=columns_to_drop
        )

        X, y = separate_features_target(
            df=df,
            target_column=target_column
        )

        y_encoded, label_encoder = encode_target(y=y)

        numerical_columns, categorical_columns = get_feature_types(X=X)

        preprocessor = build_preprocessing_pipeline(
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns
        )

        return (
            X,
            y_encoded,
            preprocessor,
            label_encoder,
            numerical_columns,
            categorical_columns
        )

    except Exception as e:
        raise RuntimeError(f"Error in prepare_training_data: {e}")


def prepare_prediction_data(
    df: pd.DataFrame,
    columns_to_drop: list | None = None,
    target_column: str | None = None,
    expected_columns: list | None = None
) -> pd.DataFrame:
    """
    Prepare input data for prediction.

    Important:
    - This function does NOT fit or transform using sklearn preprocessor.
    - The MLflow-loaded model is already a full pipeline:
        preprocessor + trained model
    - So here we only clean raw input before passing it to model.predict().

    Args:
        df (pd.DataFrame): Raw input dataframe for prediction.
        columns_to_drop (list): Columns to drop, e.g. loan_id.
        target_column (str): Target column if accidentally present.
        expected_columns (list): Final expected feature columns.

    Returns:
        pd.DataFrame: Cleaned prediction dataframe.
    """

    try:
        df = df.copy()

        # Clean column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Clean string values
        df = clean_string_values(df=df)

        # Drop target column if accidentally present
        if target_column is not None and target_column in df.columns:
            df = df.drop(columns=[target_column])

        # Drop unwanted columns, for example loan_id
        df = drop_unwanted_columns(
            df=df,
            columns_to_drop=columns_to_drop
        )

        # Validate and reorder columns if expected_columns is provided
        if expected_columns is not None:
            missing_columns = [
                col for col in expected_columns
                if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing required prediction columns: {missing_columns}"
                )

            # Keep only expected columns and preserve training column order
            df = df[expected_columns]

        return df

    except Exception as e:
        raise RuntimeError(f"Error in prepare_prediction_data: {e}")

def main():
    """
    Optional standalone smoke test with a small dummy dataframe.
    No hardcoded project data path is used here.
    """

    try:
        sample_df = pd.DataFrame(
            {
                "loan_id": [1, 2, 3],
                "education": [" Graduate", "Not Graduate", " Graduate"],
                "self_employed": ["No", " Yes", "No"],
                "income_annum": [500000, 700000, 600000],
                "loan_amount": [200000, 300000, 250000],
                "loan_status": [" Approved", " Rejected", " Approved"]
            }
        )

        (
            X,
            y_encoded,
            preprocessor,
            label_encoder,
            numerical_columns,
            categorical_columns
        ) = prepare_training_data(
            df=sample_df,
            target_column="loan_status",
            columns_to_drop=["loan_id"]
        )

        print("Preprocessing module test successful.")
        print("X shape:", X.shape)
        print("y:", y_encoded)
        print("Numerical columns:", numerical_columns)
        print("Categorical columns:", categorical_columns)
        print("Target classes:", label_encoder.classes_)

    except Exception as e:
        print(f"Preprocessing module test failed: {e}")


if __name__ == "__main__":
    main()