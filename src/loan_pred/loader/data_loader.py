"""
data_loader.py

Responsibility:
- Load CSV data
- Clean column names
- Validate target column
- Return cleaned dataframe

Note:
- No hardcoded file path here.
- File path will come from dev_config.yaml through training_pipeline.py.
"""

from pathlib import Path
import pandas as pd


def load_csv_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV file into pandas DataFrame.
    """

    try:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if path.suffix.lower() != ".csv":
            raise ValueError(f"Expected .csv file, but got: {path.suffix}")

        df = pd.read_csv(path)

        if df.empty:
            raise ValueError(f"CSV file is empty: {file_path}")

        return df

    except Exception as e:
        raise RuntimeError(f"Error in load_csv_data: {e}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names:
    - remove leading/trailing spaces
    - convert to lowercase
    - replace spaces with underscore
    """

    try:
        df = df.copy()

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        return df

    except Exception as e:
        raise RuntimeError(f"Error in clean_column_names: {e}")


def validate_target_column(df: pd.DataFrame, target_column: str) -> None:
    """
    Validate whether target column exists in dataframe.
    """

    try:
        if target_column not in df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

    except Exception as e:
        raise RuntimeError(f"Error in validate_target_column: {e}")


def load_training_data(file_path: str, target_column: str) -> pd.DataFrame:
    """
    Main function to load and validate training data.
    """

    try:
        df = load_csv_data(file_path=file_path)
        df = clean_column_names(df=df)
        validate_target_column(df=df, target_column=target_column)

        return df

    except Exception as e:
        raise RuntimeError(f"Error in load_training_data: {e}")


def main():
    """
    Optional standalone test using command-line arguments.

    Example:
        python src/loan_pred/loader/data_loader.py \
            --file_path data/loan_approval_dataset.csv \
            --target_column loan_status
    """

    try:
        import argparse

        parser = argparse.ArgumentParser(description="Test data loader module")

        parser.add_argument(
            "--file_path",
            type=str,
            required=True,
            help="Path of the CSV file"
        )

        parser.add_argument(
            "--target_column",
            type=str,
            required=True,
            help="Target column name"
        )

        args = parser.parse_args()

        df = load_training_data(
            file_path=args.file_path,
            target_column=args.target_column
        )

        print("Data loaded successfully.")
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head())

    except Exception as e:
        print(f"Data loader test failed: {e}")


if __name__ == "__main__":
    main()