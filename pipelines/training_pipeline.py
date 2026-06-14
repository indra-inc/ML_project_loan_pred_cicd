"""
training_pipeline.py

Responsibility:
- Read config files
- Load training data
- Prepare preprocessing pipeline
- Split data into train/test
- Train selected model only: random_forest or xgboost
- Override model params from command line
- Evaluate model
- Save local artifacts in artifacts/models and artifacts/reports
- Log params, metrics, tags, artifacts, and model into MLflow

Run examples:

Random Forest:
    PYTHONPATH=$PWD/src python pipelines/training_pipeline.py \
        --model_name random_forest \
        --run_name rf_run_01 \
        --n_estimators 200 \
        --max_depth 10

XGBoost:
    PYTHONPATH=$PWD/src python pipelines/training_pipeline.py \
        --model_name xgboost \
        --run_name xgb_run_01 \
        --n_estimators 300 \
        --max_depth 6 \
        --learning_rate 0.05
"""

import argparse
import copy
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


from sklearn.model_selection import train_test_split

from loan_pred.loader.data_loader import load_training_data
from loan_pred.processing.preprocessing_pipeline import prepare_training_data
from loan_pred.training.train import train_model_pipeline
from loan_pred.tracking.mlflow_tracker import create_experiment_run
from loan_pred.evaluation.evaluate import evaluate_trained_model, save_report_artifacts, save_model_artifact


def read_yaml_config(config_path: str) -> dict:
    """
    Read YAML config file.
    """

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError(f"Config file is empty: {config_path}")

        return config

    except Exception as e:
        raise RuntimeError(f"Error in read_yaml_config: {e}")


def override_model_params(
    base_params: Dict[str, Any],
    custom_params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Override default model params from command-line values.

    Rule:
    - If CLI value is None, keep config value.
    - If CLI key is not valid for selected model, raise error.
    """

    try:
        final_params = base_params.copy()

        if custom_params:
            for key, value in custom_params.items():
                if value is not None:
                    if key not in final_params:
                        raise ValueError(
                            f"Parameter '{key}' is not valid for this model. "
                            f"Allowed params: {list(final_params.keys())}"
                        )

                    final_params[key] = value

        return final_params

    except Exception as e:
        raise RuntimeError(f"Error in override_model_params: {e}")



def train_and_log_single_model(
    model_name: str,
    run_name: str,
    model_params: dict,
    experiment_name: str,
    registered_model_name: str,
    tracking_uri: str,
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    label_encoder,
    model_dir: str,
    report_dir: str,
    extra_tags: Optional[dict] = None
) -> str:
    """
    Train selected model and log it into MLflow.
    """

    try:
        model_preprocessor = copy.deepcopy(preprocessor)

        fitted_model_pipeline = train_model_pipeline(
            X_train=X_train,
            y_train=y_train,
            preprocessor=model_preprocessor,
            model_name=model_name,
            model_params=model_params
        )

        metrics = evaluate_trained_model(
            model=fitted_model_pipeline,
            X_test=X_test,
            y_test=y_test
        )

        saved_model_path = save_model_artifact(
            model=fitted_model_pipeline,
            model_dir=model_dir,
            run_name=run_name
        )

        artifact_paths = save_report_artifacts(
            model=fitted_model_pipeline,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
            report_dir=report_dir,
            run_name=run_name
        )

        final_tags = {
            "model_name": model_name,
            "run_name": run_name,
            "local_model_path": saved_model_path,
            "execution_mode": "command_line",
        }

        if extra_tags:
            final_tags.update(extra_tags)

        run_id = create_experiment_run(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name,
            model_name=model_name,
            model=fitted_model_pipeline,
            run_metrics=metrics,
            run_params=model_params,
            registered_model_name=registered_model_name,
            confusion_matrix_path=artifact_paths["confusion_matrix_path"],
            classification_report_path=artifact_paths["classification_report_path"],
            extra_tags=final_tags
        )

        print(f"{model_name} training completed.")
        print(f"Run name: {run_name}")
        print(f"Metrics: {metrics}")
        print(f"Local model saved at: {saved_model_path}")
        print(f"Reports saved at: {report_dir}")

        return run_id

    except Exception as e:
        raise RuntimeError(f"Error in train_and_log_single_model for {model_name}: {e}")


def run_training_pipeline(
    dev_config_path: str,
    mlflow_config_path: str,
    model_name: str,
    run_name: str,
    custom_params: Optional[dict] = None
) -> None:
    """
    Main training pipeline.
    """

    try:
        # -----------------------------
        # 1. Read configs
        # -----------------------------
        dev_config = read_yaml_config(dev_config_path)
        mlflow_config = read_yaml_config(mlflow_config_path)

        if model_name not in dev_config["models"]:
            raise ValueError(
                f"Invalid model_name: {model_name}. "
                f"Available models: {list(dev_config['models'].keys())}"
            )

        # -----------------------------
        # 2. Prepare model params
        # -----------------------------
        base_model_params = dev_config["models"][model_name]

        model_params = override_model_params(
            base_params=base_model_params,
            custom_params=custom_params
        )

        print(f"Selected model: {model_name}")
        print(f"Run name: {run_name}")
        print(f"Final model params: {model_params}")

        # -----------------------------
        # 3. Load data
        # -----------------------------
        data_path = dev_config["data"]["train_path"]
        target_column = dev_config["data"]["target_column"]

        df = load_training_data(
            file_path=data_path,
            target_column=target_column
        )

        # -----------------------------
        # 4. Prepare data
        # -----------------------------
        columns_to_drop = dev_config.get("preprocessing", {}).get(
            "columns_to_drop",
            ["loan_id"]
        )

        (
            X,
            y,
            preprocessor,
            label_encoder,
            numerical_columns,
            categorical_columns
        ) = prepare_training_data(
            df=df,
            target_column=target_column,
            columns_to_drop=columns_to_drop
        )

        print("Data preparation completed.")
        print("Feature shape:", X.shape)
        print("Target shape:", y.shape) # pyright: ignore[reportAttributeAccessIssue]
        print("Numerical columns:", numerical_columns)
        print("Categorical columns:", categorical_columns)
        print("Target classes:", label_encoder.classes_)

        # -----------------------------
        # 5. Train-test split
        # -----------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=dev_config["training"]["test_size"],
            random_state=dev_config["training"]["random_state"],
            stratify=y
        )

        # -----------------------------
        # 6. MLflow + artifact config
        # -----------------------------
        tracking_uri = mlflow_config["mlflow"]["tracking_uri"]
        experiment_name = mlflow_config["experiment_name"]
        registered_model_name = mlflow_config["registered_model_name"]

        model_dir = dev_config["artifacts"]["model_dir"]
        report_dir = dev_config["artifacts"]["report_dir"]

        # -----------------------------
        # 7. Train selected model only
        # -----------------------------
        train_and_log_single_model(
            model_name=model_name,
            run_name=run_name,
            model_params=model_params,
            experiment_name=experiment_name,
            registered_model_name=registered_model_name,
            tracking_uri=tracking_uri,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            preprocessor=preprocessor,
            label_encoder=label_encoder,
            model_dir=model_dir,
            report_dir=report_dir,
            extra_tags={
                "project": dev_config["project"]["name"],
                "environment": dev_config["project"]["environment"],
                "source_type": dev_config["data"]["source_type"],
            }
        )

        print("Training pipeline completed successfully.")

    except Exception as e:
        raise RuntimeError(f"Error in run_training_pipeline: {e}")


def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description="Run MLflow training pipeline for loan approval prediction"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["random_forest", "xgboost"],
        help="Model to train"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="MLflow run name"
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=None,
        help="Number of estimators"
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Maximum tree depth"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate, mainly for XGBoost"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=None,
        help="Random state"
    )

    return parser.parse_args()

def main():
    """
    Entry point.
    """

    try:
        args = parse_args()

        custom_params = {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "random_state": args.random_state,
        }

        run_training_pipeline(
            dev_config_path="configs/dev_config.yaml",
            mlflow_config_path="configs/mlflow_config.yaml",
            model_name=args.model_name,
            run_name=args.run_name,
            custom_params=custom_params
        )

    except Exception as e:
        print(f"Training pipeline failed: {e}")


if __name__ == "__main__":
    main()


# python pipelines/training_pipeline.py   --model_name xgboost   --run_name xgb_run_01   --n_estimators 300   --max_depth 6   --learning_rate 0.05