"""
mlflow_tracker.py

Responsibility:
- Set MLflow tracking URI
- Set MLflow experiment
- Log parameters
- Log metrics
- Log tags
- Log artifacts
- Log sklearn model/pipeline

Note:
- This file does not train the model.
- This file does not evaluate the model.
- It only logs the final trained model and metadata into MLflow.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient    # used for programmatic control of MLflow

def setup_mlflow(
    tracking_uri: str,
    experiment_name: str
) -> None:
    """
    Set MLflow tracking URI and experiment name.

    Args:
        tracking_uri (str): MLflow tracking server URI.
        experiment_name (str): MLflow experiment name.
    """

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    except Exception as e:
        raise RuntimeError(f"Error in setup_mlflow: {e}")


def log_params(run_params: Optional[Dict[str, Any]] = None) -> None:
    """
    Log model hyperparameters into MLflow.

    Args:
        run_params (dict): Model parameters.
    """

    try:
        if run_params:
            for param_name, param_value in run_params.items():
                mlflow.log_param(param_name, param_value)

    except Exception as e:
        raise RuntimeError(f"Error in log_params: {e}")


def log_metrics(run_metrics: Dict[str, float]) -> None:
    """
    Log evaluation metrics into MLflow.

    Args:
        run_metrics (dict): Metrics dictionary.
    """

    try:
        if not run_metrics:
            raise ValueError("run_metrics cannot be empty.")

        for metric_name, metric_value in run_metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))

    except Exception as e:
        raise RuntimeError(f"Error in log_metrics: {e}")


def log_tags(
    model_name: str,
    extra_tags: Optional[Dict[str, str]] = None
) -> None:
    """
    Log useful tags into MLflow.

    Args:
        model_name (str): Model name.
        extra_tags (dict): Additional tags.
    """

    try:
        default_tags = {
            "developer": "Indra",
            "project": "loan_approval_prediction",
            "model": model_name,
            "problem_type": "binary_classification",
            "experiment_stage": "POC"
        }

        if extra_tags:
            default_tags.update(extra_tags)

        mlflow.set_tags(default_tags)

    except Exception as e:
        raise RuntimeError(f"Error in log_tags: {e}")


def log_artifact_file(
    artifact_path: Optional[str] = None,
    artifact_folder_name: str = "artifacts"
) -> None:
    """
    Log a single artifact file into MLflow.

    Example:
        confusion matrix image
        classification report file
        metrics json file

    Args:
        artifact_path (str): Local artifact file path.
        artifact_folder_name (str): MLflow artifact folder name.
    """

    try:
        if artifact_path is None:
            return

        path = Path(artifact_path)

        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

        mlflow.log_artifact(
            local_path=str(path),
            artifact_path=artifact_folder_name
        )

    except Exception as e:
        raise RuntimeError(f"Error in log_artifact_file: {e}")


def log_model(
    model,
    registered_model_name: Optional[str] = None,
    artifact_path: str = "model"
) -> None:
    """
    Log sklearn model or sklearn pipeline into MLflow.

    Args:
        model: Trained sklearn model or sklearn pipeline.
        registered_model_name (str): Optional MLflow registered model name.
        artifact_path (str): Artifact folder name inside MLflow.
    """

    try:
        mlflow.sklearn.log_model( # type: ignore
            sk_model=model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )

    except Exception as e:
        raise RuntimeError(f"Error in log_model: {e}")



def create_experiment_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    model_name: str,
    model,
    run_metrics: Dict[str, float],
    run_params: Optional[Dict[str, Any]] = None,
    registered_model_name: Optional[str] = None,
    confusion_matrix_path: Optional[str] = None,
    classification_report_path: Optional[str] = None,
    extra_tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Create one MLflow run and log all required details.

    Args:
        tracking_uri (str): MLflow tracking server URI.
        experiment_name (str): MLflow experiment name.
        run_name (str): Current run name.
        model_name (str): Model name, e.g. random_forest / xgboost.
        model: Trained sklearn model or full sklearn pipeline.
        run_metrics (dict): Evaluation metrics.
        run_params (dict): Model parameters.
        registered_model_name (str): Optional registered model name.
        confusion_matrix_path (str): Optional confusion matrix artifact path.
        classification_report_path (str): Optional classification report artifact path.
        extra_tags (dict): Optional extra tags.

    Returns:
        str: MLflow run ID.
    """

    try:
        setup_mlflow(
            tracking_uri=tracking_uri,
            experiment_name=experiment_name
        )

        experiment = mlflow.get_experiment_by_name(experiment_name)

        print(f"Experiment name: {experiment_name}")
        print(f"Experiment ID: {experiment.experiment_id}") # type: ignore

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            log_params(run_params=run_params)

            log_metrics(run_metrics=run_metrics)

            log_tags(
                model_name=model_name,
                extra_tags=extra_tags
            )

            log_artifact_file(
                artifact_path=confusion_matrix_path,
                artifact_folder_name="confusion_matrix"
            )

            log_artifact_file(
                artifact_path=classification_report_path,
                artifact_folder_name="classification_report"
            )

            log_model(
                model=model,
                registered_model_name=registered_model_name,
                artifact_path="model"
            )

        client = MlflowClient()
        completed_run = client.get_run(run_id)

        print(
            f"Run '{run_name}' with Run ID '{run_id}' "
            f"is logged to Experiment '{experiment_name}' "
            f"with status: {completed_run.info.status}"
        )

        return run_id

    except Exception as e:
        raise RuntimeError(f"Error in create_experiment_run: {e}")
