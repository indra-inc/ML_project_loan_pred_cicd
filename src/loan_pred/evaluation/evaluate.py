from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib

from pathlib import Path

def evaluate_trained_model(model, X_test, y_test) -> dict:
    """
    Evaluate fitted model pipeline.
    """

    try:
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }

        return metrics

    except Exception as e:
        raise RuntimeError(f"Error in evaluate_trained_model: {e}")


def save_model_artifact(model, model_dir: str, run_name: str) -> str:
    """
    Save trained model pipeline locally as .pkl file.
    """

    try:
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        saved_model_path = model_path / f"{run_name}_model.pkl"

        joblib.dump(model, saved_model_path)

        return str(saved_model_path)

    except Exception as e:
        raise RuntimeError(f"Error in save_model_artifact: {e}")
    

def save_report_artifacts(
    model,
    X_test,
    y_test,
    label_encoder,
    report_dir: str,
    run_name: str
) -> dict:
    """
    Save classification report and confusion matrix locally.
    """

    try:
        report_path = Path(report_dir)
        report_path.mkdir(parents=True, exist_ok=True)

        y_pred = model.predict(X_test)

        # -----------------------------
        # Classification report
        # -----------------------------
        class_report = classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_
        )

        classification_report_path = report_path / f"{run_name}_classification_report.txt"

        with open(classification_report_path, "w") as file:
            file.write(class_report) # type: ignore

        # -----------------------------
        # Confusion matrix plot
        # -----------------------------
        confusion_matrix_path = report_path / f"{run_name}_confusion_matrix.png"

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=label_encoder.classes_
        )

        plt.title(f"Confusion Matrix - {run_name}")
        plt.tight_layout()
        plt.savefig(confusion_matrix_path)
        plt.close()

        return {
            "classification_report_path": str(classification_report_path),
            "confusion_matrix_path": str(confusion_matrix_path),
        }

    except Exception as e:
        raise RuntimeError(f"Error in save_report_artifacts: {e}")
