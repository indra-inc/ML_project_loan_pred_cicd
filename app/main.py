"""
main.py

FastAPI app for loan approval prediction.

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import app_config
from app.services import load_registered_model, predict_single_record, predict_batch_records


# Global model object.
# It will be loaded once when FastAPI starts.
model = None


class LoanPredictionRequest(BaseModel):
    """
    Request schema for single loan prediction.
    """

    loan_id: Optional[int] = Field(default=None)

    no_of_dependents: int
    education: str
    self_employed: str

    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float

    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float


class LoanPredictionResponse(BaseModel):
    """
    Response schema for single loan prediction.
    """

    encoded_prediction: int
    prediction: str
    registered_model_name: str
    model_version: Optional[str] = None
    model_alias: Optional[str] = None


class BatchLoanPredictionRequest(BaseModel):
    """
    Request schema for batch prediction.
    """

    records: list[LoanPredictionRequest]


class BatchLoanPredictionResponse(BaseModel):
    """
    Response schema for batch prediction.
    """

    predictions: list[dict]
    registered_model_name: str
    model_version: Optional[str] = None
    model_alias: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI startup/shutdown lifecycle.

    Model is loaded once during application startup.
    """

    global model

    try:
        model = load_registered_model()
        yield

    except Exception as e:
        raise RuntimeError(f"Application startup failed: {e}")


app = FastAPI(
    title=app_config.app_name,
    description="FastAPI service for loan approval prediction using MLflow registered model",
    version=app_config.app_version,
    lifespan=lifespan
)


@app.get("/")
def root():
    """
    Root endpoint.
    """

    return {
        "message": "🚀 Loan Approval Prediction API is running",
        "environment": app_config.app_env
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "registered_model_name": app_config.registered_model_name,
        "model_version": app_config.model_version,
        "model_alias": app_config.model_alias,
        "mlflow_tracking_uri": app_config.mlflow_tracking_uri
    }


@app.post("/predict", response_model=LoanPredictionResponse)
def predict(request: LoanPredictionRequest):
    """
    Predict loan approval for a single request.
    """

    try:
        if model is None:
            raise RuntimeError("Model is not loaded.")

        input_data = request.model_dump()

        result = predict_single_record(
            model=model,
            input_data=input_data
        )

        return LoanPredictionResponse(
            encoded_prediction=result["encoded_prediction"],
            prediction=result["prediction"],
            registered_model_name=app_config.registered_model_name,
            model_version=app_config.model_version,
            model_alias=app_config.model_alias
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )


@app.post("/batch_predict", response_model=BatchLoanPredictionResponse)
def batch_predict(request: BatchLoanPredictionRequest):
    """
    Predict loan approval for multiple records.
    """

    try:
        if model is None:
            raise RuntimeError("Model is not loaded.")

        input_records = [
            record.model_dump()
            for record in request.records
        ]

        results = predict_batch_records(
            model=model,
            input_records=input_records
        )

        return BatchLoanPredictionResponse(
            predictions=results,
            registered_model_name=app_config.registered_model_name,
            model_version=app_config.model_version,
            model_alias=app_config.model_alias
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {e}"
        )


# uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# ## Test case values:
# # for single prediction:
# {
#   "loan_id": 1,
#   "no_of_dependents": 2,
#   "education": "Graduate",
#   "self_employed": "No",
#   "income_annum": 9600000,
#   "loan_amount": 29900000,
#   "loan_term": 12,
#   "cibil_score": 778,
#   "residential_assets_value": 2400000,
#   "commercial_assets_value": 17600000,
#   "luxury_assets_value": 22700000,
#   "bank_asset_value": 8000000
# }

# # For batch prediction
# {
#   "records": [
#     {
#       "loan_id": 1,
#       "no_of_dependents": 2,
#       "education": "Graduate",
#       "self_employed": "No",
#       "income_annum": 9600000,
#       "loan_amount": 29900000,
#       "loan_term": 12,
#       "cibil_score": 778,
#       "residential_assets_value": 2400000,
#       "commercial_assets_value": 17600000,
#       "luxury_assets_value": 22700000,
#       "bank_asset_value": 8000000
#     },
#     {
#       "loan_id": 2,
#       "no_of_dependents": 0,
#       "education": "Not Graduate",
#       "self_employed": "Yes",
#       "income_annum": 4100000,
#       "loan_amount": 12200000,
#       "loan_term": 8,
#       "cibil_score": 417,
#       "residential_assets_value": 2700000,
#       "commercial_assets_value": 2200000,
#       "luxury_assets_value": 8800000,
#       "bank_asset_value": 3300000
#     },
#     {
#       "loan_id": 3,
#       "no_of_dependents": 4,
#       "education": "Graduate",
#       "self_employed": "No",
#       "income_annum": 9100000,
#       "loan_amount": 29700000,
#       "loan_term": 20,
#       "cibil_score": 506,
#       "residential_assets_value": 7100000,
#       "commercial_assets_value": 4500000,
#       "luxury_assets_value": 33300000,
#       "bank_asset_value": 12800000
#     }
#   ]
# }