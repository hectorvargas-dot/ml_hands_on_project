import logging
from datetime import datetime
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import pandas as pd
import uvicorn

from src.predict.predict_pipeline import run_predict_pipeline
from src.predict.predict_utils import find_best_mlflow_runs
from src.train.train_pipeline import run_train_pipeline
from src.drift_monitoring.feature_drift import feature_drift
from src.drift_monitoring.target_drift import target_drift

# Configure structured application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Churn Platform API",
    description="FastAPI delivery tier handling training, inference, and drift metrics logging.",
    version="1.0.0"
)


@app.on_event("startup")
def startup_event():
    logger.info("API started successfully")


@app.get("/")
async def root():
    return {"message": "Hello World"}


class BestRunResponse(BaseModel):
    experiment: str
    experiment_id: str
    run_id: str
    run_name: str
    test_pr_auc: float
    date: str


@app.get("/experiments/best_runs", response_model=list[BestRunResponse])
def get_best_runs():
    try:
        return find_best_mlflow_runs()
    except Exception as e:
        logger.exception("Failed retrieving best MLflow runs")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    experiment: str = Form("customer-churn-optuna-v2"),
    experiment_id: str = Form("4"),
    run_id: str = Form("7bad46961ca54823b74d72034247b02f"),
    run_name: str = Form("optuna_search_parent"),
):
    try:
        df = pd.read_csv(file.file)
        predictions = run_predict_pipeline(
            df,
            experiment_name=experiment,
            experiment_id=experiment_id,
            run_id=run_id,
            run_name=run_name,
        )
        return predictions
    except Exception as e:
        logger.exception("Inference execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    experiment_name: str = Form("customer-churn-optuna"),
):
    try:
        df = pd.read_csv(file.file)
        run_train_pipeline(df, experiment_name=experiment_name)
        return {
            "message": "Ingestion completed",
            "filename": file.filename,
            "rows": len(df),
            "experiment_name": experiment_name,
        }
    except Exception as e:
        logger.exception("Training pipeline execution aborted")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor/feature-drift")
async def monitor_drift(
    reference_file: UploadFile = File(..., description="Baseline historical validation/training data (CSV)"),
    current_file: UploadFile = File(..., description="Current production/inference window data payloads (CSV)"),
    run_id: str = Form(..., description="Target MLflow Run ID containing the reference pipeline structure")
):
    """Parses reference and production datasets to calculate and return feature drift analytics."""
    logger.info(f"Received data drift verification request mapped against Run ID: {run_id}")
    try:
        # Read files safely into dataframes
        reference_df = pd.read_csv(reference_file.file)
        current_df = pd.read_csv(current_file.file)

        logger.info(
            f"Parsing frames for drift audit. Reference rows: {len(reference_df)}, "
            f"Current inference payload rows: {len(current_df)}"
        )

        # Run feature routing and non-parametric statistical tests
        report_df, global_drift_triggered = feature_drift(
            reference_df=reference_df,
            current_df=current_df,
            run_id=run_id
        )

        # Convert the structural dataframe report into an API friendly list of records
        report_records = report_df.to_dict(orient="records")

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "global_drift_detected": global_drift_triggered,
            "metrics_report": report_records
        }

    except Exception as e:
        logger.exception("Data drift monitoring runtime calculation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Feature drift monitoring system error: {str(e)}"
        )

@app.post("/monitor/target-drift")
async def monitor_target_drift(
    reference_file: UploadFile = File(..., description="Historical labeled dataset (CSV)"),
    current_file: UploadFile = File(..., description="Current labeled dataset (CSV)"),
    target_column: str = Form("Exited", description="Target column name")
):
    """Calculates target drift between two labeled datasets."""

    try:
        reference_df = pd.read_csv(reference_file.file)
        current_df = pd.read_csv(current_file.file)

        if target_column not in reference_df.columns:
            raise HTTPException(
                status_code=400,
                detail=(f"Target column '{target_column}' not found in reference dataset."),
            )

        if target_column not in current_df.columns:
            raise HTTPException(
                status_code=400,
                detail=(f"Target column '{target_column}' not found in current dataset."),
            )

        report = target_drift(
            reference_target=reference_df[target_column],
            current_target=current_df[target_column],
        )

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "target_column": target_column,
            "target_drift": report,
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception("Target drift monitoring failed")

        raise HTTPException(
            status_code=500,
            detail=f"Target drift monitoring error: {str(e)}",
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
