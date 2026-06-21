
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from src.train.train_pipeline import run_train_pipeline
from src.predict.predict_pipeline import run_predict_pipeline
from src.predict.predict_utils import find_best_mlflow_runs
import uvicorn

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI()

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


@app.get(
    "/experiments/best_runs",
    response_model=list[BestRunResponse]
)
def get_best_runs():
    try:
        return find_best_mlflow_runs()

    except Exception as e:
        logger.exception("Failed retrieving best MLflow runs")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    experiment: str = Form("customer-churn-optuna-v2"),
    experiment_id: str = Form("4"),
    run_id: str = Form("7bad46961ca54823b74d72034247b02f"),
    run_name: str = Form("optuna_search_parent")
):
    try:

        df = pd.read_csv(file.file)

        predictions = run_predict_pipeline(
            df,
            experiment_name=experiment,
            experiment_id=experiment_id,
            run_id=run_id,
            run_name=run_name
        )

        return predictions

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/train")
async def train(
    file: UploadFile = File(...),
    experiment_name: str = Form("customer-churn-optuna")
):
    try:
        df = pd.read_csv(file.file)

        run_train_pipeline(
            df,
            experiment_name=experiment_name
        )

        return {
            "message": "Ingestion completed",
            "filename": file.filename,
            "rows": len(df),
            "experiment_name": experiment_name
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)