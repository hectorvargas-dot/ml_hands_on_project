import mlflow
import pandas as pd


def run_predict_pipeline(
    df: pd.DataFrame,
    experiment_name: str,
    experiment_id: str,
    run_id: str,
    run_name: str
):

    model_uri = f"runs:/{run_id}/best_model"

    pipeline = mlflow.sklearn.load_model(model_uri)

    customer_ids = df["CustomerId"].copy()

    # Remove identifier column if it was not used during training
    X = df.drop(columns=["CustomerId"])

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    results = [
        {
            "CustomerId": int(customer_id),
            "prediction": int(pred),
            "churn_probability": round(float(prob), 4)
        }
        for customer_id, pred, prob in zip(
            customer_ids,
            y_pred,
            y_proba
        )
    ]

    return results