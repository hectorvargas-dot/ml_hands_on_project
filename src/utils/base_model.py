import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src import mlflow_utils


def train_and_log_models(
    models: dict,
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_cols: list,
    num_cols: list,
    pass_cols: list,
) -> None:
    """Runs data pipeline training workflows and handles reporting configurations using mlflow_utils."""
    input_features = cat_cols + num_cols + pass_cols
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Construct Pipeline Execution Node
            pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])

            # Fit Training Partition
            pipeline.fit(X_train, y_train)

            # Delegate metric logging and serialization to mlflow_utils
            mlflow_utils.log_model_evaluation(
                pipeline=pipeline,
                X_test=X_test,
                y_test=y_test,
                input_features=input_features,
                artifact_path="model",
            )
            print(f"Successfully processed and logged: {model_name}")
