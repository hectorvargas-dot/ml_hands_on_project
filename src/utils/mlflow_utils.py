import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

def init_mlflow_experiment(
    experiment_name: str,
    db_path: str,
    artifacts_dir: str
) -> str:
    """Initializes tracking configurations and establishes backend experiment isolation."""
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=Path(artifacts_dir).resolve().as_uri()
        )

    experiment = mlflow.get_experiment_by_name(experiment_name)

    mlflow.set_experiment(experiment_name)

    return experiment.experiment_id

def log_classification_curves(
    y_true: pd.Series | np.ndarray, y_proba: pd.Series | np.ndarray
) -> None:
    """Generates, validates, and serializes performance curves directly to MLflow."""
    # ROC Curve Evaluation
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    mlflow.log_figure(plt.gcf(), "roc_curve.png")
    plt.close()

    # Precision-Recall Curve Evaluation
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    mlflow.log_figure(plt.gcf(), "pr_curve.png")
    plt.close()


def log_model_evaluation(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    input_features: list[str],
    artifact_path: str = "model",
) -> dict[str, float]:
    """Computes test predictions, logs classification metrics, curves, schema, reports, and serializes the pipeline."""
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }

    # Evaluate probability metrics if supported by the model structure
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        metrics["pr_auc"] = average_precision_score(y_test, y_proba)
        log_classification_curves(y_test, y_proba)

    mlflow.log_metrics(metrics)

    # Log model params from the pipeline's final estimator
    if "model" in pipeline.named_steps:
        mlflow.log_params(pipeline.named_steps["model"].get_params())

    # Export Feature Schema Definitions if preprocessor is present
    if "preprocessing" in pipeline.named_steps:
        fitted_preprocessor = pipeline.named_steps["preprocessing"]
        output_features = fitted_preprocessor.get_feature_names_out().tolist()
        mlflow.log_dict(
            {"input_features": input_features, "output_features": output_features},
            "feature_schema.json",
        )

    # Export Raw JSON Class Performance Matrix Data
    mlflow.log_dict(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "classification_report.json",
    )
    mlflow.log_dict(confusion_matrix(y_test, y_pred).tolist(), "confusion_matrix.json")

    # Serialize and Track Model Binary
    mlflow.sklearn.log_model(
        sk_model=pipeline, artifact_path=artifact_path, serialization_format="cloudpickle"
    )
    return metrics


def get_experiment_summary(experiment_name: str) -> pd.DataFrame | None:
    """Aggregates active telemetry information and returns a structured DataFrame sorted by PR-AUC."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs_df.empty:
        return runs_df

    target_columns = [
        "run_id",
        "tags.mlflow.runName",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.f1_score",
    ]

    print(runs_df.columns)

    # Safely append conditional probability metrics if logged inside target search space
    for col in ["metrics.roc_auc", "metrics.pr_auc_cv"]:
        if col in runs_df.columns:
            target_columns.append(col)

    # Include start_time if present
    if "start_time" in runs_df.columns:
        target_columns.append("start_time")

    # Filter columns that are actually present
    cols_to_keep = [col for col in target_columns if col in runs_df.columns]
    summary_df = runs_df[cols_to_keep].copy()

    if "metrics.pr_auc_cv" in summary_df.columns:
        summary_df = summary_df.sort_values(by="metrics.pr_auc_cv", ascending=False)

    return summary_df


def remove_recent_runs(experiment_name: str, count: int) -> None:
    """Deletes the latest sequence of tracking iterations from the active database log."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return

    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"]
    )

    for run_id in runs.head(count)["run_id"]:
        client.delete_run(run_id)
        print(f"Purged tracking history for run reference element: {run_id}")


def load_best_run_pipeline(
    experiment_name: str, db_path: str, order_by_metric: str = "metrics.pr_auc DESC"
):
    """
    Connects to MLflow tracking database, searches for the best run based on
    the provided metric, and returns the full serialized pipeline along with run details.
    """
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in the tracking database.")

    # Retrieve the top performing run
    best_runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], order_by=[order_by_metric], max_results=1
    )

    if best_runs_df.empty:
        raise RuntimeError(f"No runs found inside the experiment '{experiment_name}'.")

    best_run = best_runs_df.iloc[0]

    # Load the serialized sklearn pipeline architecture
    model_uri = f"runs:/{best_run.run_id}/model"
    pipeline = mlflow.sklearn.load_model(model_uri)

    return pipeline, best_run

def load_best_run_pipeline_by_run_name(
    experiment_name: str,
    db_path: str,
    run_name: str,
    order_by_metric: str = "metrics.pr_auc DESC",
):
    """
    Connects to MLflow tracking database, searches for the best run
    with the specified mlflow.runName tag, and returns the full
    serialized pipeline along with run details.
    """
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found in the tracking database."
        )

    best_runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=[order_by_metric],
        max_results=1,
    )

    if best_runs_df.empty:
        raise RuntimeError(
            f"No runs found in experiment '{experiment_name}' "
            f"with tags.mlflow.runName='{run_name}'."
        )

    best_run = best_runs_df.iloc[0]

    model_uri = f"runs:/{best_run.run_id}/model"
    pipeline = mlflow.sklearn.load_model(model_uri)

    return pipeline, best_run
