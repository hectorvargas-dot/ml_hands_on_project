import os
import mlflow
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)

def init_mlflow_experiment(experiment_name: str, db_path: str, artifacts_dir: str):
    """Initializes tracking configurations and establishes backend experiment isolation."""
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=f"file://{artifacts_dir}"
        )
    mlflow.set_experiment(experiment_name)


def log_classification_curves(y_test: pd.Series, y_proba: pd.Series):
    """Generates, validates, and serializes performance curves directly to MLflow."""
    # ROC Curve Evaluation
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)

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
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    mlflow.log_figure(plt.gcf(), "pr_curve.png")
    plt.close()


def train_and_log_models(models: dict, preprocessor, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_test: pd.DataFrame, y_test: pd.Series, cat_cols: list, num_cols: list, pass_cols: list):
    """Runs data pipeline training workflows and handles reporting configurations."""
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            
            # Construct Pipeline Execution Node
            pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('model', model)
            ])

            # Fit Training Partition
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0)
            }

            # Evaluate probability metrics if supported by the model structure
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                metrics["pr_auc"] = average_precision_score(y_test, y_proba)
                log_classification_curves(y_test, y_proba)

            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())

            # Export Feature Schema Definitions
            fitted_preprocessor = pipeline.named_steps['preprocessing']
            output_features = fitted_preprocessor.get_feature_names_out()
            
            mlflow.log_dict(
                {
                    "input_features": cat_cols + num_cols + pass_cols,
                    "output_features": output_features.tolist()
                },
                "feature_schema.json"
            )

            # Export Raw JSON Class Performance Matrix Data
            mlflow.log_dict(classification_report(y_test, y_pred, output_dict=True, zero_division=0), "classification_report.json")
            mlflow.log_dict(confusion_matrix(y_test, y_pred).tolist(), "confusion_matrix.json")

            # Serialize and Track Model Binary
            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model", serialization_format="cloudpickle")
            print(f"Successfully processed and logged: {model_name}")


def generate_experiment_summary(experiment_name: str):
    """Aggregates active telemetry information and returns a structured DataFrame sorted by PR-AUC."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"Experiment '{experiment_name}' not found.")
        return None
        
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    target_columns = [
        "run_id", "tags.mlflow.runName", "metrics.accuracy", 
        "metrics.precision", "metrics.recall", "metrics.f1_score"
    ]
    
    # Safely append conditional probability metrics if logged inside target search space
    for col in ["metrics.roc_auc", "metrics.pr_auc"]:
        if col in runs_df.columns:
            target_columns.append(col)
            
    summary_df = runs_df[target_columns].copy()
    
    if "metrics.pr_auc" in summary_df.columns:
        summary_df = summary_df.sort_values(by="metrics.pr_auc", ascending=False)
        
    return summary_df