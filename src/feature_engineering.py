import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import mlflow
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report, 
    confusion_matrix, roc_curve, precision_recall_curve
)

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer for modular Churn feature generation."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Binary Flag Identifiers
        X["no_balance"] = (X["Balance"] == 0)
        X["NumOfProducts_1"] = (X["NumOfProducts"] == 1)
        X["NumOfProducts_2"] = (X["NumOfProducts"] == 2)
        
        # Polynomial & Interaction Terms
        X["Balance_x_Tenure"] = X["Balance"] * X["Tenure"]
        X["Age_x_IsActive"] = X["Age"] * X["IsActiveMember"]

        # Financial & Engagement Ratios
        X["Balance_to_Salary"] = X["Balance"] / (X["EstimatedSalary"] + 1)
        X["Balance_per_Product"] = X["Balance"] / (X["NumOfProducts"] + 1)
        X["Salary_per_Product"] = X["EstimatedSalary"] / (X["NumOfProducts"] + 1)
        X["CreditScore_per_Age"] = X["CreditScore"] / (X["Age"] + 1)
        X["Tenure_per_Age"] = X["Tenure"] / (X["Age"] + 1)

        # Behavioral Cross-Products
        X["Inactive_x_Balance"] = (1 - X["IsActiveMember"]) * X["Balance"]
        X["Inactive_x_Age"] = (1 - X["IsActiveMember"]) * X["Age"]
        X["Products_x_Active"] = X["NumOfProducts"] * X["IsActiveMember"]

        # Monetary Accumulations & Non-linear scaling
        X["Balance_plus_Salary"] = X["Balance"] + X["EstimatedSalary"]
        X["WealthScore"] = 0.6 * X["Balance"] + 0.4 * X["EstimatedSalary"]
        X["CreditScore_x_Age"] = X["CreditScore"] * X["Age"]
        X["LogBalance"] = np.log1p(X["Balance"])
        X["LogAge"] = np.log1p(X["Age"])

        # Polynomial Degrees
        X["Age2"] = X["Age"] ** 2
        X["Balance2"] = X["Balance"] ** 2
        X["Tenure2"] = X["Tenure"] ** 2

        # Temporal Product Densities
        X["Products_per_Tenure"] = X["NumOfProducts"] / (X["Tenure"] + 1)
        X["Balance_per_Tenure"] = X["Balance"] / (X["Tenure"] + 1)

        return X


def init_mlflow_experiment(experiment_name: str, db_path: str, artifacts_dir: str):
    """Sets up connection parameters and creates experiment contexts inside MLflow."""
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(name=experiment_name, artifact_location=f"file://{artifacts_dir}")
    mlflow.set_experiment(experiment_name)


def log_classification_curves(y_true, y_proba):
    """Saves visualization summaries directly to the active MLflow run tracking window."""
    # ROC Curve
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

    # Precision-Recall Curve
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


def train_and_log_pipeline(models: dict, preprocessor, X_train, y_train, X_test, y_test, input_feature_list: list):
    """Trains arbitrary configurations and outputs artifacts, schemas, and metrics."""
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            pipeline = Pipeline([
                ("feature_engineering", FeatureEngineer()),
                ("preprocessing", preprocessor),
                ("model", model)
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0)
            }

            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                metrics["pr_auc"] = average_precision_score(y_test, y_proba)
                log_classification_curves(y_test, y_proba)

            mlflow.log_metrics(metrics)
            mlflow.log_params(model.get_params())

            # Schema Extraction Tracking
            fitted_prep = pipeline.named_steps['preprocessing']
            mlflow.log_dict(
                {"input_features": input_feature_list, "output_features": fitted_prep.get_feature_names_out().tolist()},
                "feature_schema.json"
            )
            mlflow.log_dict(classification_report(y_test, y_pred, output_dict=True, zero_division=0), "classification_report.json")
            mlflow.log_dict(confusion_matrix(y_test, y_pred).tolist(), "confusion_matrix.json")
            mlflow.sklearn.log_model(sk_model=pipeline, name="model", serialization_format="cloudpickle")

            print(f"Finished Logging Pipeline Execution: {model_name} | PR-AUC: {metrics.get('pr_auc', 0.0):.4f}")


def get_experiment_summary(experiment_name: str):
    """Returns a formatted tracking summary sorted by PR-AUC performance metrics."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    keep_cols = [
        "run_id", "tags.mlflow.runName", "metrics.accuracy", "metrics.precision",
        "metrics.recall", "metrics.f1_score", "metrics.roc_auc", "metrics.pr_auc", "start_time"
    ]
    return runs_df[[c for c in keep_cols if c in runs_df.columns]].sort_values(by="metrics.pr_auc", ascending=False)


def remove_recent_runs(experiment_name: str, count: int):
    """Deletes the latest sequence of tracking iterations from the active database log."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
    
    for run_id in runs.head(count)["run_id"]:
        client.delete_run(run_id)
        print(f"Purged tracking history for run reference element: {run_id}")


def evaluate_engineered_features(engineered_features: list, base_nomod_columns: list, dummyfy_columns: list, 
                                 norm_std_columns: list, model, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Evaluates the incremental performance contribution of engineered features one-by-one."""
    results = []

    # Target Baseline Setup Evaluation Block
    baseline_prep = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), dummyfy_columns),
        ('num', StandardScaler(), norm_std_columns),
        ('pass', 'passthrough', base_nomod_columns)
    ], remainder='drop')

    baseline_pipe = Pipeline([
        ("feature_engineering", FeatureEngineer()),
        ("preprocessing", baseline_prep),
        ("model", clone(model))
    ])
    baseline_pipe.fit(X_train, y_train)
    base_proba = baseline_pipe.predict_proba(X_test)[:, 1]
    results.append({"feature_added": "BASELINE", "pr_auc": average_precision_score(y_test, base_proba)})

    # Incremental Singular Feature Search Loop Evaluation Block
    for feature in engineered_features:
        current_numeric_scope = norm_std_columns + [feature]
        
        loop_prep = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), dummyfy_columns),
            ('num', StandardScaler(), current_numeric_scope),
            ('pass', 'passthrough', base_nomod_columns)
        ], remainder='drop')

        loop_pipe = Pipeline([
            ("feature_engineering", FeatureEngineer()),
            ("preprocessing", loop_prep),
            ("model", clone(model))
        ])
        loop_pipe.fit(X_train, y_train)
        y_proba = loop_pipe.predict_proba(X_test)[:, 1]
        results.append({"feature_added": feature, "pr_auc": average_precision_score(y_test, y_proba)})

    return pd.DataFrame(results).sort_values(by="pr_auc", ascending=False).reset_index(drop=True)
