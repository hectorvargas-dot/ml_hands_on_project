import mlflow
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer

from src import mlflow_utils

def init_mlflow_experiment(experiment_name: str, db_path: str, artifacts_dir: str) -> None:
    """Sets up connection parameters and creates experiment contexts inside MLflow (delegates to mlflow_utils)."""
    mlflow_utils.init_mlflow_experiment(experiment_name, db_path, artifacts_dir)


def log_classification_curves(y_true, y_proba) -> None:
    """Saves visualization summaries directly to the active MLflow run tracking window (delegates to mlflow_utils)."""
    mlflow_utils.log_classification_curves(y_true, y_proba)


def get_experiment_summary(experiment_name: str) -> pd.DataFrame | None:
    """Returns a formatted tracking summary sorted by PR-AUC performance metrics."""
    return mlflow_utils.get_experiment_summary(experiment_name)


def remove_recent_runs(experiment_name: str, count: int) -> None:
    """Deletes the latest sequence of tracking iterations from the active database log."""
    mlflow_utils.remove_recent_runs(experiment_name, count)

def dynamic_feature_engineer(X: pd.DataFrame, selected_features: list = None, FULL_REGISTRY: dict = None) -> pd.DataFrame:
    """Applies binary and continuous features from the registries to the DataFrame."""
    X = X.copy()
    features_to_build = selected_features if selected_features is not None else FULL_REGISTRY.keys()
    
    for feature_name in features_to_build:
        if feature_name in FULL_REGISTRY:
            X[feature_name] = FULL_REGISTRY[feature_name](X)
            
    return X


def run_sequential_selection(
    X, 
    y, 
    routing_config: dict, 
    base_model, 
    forward: bool = True, 
    k_features: int = 5, 
    FULL_REGISTRY: dict = None):
    """
    Combines engineering and preprocessing, extracts transformed feature names, 
    and applies mlxtend's SequentialFeatureSelector.
    """
    # 1. Deduce engineered features needed for this experiment layout
    all_layout_cols = (
        routing_config.get("passthrough", []) +
        routing_config.get("standard_scale", []) +
        routing_config.get("one_hot_encode", [])
    )
    needed_engineered = [col for col in all_layout_cols if col in FULL_REGISTRY]
    
    # 2. Assemble End-to-End Feature Generation & Transformation Pipelines
    fe_step = FunctionTransformer(
        dynamic_feature_engineer, 
        kw_args={"selected_features": needed_engineered, "FULL_REGISTRY": FULL_REGISTRY}
    )
    
    prep_step = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), routing_config.get("one_hot_encode", [])),
            ("num", StandardScaler(), routing_config.get("standard_scale", [])),
            ("pass", "passthrough", routing_config.get("passthrough", []))
        ],
        remainder="drop"
    )
    
    transform_pipe = Pipeline([("fe", fe_step), ("prep", prep_step)])
    
    # 3. Fit-Transform Data & Map Correct Post-Encoded String Headings
    X_transformed = transform_pipe.fit_transform(X, y)
    feature_names = transform_pipe.named_steps["prep"].get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # 4. Configure Sequential Selector Engine
    direction = "Forward" if forward else "Backward"
    sfs = SFS(
        clone(base_model),
        k_features=k_features,
        forward=forward,
        floating=False,
        scoring="average_precision", # Optimizing directly for PR-AUC
        cv=5,
        n_jobs=-1
    )
    
    print(f"\n--- Running {direction} Feature Selection on {base_model.__class__.__name__} ---")
    sfs.fit(X_transformed_df, y)
    
    print(f"Optimal Feature Subset Size: {len(sfs.k_feature_idx_)}")
    print(f"Optimal Feature Names: {sfs.k_feature_names_}")
    print(f"Best CV Score (PR-AUC): {sfs.k_score_:.4f}")
    
    return sfs, X_transformed_df

def run_and_log_sequential_selection(
    X,
    y,
    routing_config,
    base_model,
    experiment_label,
    forward=True,
    k_features=10,
    FULL_REGISTRY=None
):
    """
    Runs SFS and logs results to MLflow.
    """

    sfs, X_transformed_df = run_sequential_selection(
        X=X,
        y=y,
        routing_config=routing_config,
        base_model=base_model,
        forward=forward,
        k_features=k_features,
        FULL_REGISTRY=FULL_REGISTRY
    )

    method = "forward" if forward else "backward"

    with mlflow.start_run(
        run_name=f"{experiment_label}_{method}"
    ):

        mlflow.log_param(
            "model",
            base_model.__class__.__name__
        )

        mlflow.log_param(
            "selection_method",
            method
        )

        mlflow.log_param(
            "target_features",
            k_features
        )

        mlflow.log_metric(
            "pr_auc_cv",
            sfs.k_score_
        )

        mlflow.log_metric(
            "selected_feature_count",
            len(sfs.k_feature_names_)
        )

        selected_features = list(sfs.k_feature_names_)

        mlflow.log_text(
            "\n".join(selected_features),
            "selected_features.txt"
        )

        feature_df = pd.DataFrame(
            {"selected_feature": selected_features}
        )

        feature_df.to_csv(
            "selected_features.csv",
            index=False
        )

        mlflow.log_artifact(
            "selected_features.csv"
        )

    return sfs
