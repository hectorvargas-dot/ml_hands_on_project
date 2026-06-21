from aiohttp._websocket import models
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from src.utils import mlflow_utils as mlf_utils

# Configure structured module logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def find_project_root() -> Path:
    """Finds the project root directory by looking for pyproject.toml.

    Starting from the current working directory, this function traverses upward
    through parent directories until it locates a 'pyproject.toml' file.

    Returns:
        Path: The absolute path to the project root directory.

    Raises:
        FileNotFoundError: If 'pyproject.toml' cannot be found in any parent.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            logger.debug(f"Project root identified at: {parent}")
            return parent
    raise FileNotFoundError("Could not find project root (pyproject.toml)")


def setup_paths() -> Dict[str, Path]:
    """Generates and maps absolute paths required for the optimization workflow.

    Returns:
        Dict[str, Path]: A dictionary containing resolved project directory paths
            mapped to keys: 'project_root', 'db_path', 'artifacts_dir',
            'raw_features', 'target', and 'feature_registry'.
    """
    logger.info("Initializing system directory mappings.")
    project_root = find_project_root()
    return {
        "project_root": project_root,
        "db_path": project_root / "mlflow.db",
        "artifacts_dir": project_root / "mlartifacts",
        "raw_features": project_root / "data" / "processed" / "raw_features",
        "target": project_root / "data" / "processed" / "target",
        "feature_registry": project_root / "src" / "selected_features.json",
    }


def get_or_create_experiment(experiment_name: str) -> str:
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is not None:
        return experiment.experiment_id

    return client.create_experiment(experiment_name)


def initialize_mlflow(paths: Dict[str, Path], experiment_name: str) -> str:
    """Initializes the MLflow tracking environment and backend storage tables.

    Args:
        paths (Dict[str, Path]): Dictionary containing project filesystem paths.

    Returns:
        str: The registered MLflow experiment ID string.
    """
    tracking_uri = f"sqlite:///{paths['db_path']}"
    
    logger.info(f"Setting MLflow tracking URI backend to: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    experiment_id = get_or_create_experiment(experiment_name)

    mlflow.set_experiment(experiment_name)

    logger.info(
        f"MLflow active experiment set to '{experiment_name}' "
        f"(ID: {experiment_id})"
    )

    return experiment_id


def load_dataset(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads split training and test matrices from data directories.

    Args:
        paths (Dict[str, Path]): Dictionary containing project filesystem paths.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A 4-element tuple
            containing X_train, X_test, y_train, and y_test.
    """
    logger.info("Loading processing datasets from disk storage layers.")
    X_train = pd.read_csv(paths["raw_features"] / "X_train.csv")
    X_test = pd.read_csv(paths["raw_features"] / "X_test.csv")
    y_train = pd.read_csv(paths["target"] / "y_train.csv").squeeze()
    y_test = pd.read_csv(paths["target"] / "y_test.csv").squeeze()
    logger.info(f"Datasets ingested successfully. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def log_and_artifact_pr_curve(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Generates, saves, and logs a Precision-Recall Curve artifact to MLflow.

    Args:
        pipeline (Pipeline): The final optimized model pipeline.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series): True test labels.

    Returns:
        float: Calculated Test Precision-Recall Area Under the Curve (PR-AUC).
    """
    logger.info("Computing Precision-Recall curves across independent test sets.")
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    test_pr_auc = average_precision_score(y_test, probabilities)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"PR-AUC={test_pr_auc:.5f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Test Set)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save to a temporary file path and log directly as an MLflow artifact
    plot_path = "precision_recall_curve.png"
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Uploading generated plot file artifact to MLflow storage: {plot_path}")
    mlflow.log_artifact(plot_path)
    
    if os.path.exists(plot_path):
        os.remove(plot_path)

    return test_pr_auc

def log_shap_artifacts(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    sample_size: int = 500,
) -> None:
    """
    Generate SHAP visualizations and log them as MLflow artifacts.

    Args:
        pipeline: Trained pipeline.
        X_test: Raw test dataframe.
        sample_size: Number of rows used for SHAP.
    """
    logger.info("Generating SHAP explanations.")

    X_sample = X_test.sample(
        min(sample_size, len(X_test)),
        random_state=42,
    )

    # Transform through feature engineering
    X_engineered = (
        pipeline.named_steps["feature_engineering"]
        .transform(X_sample)
    )

    # Transform through preprocessing
    X_processed = (
        pipeline.named_steps["preprocessing"]
        .transform(X_engineered)
    )

    model = pipeline.named_steps["model"]

    # Recover feature names
    feature_names = (
        pipeline.named_steps["preprocessing"]
        .get_feature_names_out()
    )

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    feature_names = (
        pipeline.named_steps["preprocessing"]
        .get_feature_names_out()
    )

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
        index=X_sample.index,
    )

    # 1. Extract your compound model pipeline step
    model_step = pipeline.named_steps["model"]

    # 2. Extract the actual XGBClassifier instance nested inside it
    if isinstance(model_step, Pipeline) and "xgb" in model_step.named_steps:
        raw_xgb_estimator = model_step.named_steps["xgb"]
    else:
        raw_xgb_estimator = model_step

    # 3. Transform your background data through the preprocessing stages up to the model step
    # (Make sure to pass the features through the caster so SHAP uses the right dtypes!)
    X_background = pipeline.named_steps["feature_engineering"].transform(X_test)
    X_background = pipeline.named_steps["preprocessing"].transform(X_background)

    if isinstance(model_step, Pipeline) and "cast_to_numeric" in model_step.named_steps:
        X_background = model_step.named_steps["cast_to_numeric"].transform(X_background)

    # 4. Initialize TreeExplainer cleanly with the raw estimator
    explainer = shap.TreeExplainer(raw_xgb_estimator, data=X_background)

    shap_values = explainer(X_processed_df)

    artifact_dir = "shap_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)

    #
    # 1. Beeswarm plot
    #
    shap.plots.beeswarm(
        shap_values,
        max_display=20,
        show=False,
    )

    beeswarm_path = f"{artifact_dir}/shap_beeswarm.png"

    plt.tight_layout()
    plt.savefig(beeswarm_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(
        beeswarm_path,
        artifact_path="shap",
    )

    #
    # 2. Importance plot
    #
    shap.plots.bar(
        shap_values,
        max_display=20,
        show=False,
    )

    importance_path = (
        f"{artifact_dir}/shap_importance.png"
    )

    plt.tight_layout()
    plt.savefig(
        importance_path,
        bbox_inches="tight",
    )
    plt.close()

    mlflow.log_artifact(
        importance_path,
        artifact_path="shap",
    )

    #
    # 3. Waterfall plot
    #
    shap.plots.waterfall(
        shap_values[0],
        max_display=15,
        show=False,
    )

    waterfall_path = (
        f"{artifact_dir}/shap_waterfall.png"
    )

    plt.tight_layout()
    plt.savefig(
        waterfall_path,
        bbox_inches="tight",
    )
    plt.close()

    mlflow.log_artifact(
        waterfall_path,
        artifact_path="shap",
    )

    if os.path.exists(artifact_dir):
        os.remove(artifact_dir)

    logger.info("SHAP artifacts logged successfully.")
