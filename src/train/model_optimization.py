import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import MedianPruner
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline

from src.utils import feature_engineering_utils as fe
from src.utils import mlflow_utils as mlf_utils
from src.utils import optuna_optimization as utils

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

    experiment_id = mlf_utils.init_mlflow_experiment(
        experiment_name,
        str(paths["db_path"]),
        str(paths["artifacts_dir"]),
    )
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow active experiment set to '{experiment_name}' (ID: {experiment_id})")
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


def build_feature_layout(paths: Dict[str, Path]) -> Dict[str, list]:
    """Builds the categorical, numeric, and passthrough feature schema layout.

    Args:
        paths (Dict[str, Path]): Dictionary containing project filesystem paths.

    Returns:
        Dict[str, list]: Explicit structural feature blueprints required by the
            ColumnTransformer pipeline step.
    """
    logger.info("Extracting feature engineering registry layouts.")
    with open(paths["feature_registry"], "r") as f:
        config = json.load(f)

    selected_features = config["selected_features"]
    dummy_engineer = fe.DynamicFeatureEngineer()

    binary_features = dummy_engineer._get_all_binary_features()
    continuous_features = dummy_engineer._get_all_continuous_features()

    filtered_binary = [f for f in selected_features if f in binary_features]
    filtered_continuous = [f for f in selected_features if f in continuous_features]

    return {
        "passthrough": config["nomod_columns"] + filtered_binary,
        "standard_scale": config["norm_std_columns"] + filtered_continuous,
        "one_hot_encode": config["dummyfy_columns"],
    }


def create_optuna_study(experiment_id: str) -> Tuple[optuna.study.Study, MLflowCallback]:
    """Constructs the core Optuna Study optimization state and MLflow callbacks.

    Args:
        experiment_id (str): Destination MLflow experiment identifier.

    Returns:
        Tuple[optuna.study.Study, MLflowCallback]: The target study coordinator
            and associated real-time MLflow logging handler callback.
    """
    logger.info("Initializing Optuna study context with automated MLflow tracking.")
    callback = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name="pr_auc",
        create_experiment=False,
        mlflow_kwargs={
            "nested": True,
            "experiment_id": experiment_id,
        },
    )

    study = optuna.create_study(
        study_name="customer-churn-xgb-search-train-pipeline-v1",
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    return study, callback


def optimize_model_multi_stage(
    study: optuna.study.Study,
    callback: MLflowCallback,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    current_layout: Dict[str, list],
) -> Tuple[optuna.study.Study, dict]:
    """Runs a 4-stage sequential trial matrix looping dynamic hyperparameter spaces.

    Runs 500 initial base estimator runs, evaluates parametric performance distribution,
    and updates parameter boundaries across 3 successive 200-trial stages.

    Args:
        study (optuna.study.Study): Target hyperparameter study manager.
        callback (MLflowCallback): Real-time parameter logging runner callback.
        X_train (pd.DataFrame): Training feature space matrix.
        y_train (pd.Series): Target objective labels vector.
        current_layout (Dict[str, list]): Structural transformation columns setup mapping.

    Returns:
        Tuple[optuna.study.Study, dict]: Optimized optimization study profile
            and the final modified search boundary parameter constraints dictionary.
    """
    stages = [
        (500, "Base Phase"),
        (200, "Phase 2 Adaptive Step"),
        (200, "Phase 3 Adaptive Step"),
        (200, "Phase 4 Adaptive Step"),
    ]

    current_override_ranges = {}

    for i, (n_trials, stage_name) in enumerate(stages):
        logger.info(f"=== Beginning Hyperparameter Optimizing Cycle: {stage_name} ({n_trials} Trials) ===")

        objective = utils.ObjectiveCV(
            X=X_train,
            y=y_train,
            current_layout=current_layout,
            n_splits=5,
            random_state=42,
            override_ranges=current_override_ranges,
        )

        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

        if i < len(stages) - 1:
            logger.info(f"Analyzing stage results for '{stage_name}' to rewrite parameter space.")
            ranges_df = utils.suggest_numeric_ranges(study)

            current_override_ranges = {}
            for _, row in ranges_df.iterrows():
                current_override_ranges[row["parameter"]] = {
                    "min": row["suggested_min"],
                    "max": row["suggested_max"],
                }

            logger.info(f"Hyperparameter space limits modified after completion of {stage_name}.")
            print(ranges_df[["parameter", "action", "suggested_min", "suggested_max"]])

    return study, current_override_ranges


def train_best_model(
    study: optuna.study.Study, current_layout: Dict[str, list], X_train: pd.DataFrame, override_ranges: dict
) -> Pipeline:
    """Rebuilds and trains the absolute best pipeline profile from parameter logs.

    Args:
        study (optuna.study.Study): Optimization trial registry source.
        current_layout (Dict[str, list]): Pipelines data engineering column guide.
        X_train (pd.DataFrame): Ingested complete baseline train frame.
        override_ranges (dict): Search space parameter overrides.

    Returns:
        Pipeline: A fully trained, unified scikit-learn matching estimator pipeline.
    """
    logger.info(f"Re-assembling global best pipeline architecture (Trial #{study.best_trial.number}).")
    pipeline = utils.build_pipeline(
        trial=study.best_trial,
        current_layout=current_layout,
        random_state=42,
        override_ranges=override_ranges,
    )

    pipeline.named_steps["feature_engineering"].fit(X_train)
    X_engineered = pipeline.named_steps["feature_engineering"].transform(X_train)
    pipeline.named_steps["preprocessing"].fit(X_engineered)

    return pipeline


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


def verify_and_promote_model(
    experiment_id: str, new_model_uri: str, candidate_score: float
) -> None:
    """Compares the current score against historical runs and promotes the model if it wins.

    Args:
        experiment_id (str): The current MLflow experiment ID.
        new_model_uri (str): The artifact URI of the model logged in the current run.
        candidate_score (float): The PR-AUC achieved by the current candidate model.
    """
    logger.info("Querying MLflow tracking server to evaluate historical run metrics for promotion.")
    
    # Query all past non-nested runs inside this experiment
    all_runs = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.parentRunId IS NULL",
        order_by=["metrics.test_pr_auc DESC"],
    )

    is_champion = True
    historical_best = 0.0

    if not all_runs.empty and "metrics.test_pr_auc" in all_runs.columns:
        # Drop the current run from the historical comparison array if it exists there
        past_scores = all_runs[all_runs["run_id"] != mlflow.active_run().info.run_id]["metrics.test_pr_auc"].dropna()
        
        if not past_scores.empty:
            historical_best = past_scores.max()
            logger.info(f"Current top historical benchmark score: PR-AUC = {historical_best:.5f}")
            if candidate_score <= historical_best:
                is_champion = False

    if is_champion:
        logger.info(f" Promotion condition met! New Best Model: {candidate_score:.5f} > Previous Best: {historical_best:.5f}")
        # Apply metadata tracking indicators to denote its champion production state
        mlflow.set_tag("model_status", "Production_Champion")
        mlflow.log_param("promoted_to_production", "True")
        
        # Note: If utilizing the MLflow Model Registry, execute the call below:
        # mlflow.register_model(model_uri=new_model_uri, name="CustomerChurnXGBClassifier")
    else:
        logger.info(f" Candidate model ({candidate_score:.5f}) failed to outperform the historical champion ({historical_best:.5f}). Model skipped promotion.")
        mlflow.set_tag("model_status", "Candidate_Rejected")


def run_model_optimization(experiment_name: str) -> None:
    """Core runtime engine orchestration setup execution flow."""
    paths = setup_paths()
    experiment_id = initialize_mlflow(paths, experiment_name)
    X_train, X_test, y_train, y_test = load_dataset(paths)
    current_layout = build_feature_layout(paths)

    study, callback = create_optuna_study(experiment_id)

    with mlflow.start_run(
        run_name="optuna_search_parent",
        experiment_id=experiment_id,
    ) as parent_run:
        
        # Execute hyperparameter sweeps
        study, final_override_ranges = optimize_model_multi_stage(
            study, callback, X_train, y_train, current_layout
        )

        mlflow.log_metric("best_optuna_val_pr_auc", study.best_value)
        mlflow.log_params(study.best_params)

        # Retrain champion candidate setup
        best_pipeline = train_best_model(
            study, current_layout, X_train, final_override_ranges
        )

        # Standard metrics tracking and artifact serialization
        utils.evaluate_and_log_best_model(
            best_pipeline=best_pipeline,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            current_layout=current_layout,
        )

        # Generate, save, and log PR-AUC Curve plot artifact to current run
        test_pr_auc = log_and_artifact_pr_curve(best_pipeline, X_test, y_test)

        # Evaluate model performance against existing benchmarks and decide on production deployment
        current_model_uri = f"runs:/{parent_run.info.run_id}/best_model"
        verify_and_promote_model(
            experiment_id=experiment_id,
            new_model_uri=current_model_uri,
            candidate_score=test_pr_auc
        )


if __name__ == "__main__":
    run_model_optimization()