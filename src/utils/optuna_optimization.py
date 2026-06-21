import logging

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)
from xgboost import XGBClassifier

from src.utils import feature_engineering_utils as fe

logger = logging.getLogger(__name__)


def build_pipeline(
    trial,
    current_layout: dict,
    random_state: int = 42,
    override_ranges: dict = None,
) -> Pipeline:
    """Build a complete machine learning pipeline with XGBoost."""
    if override_ranges is None:
        override_ranges = {}

    logger.debug("Building pipeline for Optuna trial %s", trial.number)

    cat_cols = current_layout["one_hot_encode"]
    num_cols = current_layout["standard_scale"]
    pass_cols = current_layout["passthrough"]
    all_layout_cols = cat_cols + num_cols + pass_cols

    dummy_instance = fe.DynamicFeatureEngineer()
    available_transformations = (
        dummy_instance._get_all_binary_features()
        + dummy_instance._get_all_continuous_features()
    )
    needed_engineered = [
        col for col in all_layout_cols if col in available_transformations
    ]

    logger.info("Selected engineered features: %s", needed_engineered)

    feature_engineering = fe.DynamicFeatureEngineer(
        selected_features=needed_engineered
    )

    scaler_name = trial.suggest_categorical(
        "scaler", ["std", "minmax", "robust"]
    )
    encoder_name = trial.suggest_categorical(
        "encoder", ["drop_first", "no_drop"]
    )

    scalers = {
        "std": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    encoder = OneHotEncoder(
        handle_unknown="ignore",
        drop="first" if encoder_name == "drop_first" else None,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", scalers[scaler_name], num_cols),
            ("pass", "passthrough", pass_cols),
        ],
        remainder="drop",
    )

    # Dynamic boundary helper for XGBoost parameters
    def get_xgb_bounds(param, default_min, default_max):
        if param in override_ranges:
            return override_ranges[param]["min"], override_ranges[param]["max"]
        return default_min, default_max

    xgb_n_min, xgb_n_max = get_xgb_bounds("xgb_n_estimators", 900, 1700)
    xgb_lr_min, xgb_lr_max = get_xgb_bounds("xgb_learning_rate", 0.018, 0.042)
    xgb_sub_min, xgb_sub_max = get_xgb_bounds("xgb_subsample", 0.94, 0.99)
    xgb_col_min, xgb_col_max = get_xgb_bounds("xgb_colsample_bytree", 0.45, 0.60)
    xgb_gam_min, xgb_gam_max = get_xgb_bounds("xgb_gamma", 3.1, 3.5)
    xgb_alp_min, xgb_alp_max = get_xgb_bounds("xgb_reg_alpha", 1e-9, 1e-6)
    xgb_lam_min, xgb_lam_max = get_xgb_bounds("xgb_reg_lambda", 1e-9, 1e-4)
    xgb_spw_min, xgb_spw_max = get_xgb_bounds("xgb_scale_pos_weight", 1.1, 1.7)

    # Ensure integer bounds stay valid integers
    xgb_n_min, xgb_n_max = max(1, int(xgb_n_min)), max(1, int(xgb_n_max))

    model = XGBClassifier(
        n_estimators=trial.suggest_int("xgb_n_estimators", xgb_n_min, xgb_n_max),
        max_depth=5,
        learning_rate=trial.suggest_float(
            "xgb_learning_rate", max(1e-5, xgb_lr_min), xgb_lr_max, log=True
        ),
        subsample=trial.suggest_float(
            "xgb_subsample", max(0.1, xgb_sub_min), min(1.0, xgb_sub_max)
        ),
        colsample_bytree=trial.suggest_float(
            "xgb_colsample_bytree",
            max(0.1, xgb_col_min),
            min(1.0, xgb_col_max),
        ),
        min_child_weight=2,
        gamma=trial.suggest_float("xgb_gamma", max(0.0, xgb_gam_min), xgb_gam_max),
        reg_alpha=trial.suggest_float(
            "xgb_reg_alpha", max(1e-10, xgb_alp_min), xgb_alp_max, log=True
        ),
        reg_lambda=trial.suggest_float(
            "xgb_reg_lambda", max(1e-10, xgb_lam_min), xgb_lam_max, log=True
        ),
        scale_pos_weight=trial.suggest_float(
            "xgb_scale_pos_weight", max(0.1, xgb_spw_min), xgb_spw_max
        ),
        random_state=random_state,
        n_jobs=-1,
        eval_metric="aucpr",
    )

    return Pipeline(
        [
            ("feature_engineering", feature_engineering),
            ("preprocessing", preprocessor),
            ("model", model),
        ]
    )


class ObjectiveCV:
    """Callable Optuna objective performing cross-validation with XGBoost."""

    def __init__(
        self, X, y, current_layout, n_splits, random_state, override_ranges=None
    ):
        self.X = X
        self.y = y
        self.current_layout = current_layout
        self.n_splits = n_splits
        self.random_state = random_state
        self.override_ranges = override_ranges

    def __call__(self, trial):
        logger.info("Starting Optuna trial %s", trial.number)

        pipeline = build_pipeline(
            trial,
            self.current_layout,
            self.random_state,
            override_ranges=self.override_ranges,
        )

        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        scores = cross_val_score(
            pipeline,
            self.X,
            self.y,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
        )

        mean_score = np.mean(scores)
        logger.info("Trial %s PR-AUC %.5f", trial.number, mean_score)
        trial.report(mean_score, step=0)

        if trial.should_prune():
            logger.warning("Trial %s pruned", trial.number)
            raise optuna.TrialPruned()

        return mean_score


def evaluate_and_log_best_model(
    best_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    current_layout: dict,
):
    """Train final pipeline, evaluate metrics, and log artifacts."""
    logger.info("Training final optimized pipeline")
    best_pipeline.fit(X_train, y_train)

    logger.info("Generating predictions")
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    test_pr_auc = average_precision_score(y_test, y_test_proba)
    logger.info("Final test PR-AUC %.5f", test_pr_auc)

    mlflow.log_metric("test_pr_auc", test_pr_auc)
    mlflow.sklearn.log_model(
        best_pipeline,
        artifact_path="best_model",
        serialization_format="cloudpickle",
    )
    logger.info("Model artifact logged successfully")


def suggest_numeric_ranges(
    study,
    top_quantile=0.95,
    boundary_threshold=0.15,
    expansion_factor=0.5,
):
    """Suggest new Optuna search ranges for numeric parameters."""
    df = study.trials_dataframe()
    completed = df[df["state"] == "COMPLETE"].copy()
    threshold = completed["value"].quantile(top_quantile)
    top_trials = completed[completed["value"] >= threshold]

    results = []
    for col in completed.columns:
        if not col.startswith("params_"):
            continue
        if not pd.api.types.is_numeric_dtype(completed[col]):
            continue

        all_values = completed[col].dropna()
        if len(all_values) < 5:
            continue

        top_values = top_trials[col].dropna()
        current_min = all_values.min()
        current_max = all_values.max()
        span = current_max - current_min
        if span == 0:
            continue

        median_top = top_values.median()
        relative_position = (median_top - current_min) / span

        if relative_position < boundary_threshold:
            action = "move_left"
            suggested_min = current_min - expansion_factor * span
            suggested_max = current_max
        elif relative_position > (1 - boundary_threshold):
            action = "move_right"
            suggested_min = current_min
            suggested_max = current_max + expansion_factor * span
        else:
            q10 = top_values.quantile(0.10)
            q90 = top_values.quantile(0.90)
            concentration = (q90 - q10) / span

            if concentration < 0.40:
                action = "narrow"
                padding = (q90 - q10) * 0.20
                suggested_min = q10 - padding
                suggested_max = q90 + padding
            else:
                action = "keep"
                suggested_min = current_min
                suggested_max = current_max

        results.append(
            {
                "parameter": col.replace("params_", ""),
                "current_min": current_min,
                "current_max": current_max,
                "best_median": median_top,
                "action": action,
                "suggested_min": suggested_min,
                "suggested_max": suggested_max,
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values(["action", "parameter"], ascending=[True, True])
        .reset_index(drop=True)
    )
