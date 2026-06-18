from mlflow.tracing.processor import base_mlflow
import mlflow
import optuna
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
    FunctionTransformer,
)
from xgboost import XGBClassifier

from src import feature_engineering as fe

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def build_pipeline(trial, current_layout: dict, random_state: int = 42) -> Pipeline:
    """Constructs an end-to-end pipeline using the class-based feature transformer."""
    
    cat_cols = current_layout['one_hot_encode']
    num_cols = current_layout['standard_scale']
    pass_cols = current_layout['passthrough']

    # Deduce which engineered features are needed by checking what columns exist in our configuration layout
    all_layout_cols = cat_cols + num_cols + pass_cols
    
    # Instantiate the class transformer directly
    # It dynamically infers which rules to apply based on your layout strings!
    dummy_instance = fe.DynamicFeatureEngineer()
    available_transformations = (
        dummy_instance._get_all_binary_features() + 
        dummy_instance._get_all_continuous_features()
    )
    needed_engineered = [col for col in all_layout_cols if col in available_transformations]
    
    fe_step = fe.DynamicFeatureEngineer(selected_features=needed_engineered)

    # Preprocessing Space Selection
    scaler_name = trial.suggest_categorical("scaler", ["std", "minmax", "robust"])
    encoder_name = trial.suggest_categorical("encoder", ["drop_first", "no_drop"])
    scalers = {"std": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}
    encoder = OneHotEncoder(handle_unknown="ignore", drop="first" if encoder_name == "drop_first" else None)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", scalers[scaler_name], num_cols),
            ("pass", "passthrough", pass_cols),
        ],
        remainder="drop",
    )

    # Model Search Space Selection (Remains identical)
    model_name = trial.suggest_categorical("model", ["rf", "xgb"])
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 280, 295),
            max_depth=trial.suggest_int("rf_max_depth", 9, 13),
            min_samples_split=trial.suggest_int("rf_min_samples_split", 18, 28),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 3),
            random_state=random_state, 
            n_jobs=-1,
        )
    else:
        model = XGBClassifier(
            n_estimators=trial.suggest_int("xgb_n_estimators", 800, 1400),
            max_depth=5,
            learning_rate=trial.suggest_float("xgb_learning_rate", 0.025, 0.045, log=True),
            subsample=trial.suggest_float("xgb_subsample", 0.94, 0.98),
            colsample_bytree=trial.suggest_float("xgb_colsample_bytree", 0.45, 0.60),
            min_child_weight=2,
            gamma=trial.suggest_float("xgb_gamma", 1.5, 3.5),
            reg_alpha=trial.suggest_float("xgb_reg_alpha", 1e-8, 1e-3, log=True),
            reg_lambda=trial.suggest_float("xgb_reg_lambda", 1e-9, 1e-4, log=True),
            scale_pos_weight=trial.suggest_float("xgb_scale_pos_weight", 1.6, 2.1),
            random_state=random_state, 
            n_jobs=-1, 
            eval_metric="aucpr",
        )

    # Return the fully unified sequence including Feature Engineering
    return Pipeline([
        ("feature_engineering", fe_step),
        ("preprocessing", preprocessor), 
        ("model", model)
    ])


class ObjectiveCV:
    """Objective factory evaluating cross-validation runs safely over isolated scopes 
    and automatically logging full test/train breakdowns to active child trials.
    """
    
    # STEP 1A: Accept the baseline holdout matrices into memory
    def __init__(self, X, y, current_layout, n_splits, random_state):
        self.X = X
        self.y = y
        self.current_layout = current_layout
        self.n_splits = n_splits
        self.random_state = random_state

    def __call__(self, trial):
        # Build the specific dynamic pipeline structure for this active trial
        pipeline = build_pipeline(
            trial=trial, 
            current_layout=self.current_layout, 
            random_state=self.random_state
        )
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # Calculate cross-validation performance score
        scores = cross_val_score(
            pipeline, self.X, self.y, scoring="average_precision", cv=cv, n_jobs=-1
        )
        mean_auc = np.mean(scores)
        
        # Report progress back to Optuna for potential pruning
        trial.report(mean_auc, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return mean_auc


def evaluate_and_log_best_model(
    best_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    current_layout: dict,
):
    """Fits the best model, logs train/test performance indicators, and packages runtime artifacts."""

    cat_cols = current_layout['one_hot_encode']
    num_cols = current_layout['standard_scale']
    pass_cols = current_layout['passthrough']

    best_pipeline.fit(X_train, y_train)
    preprocessor = best_pipeline.named_steps["preprocessing"]

    # Schema Logging
    input_features = cat_cols + num_cols + pass_cols
    output_features = preprocessor.get_feature_names_out().tolist()

    mlflow.log_param("n_input_features", len(input_features))
    mlflow.log_param("n_output_features", len(output_features))
    mlflow.log_dict(
        {"input_features": input_features, "output_features": output_features},
        "feature_schema.json",
    )

    # Compute Fixed Partition Targets Evaluations
    y_train_pred = best_pipeline.predict(X_train)
    y_train_proba = best_pipeline.predict_proba(X_train)[:, 1]

    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]

    mlflow.log_metrics(
        {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
            "train_roc_auc": roc_auc_score(y_train, y_train_proba),
            "train_pr_auc": average_precision_score(y_train, y_train_proba), # Added
            
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, y_test_proba),
            "test_pr_auc": average_precision_score(y_test, y_test_proba), # Added
        }
    )

    # Structural Matrix Architecture Artifacts Serialization
    mlflow.log_dict(
        classification_report(y_test, y_test_pred, output_dict=True, zero_division=0),
        "classification_report.json",
    )
    mlflow.log_dict(confusion_matrix(y_test, y_test_pred).tolist(), "confusion_matrix.json")

    mlflow.sklearn.log_model(
        sk_model=best_pipeline, artifact_path="best_model", serialization_format="cloudpickle"
    )

def suggest_numeric_ranges(
    study,
    top_quantile=0.95,
    boundary_threshold=0.15,
    expansion_factor=0.5,
):
    """
    Suggest new Optuna search ranges for numeric parameters.

    Parameters
    ----------
    study : optuna.study.Study

    top_quantile : float
        Use top X% of trials to infer the optimum region.

    boundary_threshold : float
        Fraction of range considered "close to boundary".

    expansion_factor : float
        How much to expand the range when boundary pressure exists.

    Returns
    -------
    pd.DataFrame
    """

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

        relative_position = (
            median_top - current_min
        ) / span

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
        .sort_values(
            ["action", "parameter"],
            ascending=[True, True]
        )
        .reset_index(drop=True)
    )
