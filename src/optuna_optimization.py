from optuna.samplers import _lazy_random_state
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
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

def build_pipeline(
    trial, 
    FULL_REGISTRY: dict,
    current_layout:dict,
    random_state: int = 42
) -> Pipeline:
    """Constructs an end-to-end pipeline containing dynamic feature engineering 

    and modeling parameters tailored to an active Optuna trial space.
    """

    cat_cols = current_layout['one_hot_encode']
    num_cols = current_layout['standard_scale']
    pass_cols = current_layout['passthrough']

    # 1. ADDED: Deduce which engineered features are expected by this configuration layout
    all_layout_cols = cat_cols + num_cols + pass_cols
    needed_engineered = [col for col in all_layout_cols if col in FULL_REGISTRY]
    
    fe_step = FunctionTransformer(
        fe.dynamic_feature_engineer,
        kw_args={
            "selected_features": needed_engineered,
            "FULL_REGISTRY": FULL_REGISTRY,
        },
    )

    # 2. Preprocessing Space Selection
    scaler_name = trial.suggest_categorical("scaler", ["std", "minmax", "robust"])
    encoder_name = trial.suggest_categorical("encoder", ["drop_first", "no_drop"])

    scalers = {"std": StandardScaler(), "minmax": MinMaxScaler(), "robust": RobustScaler()}

    encoder = OneHotEncoder(
        handle_unknown="ignore", drop="first" if encoder_name == "drop_first" else None
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", encoder, cat_cols),
            ("num", scalers[scaler_name], num_cols),
            ("pass", "pass_through" if isinstance(pass_cols, str) else "passthrough", pass_cols),
        ],
        remainder="drop",
    )

    # 3. Model Search Space Selection
    model_name = trial.suggest_categorical("model", ["rf"])#, "xgb"])

    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int(
                "rf_n_estimators",
                280,
                320
            ),

            max_depth=trial.suggest_int(
                "rf_max_depth",
                8,
                10
            ),

            min_samples_split=trial.suggest_int(
                "rf_min_samples_split",
                18,
                28
            ),

            min_samples_leaf=trial.suggest_int(
                "rf_min_samples_leaf",
                1,
                3
            ),

            random_state=random_state,
            n_jobs=-1,
        )
    else:
        model = XGBClassifier(
            n_estimators=trial.suggest_int(
                "xgb_n_estimators",
                800,
                1400
            ),

            # Fixed after convergence
            max_depth=5,

            learning_rate=trial.suggest_float(
                "xgb_learning_rate",
                0.025,
                0.045,
                log=True
            ),

            subsample=trial.suggest_float(
                "xgb_subsample",
                0.94,
                0.98
            ),

            colsample_bytree=trial.suggest_float(
                "xgb_colsample_bytree",
                0.45,
                0.60
            ),

            # Fixed after convergence
            min_child_weight=2,

            gamma=trial.suggest_float(
                "xgb_gamma",
                1.5,
                3.5
            ),

            reg_alpha=trial.suggest_float(
                "xgb_reg_alpha",
                1e-8,
                1e-3,
                log=True
            ),

            reg_lambda=trial.suggest_float(
                "xgb_reg_lambda",
                1e-9,
                1e-4,
                log=True
            ),

            scale_pos_weight=trial.suggest_float(
                "xgb_scale_pos_weight",
                1.8,
                2.2
            ),

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
    """Objective factory evaluating cross-validation runs safely over isolated scopes."""

    def __init__(
        self,
        X,
        y,
        FULL_REGISTRY,
        current_layout,
        n_splits,
        random_state,
    ):
        self.X = X
        self.y = y
        self.FULL_REGISTRY = FULL_REGISTRY
        self.current_layout = current_layout
        self.n_splits = n_splits
        self.random_state = random_state

    def __call__(self, trial):
        pipeline = build_pipeline(
            trial,
            self.FULL_REGISTRY,
            self.current_layout,
            self.random_state,
        )

        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(
            pipeline, self.X, self.y, scoring="average_precision", cv=cv, n_jobs=-1
        )

        mean_auc = np.mean(scores)
        trial.report(mean_auc, step=0)

        import optuna

        if trial.should_prune():
            raise optuna.TrialPruned()

        return mean_auc


def evaluate_and_log_best_model(
    best_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cat_cols: list,
    num_cols: list,
    pass_cols: list,
):
    """Fits the best model, logs train/test performance indicators, and packages runtime artifacts."""

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
            "train_auc": roc_auc_score(y_train, y_train_proba),
            "test_accuracy": accuracy_score(y_test, y_test_pred),
            "test_precision": precision_score(y_test, y_test_pred, zero_division=0),
            "test_recall": recall_score(y_test, y_test_pred, zero_division=0),
            "test_f1": f1_score(y_test, y_test_pred, zero_division=0),
            "test_auc": roc_auc_score(y_test, y_test_proba),
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
