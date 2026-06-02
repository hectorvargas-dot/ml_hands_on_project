import os
import numpy as np
import pandas as pd
import mlflow

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

def build_pipeline(trial, cat_cols: list, num_cols: list, pass_cols: list, random_state: int = 42) -> Pipeline:
    """Constructs a dynamic pipeline instance parameterized by an active Optuna trial space."""
    
    # Preprocessing Space Selection
    scaler_name = trial.suggest_categorical('scaler', ['std', 'minmax', 'robust'])
    encoder_name = trial.suggest_categorical('encoder', ['drop_first', 'no_drop'])

    scalers = {
        'std': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }

    encoder = OneHotEncoder(
        handle_unknown='ignore',
        drop='first' if encoder_name == 'drop_first' else None
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', encoder, cat_cols),
            ('num', scalers[scaler_name], num_cols),
            ('pass', 'passthrough', pass_cols)
        ],
        remainder='drop'
    )

    # Model Search Space Selection
    model_name = trial.suggest_categorical('model', ['rf', 'xgb'])

    if model_name == 'rf':
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int('rf_n_estimators', 100, 500),
            max_depth=trial.suggest_int('rf_max_depth', 2, 20),
            min_samples_split=trial.suggest_int('rf_min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('rf_min_samples_leaf', 1, 10),
            random_state=random_state,
            n_jobs=-1
        )
    else:
        model = XGBClassifier(
            n_estimators=trial.suggest_int('xgb_n_estimators', 500, 1000),
            max_depth=trial.suggest_int('xgb_max_depth', 4, 6),
            learning_rate=trial.suggest_float('xgb_learning_rate', 0.03, 0.08, log=True),
            subsample=trial.suggest_float('xgb_subsample', 0.90, 1.0),
            colsample_bytree=trial.suggest_float('xgb_colsample_bytree', 0.50, 0.70),
            min_child_weight=trial.suggest_int('xgb_min_child_weight', 2, 5),
            gamma=trial.suggest_float('xgb_gamma', 2.5, 5.5),
            reg_alpha=trial.suggest_float('xgb_reg_alpha', 1e-5, 1e-2, log=True),
            reg_lambda=trial.suggest_float('xgb_reg_lambda', 1e-6, 1e-3, log=True),
            scale_pos_weight=trial.suggest_float('xgb_scale_pos_weight', 1.5, 2.5),
            random_state=random_state,
            n_jobs=-1,
            eval_metric='aucpr'
        )

    return Pipeline([
        ('preprocessing', preprocessor),
        ('model', model)
    ])


class ObjectiveCV:
    """Objective factory evaluating cross-validation runs safely over isolated scopes."""
    def __init__(self, X: pd.DataFrame, y: pd.Series, cat_cols: list, num_cols: list, 
                 pass_cols: list, n_splits: int, random_state: int):
        self.X = X
        self.y = y
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.pass_cols = pass_cols
        self.n_splits = n_splits
        self.random_state = random_state

    def __call__(self, trial):
        pipeline = build_pipeline(trial, self.cat_cols, self.num_cols, self.pass_cols, self.random_state)
        
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        scores = cross_val_score(
            pipeline, self.X, self.y,
            scoring='average_precision', cv=cv, n_jobs=-1
        )
        
        mean_auc = np.mean(scores)
        trial.report(mean_auc, step=0)
        
        import optuna
        if trial.should_prune():
            raise optuna.TrialPruned()
            
        return mean_auc


def evaluate_and_log_best_model(best_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_test: pd.DataFrame, y_test: pd.Series, cat_cols: list, num_cols: list, pass_cols: list):
    """Fits the best model, logs train/test performance indicators, and packages runtime artifacts."""
    
    best_pipeline.fit(X_train, y_train)
    preprocessor = best_pipeline.named_steps['preprocessing']
    
    # Schema Logging
    input_features = cat_cols + num_cols + pass_cols
    output_features = preprocessor.get_feature_names_out().tolist()
    
    mlflow.log_param("n_input_features", len(input_features))
    mlflow.log_param("n_output_features", len(output_features))
    mlflow.log_dict({"input_features": input_features, "output_features": output_features}, "feature_schema.json")
    
    # Compute Fixed Partition Targets Evaluations
    y_train_pred = best_pipeline.predict(X_train)
    y_train_proba = best_pipeline.predict_proba(X_train)[:, 1]
    
    y_test_pred = best_pipeline.predict(X_test)
    y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
    
    mlflow.log_metrics({
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
    })
    
    # Structural Matrix Architecture Artifacts Serialization
    mlflow.log_dict(classification_report(y_test, y_test_pred, output_dict=True, zero_division=0), "classification_report.json")
    mlflow.log_dict(confusion_matrix(y_test, y_test_pred).tolist(), "confusion_matrix.json")
    
    mlflow.sklearn.log_model(sk_model=best_pipeline, name="best_model", serialization_format="cloudpickle")