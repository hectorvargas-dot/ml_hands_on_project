import os
import mlflow
import shap
import pandas as pd
import matplotlib.pyplot as plt

def load_best_run_pipeline(experiment_name: str, db_path: str, order_by_metric: str = "metrics.pr_auc DESC"):
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
        experiment_ids=[experiment.experiment_id],
        order_by=[order_by_metric],
        max_results=1
    )
    
    if best_runs_df.empty:
        raise RuntimeError(f"No runs found inside the experiment '{experiment_name}'.")
        
    best_run = best_runs_df.iloc[0]
    
    # Load the serialized sklearn pipeline architecture
    model_uri = f"runs:/{best_run.run_id}/model"
    pipeline = mlflow.sklearn.load_model(model_uri)
    
    return pipeline, best_run


def transform_and_label_features(preprocessor, X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Passes raw features through the fitted preprocessor pipeline and returns 
    a properly labeled DataFrame with correct output feature column names.
    """
    X_transformed = preprocessor.transform(X_raw)
    feature_names = preprocessor.get_feature_names_out()
    
    return pd.DataFrame(X_transformed, columns=feature_names)


def calculate_global_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extracts MDI feature importance from a fitted tree model and computes 
    relative and cumulative importance matrices.
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    return importance_df


def compute_shap_values(model, X_transformed_df: pd.DataFrame):
    """Initializes a SHAP TreeExplainer and calculates local explanation matrices."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed_df)
    return shap_values


def plot_shap_summary(shap_values, X_df: pd.DataFrame, plot_type: str = None, class_idx: int = 1):
    """Generates standard SHAP summary plots (dot or bar distribution charts)."""
    plt.figure()
    # Handle both raw array slices and Explanation objects across SHAP versions safely
    target_shap_values = shap_values[:, :, class_idx] if len(shap_values.shape) == 3 else shap_values
    
    shap.summary_plot(target_shap_values, X_df, plot_type=plot_type, show=False)
    plt.tight_layout()
    plt.show()


def plot_shap_dependence(feature_name: str, shap_values, X_df: pd.DataFrame, class_idx: int = 1):
    """Generates a feature dependency plot mapping feature values to local SHAP attribution scores."""
    plt.figure()
    target_shap_values = shap_values[:, :, class_idx] if len(shap_values.shape) == 3 else shap_values
    
    shap.dependence_plot(feature_name, target_shap_values, X_df, show=False)
    plt.tight_layout()
    plt.show()