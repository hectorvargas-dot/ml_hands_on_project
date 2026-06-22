import logging
from typing import Dict, List, Tuple

import mlflow
import pandas as pd
from src.drift_monitoring.monitoring_utils import detect_feature_drift

logger = logging.getLogger(__name__)


def extract_and_classify_features(
    pipeline: mlflow.pyfunc.PyFuncModel, 
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Transforms raw data and classifies the output features by statistical type.

    Processes the dataframe through the pipeline's feature engineering and
    preprocessing stages, then classifies columns as numerical or categorical
    based on scikit-learn names and cardinality constraints.

    Args:
        pipeline: Loaded MLflow or scikit-learn pipeline instance.
        df (pd.DataFrame): Raw incoming input dataframe.

    Returns:
        Tuple[pd.DataFrame, List[str], List[str]]: A 3-element tuple containing:
            - transformed_df (pd.DataFrame): Transformed dataframe with correct dtypes.
            - final_num_cols (List[str]): Verified continuous numerical features.
            - final_cat_cols (List[str]): Verified categorical/binary features.
    """
    fe_step = pipeline.with_config({}) if hasattr(pipeline, "with_config") else pipeline
    
    # 1. Apply Feature Engineering and Column Transformation
    X_engineered = fe_step.named_steps["feature_engineering"].transform(df)
    prep_step = fe_step.named_steps["preprocessing"]
    X_transformed = prep_step.transform(X_engineered)
    
    # 2. Extract feature names out and build a typed DataFrame
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
        
    feature_names = prep_step.get_feature_names_out()
    transformed_df = pd.DataFrame(
        X_transformed, 
        columns=feature_names, 
        index=df.index
    ).astype("float32")

    # 3. Classify features using scikit-learn prefixes
    raw_cat_cols = [c for c in feature_names if c.startswith("cat__")]
    raw_num_cols = [c for c in feature_names if c.startswith("num__")]
    raw_pass_cols = [c for c in feature_names if c.startswith("pass__")]

    final_num_cols = list(raw_num_cols)
    final_cat_cols = list(raw_cat_cols)

    # 4. Smart Routing for Passthrough Columns
    for col in raw_pass_cols:
        unique_vals = transformed_df[col].dropna().unique()
        
        # If the column is binary or behaves like an indicator flag, route to categorical
        if len(unique_vals) <= 2 or set(unique_vals).issubset({0.0, 1.0}):
            final_cat_cols.append(col)
            logger.debug(f"Routing binary passthrough feature '{col}' to Categorical analysis.")
        else:
            final_num_cols.append(col)
            logger.debug(f"Routing continuous passthrough feature '{col}' to Numerical analysis.")

    return transformed_df, final_num_cols, final_cat_cols


def feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    run_id: str,
) -> Tuple[pd.DataFrame, bool]:
    """Calculates data drift metrics between a baseline and production dataset.

    Args:
        reference_df (pd.DataFrame): Historical baseline dataset (e.g., training data).
        current_df (pd.DataFrame): Current production payload dataset.
        run_id (str): Target MLflow Run ID containing the reference model pipeline.

    Returns:
        Tuple[pd.DataFrame, bool]: A summary tracking report DataFrame and a global
            boolean flag indicating if data drift has been triggered.
    """
    logger.info(f"Loading tracking production model pipeline from Run ID: {run_id}")
    model_uri = f"runs:/{run_id}/best_model"
    pipeline = mlflow.sklearn.load_model(model_uri)

    # Transform and classify reference baseline dataset
    ref_transformed, num_cols, cat_cols = extract_and_classify_features(
        pipeline, reference_df
    )

    # Transform current production payload dataset
    cur_transformed, _, _ = extract_and_classify_features(
        pipeline, current_df
    )

    logger.info(
        f"Evaluating Data Drift profiles across {len(num_cols)} Numerical "
        f"and {len(cat_cols)} Categorical features."
    )

    # Apply specialized statistical checks across correctly routed variables
    report, global_drift = detect_feature_drift(
        reference_df=ref_transformed,
        current_df=cur_transformed,
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    return report, global_drift
