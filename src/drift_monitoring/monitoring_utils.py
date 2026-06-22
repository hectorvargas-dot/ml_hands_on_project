import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, bool]:
    """Detects structural data drift between reference and production datasets.

    Applies a two-sample Kolmogorov-Smirnov test for numerical continuous features
    and a Chi-Square contingency test for categorical features to evaluate if
    the production data distribution has statistically shifted from the baseline.

    Args:
        reference_df (pd.DataFrame): Baseline/training dataset.
        current_df (pd.DataFrame): Incoming production/inference dataset.
        num_cols (List[str]): List of continuous numerical column names.
        cat_cols (List[str]): List of categorical column names.
        alpha (float): Significance threshold for rejecting the null hypothesis 
            (no drift). Defaults to 0.05.

    Returns:
        Tuple[pd.DataFrame, bool]: 
            - A summary report DataFrame detailing stats, p-values, and drift status per feature.
            - A global boolean flag indicating if at least one feature has drifted.
    """
    start_time = time.time()
    logger.info("Initializing automated feature drift check profile.")
    
    drift_report = []
    global_drift_detected = False

    # 1. Evaluate Continuous Numerical Features (KS Test)
    for col in num_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            logger.warning(f"Numerical feature '{col}' missing from dataframes. Skipping.")
            continue

        ref_data = reference_df[col].dropna()
        cur_data = current_df[col].dropna()

        # Execute two-sample Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)
        has_drift = p_value < alpha

        if has_drift:
            global_drift_detected = True
            logger.warning(f"Numerical Drift Detected in feature '{col}' (p-val: {p_value:.5f})")

        drift_report.append({
            "feature": col,
            "type": "numerical",
            "metric_name": "KS-Statistic",
            "metric_value": ks_stat,
            "p_value": p_value,
            "drift_detected": has_drift
        })

    # 2. Evaluate Categorical Features (Chi-Square Test)
    for col in cat_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            logger.warning(f"Categorical feature '{col}' missing from dataframes. Skipping.")
            continue

        # Compute observed frequency distributions
        ref_counts = reference_df[col].value_counts()
        cur_counts = current_df[col].value_counts()

        # Align categories present across both datasets
        all_categories = ref_counts.index.union(cur_counts.index)
        ref_aligned = ref_counts.reindex(all_categories, fill_value=0) + 1  # Laplace smoothing
        cur_aligned = cur_counts.reindex(all_categories, fill_value=0) + 1

        # Execute Chi-Square Contingency Test
        contingency_table = np.array([ref_aligned.values, cur_aligned.values])
        chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
        has_drift = p_value < alpha

        if has_drift:
            global_drift_detected = True
            logger.warning(f"Categorical Drift Detected in feature '{col}' (p-val: {p_value:.5f})")

        drift_report.append({
            "feature": col,
            "type": "categorical",
            "metric_name": "Chi2-Statistic",
            "metric_value": chi2_stat,
            "p_value": p_value,
            "drift_detected": has_drift
        })

    report_df = pd.DataFrame(drift_report)
    
    logger.info(
        f"Feature drift auditing completed in {time.time() - start_time:.2f}s. "
        f"Global system drift status: {global_drift_detected}"
    )
    
    return report_df, global_drift_detected
