import logging
from typing import Tuple

import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def target_drift(
    reference_target: pd.Series,
    current_target: pd.Series,
    alpha: float = 0.05,
) -> Tuple[pd.DataFrame, bool]:
    """Detects target drift between baseline and production labels.

    Uses a Chi-Square test to compare class distributions.

    Args:
        reference_target (pd.Series):
            Historical training labels.

        current_target (pd.Series):
            Recent production labels.

        alpha (float):
            Significance threshold.

    Returns:
        Tuple[pd.DataFrame, bool]:
            Drift report and drift flag.
    """

    ref_counts = reference_target.value_counts().sort_index()
    cur_counts = current_target.value_counts().sort_index()

    all_classes = ref_counts.index.union(cur_counts.index)

    ref_aligned = ref_counts.reindex(all_classes, fill_value=0) + 1
    cur_aligned = cur_counts.reindex(all_classes, fill_value=0) + 1

    contingency = [ref_aligned.values, cur_aligned.values]

    chi2_stat, p_value, _, _ = stats.chi2_contingency(
        contingency
    )
    reference_rate = reference_target.mean()
    current_rate = current_target.mean()

    absolute_change = current_rate - reference_rate
    relative_change = absolute_change / reference_rate

    drift_detected = p_value < alpha

    logger.info(
        "Target Drift Check | Chi2=%.4f | p-value=%.6f | Drift=%s",
        chi2_stat,
        p_value,
        drift_detected,
    )

    return {
        "metric_name": "Chi2-Statistic",
        "metric_value": float(chi2_stat),
        "p_value": float(p_value),
        "drift_detected": bool(drift_detected),
        "reference_positive_rate": float(reference_rate),
        "current_positive_rate": float(current_rate),
        "absolute_change": float(absolute_change),
        "relative_change_pct": float(relative_change * 100),
    }
