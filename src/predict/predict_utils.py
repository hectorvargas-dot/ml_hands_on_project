from asyncio import selector_events
import logging
import pandas as pd
from datetime import datetime

import mlflow


logger = logging.getLogger(__name__)

def format_mlflow_date(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, datetime):
        return value.isoformat()

    return datetime.fromtimestamp(value / 1000).isoformat()


def find_best_mlflow_runs():
    results = []

    logger.info("Searching MLflow experiments...")

    experiments = mlflow.search_experiments()

    logger.info(
        "Found %d MLflow experiments",
        len(experiments)
    )

    for experiment in experiments:
        logger.info(
            "Processing experiment: %s (ID: %s)",
            experiment.name,
            experiment.experiment_id,
        )

        runs_df = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(
                "tags.mlflow.parentRunId IS NULL "
                "AND status = 'FINISHED'"
            ),
            order_by=[
                "metrics.test_pr_auc DESC",
                "metrics.best_optuna_val_pr_auc DESC",
            ],
            max_results=1,
        )

        if runs_df.empty:
            logger.warning(
                "No completed parent runs found for experiment '%s'",
                experiment.name,
            )
            continue

        best_run = runs_df.iloc[0]

        # Prefer test metric, fallback to optuna validation metric
        score = (
            best_run.get("metrics.test_pr_auc")
            or best_run.get("metrics.best_optuna_val_pr_auc")
        )

        if score is None:
            logger.warning(
                "Experiment '%s' has no PR-AUC metrics",
                experiment.name,
            )
            continue

        results.append(
            {
                "experiment": experiment.name,
                "experiment_id": str(experiment.experiment_id),
                "run_id": str(best_run["run_id"]),
                "run_name": best_run.get(
                    "tags.mlflow.runName",
                    "Unnamed Model"
                ),
                "test_pr_auc": float(score),
                "date": best_run["start_time"].isoformat(),
            }
        )

        logger.info(
            "Best run for '%s': %s | PR-AUC %.5f",
            experiment.name,
            best_run["run_id"],
            score,
        )

    return results