from sqlalchemy.engine import result
import json
import logging
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.utils import feature_engineering_utils as fe


logger = logging.getLogger(__name__)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "raw_features"
TARGET_PATH = PROJECT_ROOT / "data" / "processed" / "target"
FEATURE_REGISTRY_PATH = PROJECT_ROOT / "src" / "selected_features.json"


def run_feature_engineering() -> None:
    """Runs feature engineering and sequential feature selection pipeline.

    Loads processed train/test datasets, defines feature transformation
    strategies, evaluates feature subsets using sequential feature selection,
    and saves the best feature configuration as a JSON registry.

    Raises:
        Exception: Re-raises exceptions after logging pipeline failure.
    """
    logger.info("Starting feature engineering pipeline.")

    try:
        # ------------------------------------------------------------------
        # Load datasets
        # ------------------------------------------------------------------
        logger.info("Loading processed datasets.")

        X_train = pd.read_csv(RAW_FEATURES_PATH / "X_train.csv")
        y_train = pd.read_csv(TARGET_PATH / "y_train.csv").squeeze()

        logger.info(
            "Datasets loaded successfully. X_train=%s",
            X_train.shape,
        )

        # ------------------------------------------------------------------
        # Feature definitions
        # ------------------------------------------------------------------
        logger.info("Building feature engineering configuration.")

        nomod_columns = [
            "HasCrCard",
            "IsActiveMember",
        ]

        dummyfy_columns = [
            "Card Type",
            "Gender",
        ]

        norm_std_columns = [
            "Balance",
            "Point Earned",
            "CreditScore",
            "Age",
            "Tenure",
            "Satisfaction Score",
            "EstimatedSalary",
        ]

        selected_features = [
            "no_balance",
            "Num_Of_Products_1",
            "Num_Of_Products_2",
            "is_germany",
            "is_spain",
            "is_france",
            "middle_age",
            "Balance_x_Tenure",
            "Age_x_IsActive",
            "Balance_to_Salary",
            "Balance_per_Product",
            "Salary_per_Product",
            "CreditScore_per_Age",
            "Tenure_per_Age",
            "Inactive_x_Balance",
            "Inactive_x_Age",
            "Products_x_Active",
            "Balance_plus_Salary",
            "WealthScore",
            "CreditScore_x_Age",
            "LogBalance",
            "LogAge",
            "Age2",
            "Balance2",
            "Tenure2",
            "Products_per_Tenure",
            "Balance_per_Tenure",
        ]

        feature_engineer_object = fe.DynamicFeatureEngineer(
            selected_features=selected_features
        )

        binary_features = (
            feature_engineer_object._get_all_binary_features()
        )

        continuous_features = (
            feature_engineer_object._get_all_continuous_features()
        )

        # ------------------------------------------------------------------
        # Experiment configuration
        # ------------------------------------------------------------------
        experiment_registry = {
            "experiment_1": {
                "passthrough": nomod_columns + binary_features,
                "standard_scale": norm_std_columns + continuous_features,
                "one_hot_encode": dummyfy_columns,
            }
        }

        models_zoo = {
            "xgboost": XGBClassifier(
                random_state=42,
                n_jobs=-1,
                eval_metric="aucpr",
            )
        }

        layout = experiment_registry["experiment_1"]

        search_directions = {
            "Forward": True,
            "Backward": False,
        }

        all_loop_results = []

        # ------------------------------------------------------------------
        # Feature selection sweep
        # ------------------------------------------------------------------
        logger.info("Starting sequential feature selection sweep.")

        for model_name, model_obj in models_zoo.items():
            for direction_label, is_forward in search_directions.items():

                logger.info(
                    "Running model=%s direction=%s",
                    model_name,
                    direction_label,
                )

                for target_features in range(14, 18):

                    logger.info(
                        "Evaluating subset size=%s",
                        target_features,
                    )

                    sfs_obj, _ = fe.run_sequential_selection(
                        X=X_train,
                        y=y_train,
                        routing_config=layout,
                        base_model=model_obj,
                        forward=is_forward,
                        k_features=target_features,
                    )

                    all_loop_results.append(
                        {
                            "model": model_name,
                            "direction": direction_label,
                            "n_features": target_features,
                            "pr_auc_cv": sfs_obj.k_score_,
                            "features": sorted(
                                list(sfs_obj.k_feature_names_)
                            ),
                        }
                    )

        # ------------------------------------------------------------------
        # Select best configuration
        # ------------------------------------------------------------------
        logger.info("Selecting best feature configuration.")

        results_df = pd.DataFrame(all_loop_results)

        best_row = results_df.loc[
            results_df["pr_auc_cv"].idxmax()
        ]

        best_features = best_row["features"]

        logger.info(
            "Best configuration found. PR-AUC=%s, Features=%s, direction=%s, model=%s, n_features=%s",
            best_row["pr_auc_cv"],
            len(best_features),
            best_row["direction"],
            best_row["model"],
            best_row["n_features"],
        )

        result = {
            "selected_features": [],
            "nomod_columns": [],
            "dummyfy_columns": [],
            "norm_std_columns": [],
        }

        for feature in best_features:
            clean_name = feature.split("__", 1)[-1]

            if clean_name in nomod_columns:
                result["nomod_columns"].append(clean_name)

            elif any(
                clean_name.startswith(f"{col}_")
                for col in dummyfy_columns
            ):
                original_column = next(
                    col
                    for col in dummyfy_columns
                    if clean_name.startswith(f"{col}_")
                )

                if original_column not in result["dummyfy_columns"]:
                    result["dummyfy_columns"].append(original_column)

            elif clean_name in norm_std_columns:
                result["norm_std_columns"].append(clean_name)

            else:
                result["selected_features"].append(clean_name)

        # ------------------------------------------------------------------
        # Save registry
        # ------------------------------------------------------------------
        logger.info(
            "Saving selected feature registry to %s",
            FEATURE_REGISTRY_PATH,
        )

        with open(FEATURE_REGISTRY_PATH, "w") as file:
            json.dump(result, file, indent=4)

        logger.info("Feature engineering completed successfully.")

    except Exception:
        logger.exception(
            "Feature engineering pipeline failed."
        )
        raise
