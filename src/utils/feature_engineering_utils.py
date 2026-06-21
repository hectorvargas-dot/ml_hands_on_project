import logging
import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


logger = logging.getLogger(__name__)


class DynamicFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer for dynamic feature generation.

    This transformer creates engineered churn features using predefined
    computation methods. Features are discovered dynamically based on method
    naming conventions:

    - `_compute_bin_*` for binary features.
    - `_compute_cont_*` for continuous features.

    The transformer is stateless and can be serialized inside sklearn
    pipelines.

    Args:
        selected_features (list, optional): Specific engineered features to
            generate. If None, all available features are generated.
            Defaults to None.
    """

    def __init__(self, selected_features: list = None):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        """Fits the transformer.

        The transformer does not learn parameters, so this method returns
        itself unchanged.

        Args:
            X (pd.DataFrame): Input dataframe.
            y (pd.Series, optional): Target variable. Defaults to None.

        Returns:
            DynamicFeatureEngineer: Fitted transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generates engineered features.

        Args:
            X (pd.DataFrame): Input feature dataframe.

        Returns:
            pd.DataFrame: Dataframe containing original and engineered
            features.

        Raises:
            Exception: If feature generation fails.
        """
        start_time = time.time()

        try:
            X = X.copy()

            features_to_run = (
                self.selected_features
                if self.selected_features is not None
                else (
                    self._get_all_binary_features()
                    + self._get_all_continuous_features()
                )
            )

            logger.debug(
                "Generating %s engineered features.",
                len(features_to_run),
            )

            for feature in features_to_run:
                computation_method = (
                    getattr(self, f"_compute_bin_{feature}", None)
                    or getattr(self, f"_compute_cont_{feature}", None)
                )

                if computation_method:
                    X[feature] = computation_method(X)

            logger.info(
                "Feature engineering completed. Added %s features in %.2fs.",
                len(features_to_run),
                time.time() - start_time,
            )

            return X

        except Exception:
            logger.exception(
                "Feature engineering transformation failed."
            )
            raise

    def _get_all_binary_features(self) -> list:
        """Returns available binary feature generators.

        Returns:
            list: Names of binary engineered features.
        """
        return [
            method.replace("_compute_bin_", "")
            for method in dir(self)
            if method.startswith("_compute_bin_")
        ]

    def _get_all_continuous_features(self) -> list:
        """Returns available continuous feature generators.

        Returns:
            list: Names of continuous engineered features.
        """
        return [
            method.replace("_compute_cont_", "")
            for method in dir(self)
            if method.startswith("_compute_cont_")
        ]

    # BINARY FLAGS
    def _compute_bin_is_silver(self, X):        return X["Card Type"] == "SILVER"
    def _compute_bin_is_germany(self, X):       return X["Geography"] == "Germany"
    def _compute_bin_is_spain(self, X):         return X["Geography"] == "Spain"
    def _compute_bin_is_france(self, X):        return X["Geography"] == "France"
    def _compute_bin_no_balance(self, X):       return X["Balance"] < 2500
    def _compute_bin_middle_age(self, X):       return X["Age"].between(25, 45, inclusive="neither")

    # PRODUCT COUNTS
    def _compute_bin_Num_Of_Products_1(self, X):  return X["NumOfProducts"] == 1
    def _compute_bin_Num_Of_Products_2(self, X):  return X["NumOfProducts"] == 2
    def _compute_bin_Num_Of_Products_3(self, X):  return X["NumOfProducts"] == 3
    def _compute_bin_Num_Of_Products_4(self, X):  return X["NumOfProducts"] == 4

    # POLYNOMIAL & INTERACTION TERMS
    def _compute_cont_Age_x_IsActive(self, X):     return X["Age"] * X["IsActiveMember"]
    def _compute_cont_Balance_x_Tenure(self, X):   return X["Balance"] * X["Tenure"]
    def _compute_cont_CreditScore_x_Age(self, X):  return X["CreditScore"] * X["Age"]

    # FINANCIAL & ENGAGEMENT RATIOS
    def _compute_cont_Balance_to_Salary(self, X):    return X["Balance"] / (X["EstimatedSalary"] + 1)
    def _compute_cont_Balance_per_Product(self, X):  return X["Balance"] / (X["NumOfProducts"] + 1)
    def _compute_cont_Salary_per_Product(self, X):   return X["EstimatedSalary"] / (X["NumOfProducts"] + 1)
    def _compute_cont_CreditScore_per_Age(self, X):  return X["CreditScore"] / (X["Age"] + 1)
    def _compute_cont_Tenure_per_Age(self, X):       return X["Tenure"] / (X["Age"] + 1)

    # BEHAVIORAL CROSS-PRODUCTS
    def _compute_cont_Inactive_x_Balance(self, X):   return (1 - X["IsActiveMember"]) * X["Balance"]
    def _compute_cont_Inactive_x_Age(self, X):       return (1 - X["IsActiveMember"]) * X["Age"]
    def _compute_cont_Products_x_Active(self, X):    return X["NumOfProducts"] * X["IsActiveMember"]

    # MONETARY ACCUMULATIONS & NON-LINEAR SCALING
    def _compute_cont_Balance_plus_Salary(self, X):  return X["Balance"] + X["EstimatedSalary"]
    def _compute_cont_WealthScore(self, X):          return 0.6 * X["Balance"] + 0.4 * X["EstimatedSalary"]
    def _compute_cont_LogBalance(self, X):           return np.log1p(X["Balance"])
    def _compute_cont_LogAge(self, X):               return np.log1p(X["Age"])

    # POLYNOMIAL DEGREES
    def _compute_cont_Age2(self, X):                 return X["Age"] ** 2
    def _compute_cont_Balance2(self, X):             return X["Balance"] ** 2
    def _compute_cont_Tenure2(self, X):              return X["Tenure"] ** 2

    # TEMPORAL PRODUCT DENSITIES
    def _compute_cont_Products_per_Tenure(self, X):  return X["NumOfProducts"] / (X["Tenure"] + 1)
    def _compute_cont_Balance_per_Tenure(self, X):   return X["Balance"] / (X["Tenure"] + 1)


def run_sequential_selection(
    X,
    y,
    routing_config: dict,
    base_model,
    forward: bool = True,
    k_features: int = 5,
):
    """Runs preprocessing and sequential feature selection.

    The function dynamically engineers features, applies preprocessing,
    transforms the dataset, and searches for the optimal feature subset using
    mlxtend SequentialFeatureSelector.

    Args:
        X (pd.DataFrame): Input feature dataframe.
        y (pd.Series): Target variable.
        routing_config (dict): Feature routing configuration containing:
            - passthrough
            - standard_scale
            - one_hot_encode
        base_model: Estimator used for feature selection.
        forward (bool): Selection direction.
            True performs forward selection.
            False performs backward selection.
        k_features (int): Number of features to select.

    Returns:
        tuple:
            sfs (SequentialFeatureSelector): Fitted selector.
            X_transformed_df (pd.DataFrame): Preprocessed feature matrix.

    Raises:
        Exception: If preprocessing or feature selection fails.
    """

    start_time = time.time()

    try:
        direction = "Forward" if forward else "Backward"

        logger.info(
            "Starting %s feature selection using %s.",
            direction,
            base_model.__class__.__name__,
        )

        all_layout_cols = (
            routing_config.get("passthrough", [])
            + routing_config.get("standard_scale", [])
            + routing_config.get("one_hot_encode", [])
        )

        dummy_engineer = DynamicFeatureEngineer()

        available_transformations = (
            dummy_engineer._get_all_binary_features()
            + dummy_engineer._get_all_continuous_features()
        )

        needed_engineered = [
            col
            for col in all_layout_cols
            if col in available_transformations
        ]

        logger.debug(
            "Selected engineered features: %s",
            needed_engineered,
        )

        fe_step = DynamicFeatureEngineer(
            selected_features=needed_engineered
        )

        prep_step = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",
                        sparse_output=False,
                    ),
                    routing_config.get("one_hot_encode", []),
                ),
                (
                    "num",
                    StandardScaler(),
                    routing_config.get("standard_scale", []),
                ),
                (
                    "pass",
                    "passthrough",
                    routing_config.get("passthrough", []),
                ),
            ],
            remainder="drop",
        )

        transform_pipe = Pipeline(
            [
                ("fe", fe_step),
                ("prep", prep_step),
            ]
        )

        logger.info("Applying preprocessing pipeline.")

        X_transformed = transform_pipe.fit_transform(X, y)

        feature_names = (
            transform_pipe
            .named_steps["prep"]
            .get_feature_names_out()
        )

        X_transformed_df = pd.DataFrame(
            X_transformed,
            columns=feature_names,
        )

        logger.info(
            "Transformation completed. Shape=%s",
            X_transformed_df.shape,
        )

        sfs = SFS(
            clone(base_model),
            k_features=k_features,
            forward=forward,
            floating=False,
            scoring="average_precision",
            cv=5,
            n_jobs=5,
        )

        logger.info(
            "Running Sequential Feature Selection. k_features=%s",
            k_features,
        )

        sfs.fit(X_transformed_df, y)

        logger.info(
            "Feature selection completed. Selected=%s PR-AUC=%.4f",
            len(sfs.k_feature_idx_),
            sfs.k_score_,
        )

        logger.debug(
            "Selected features: %s",
            sfs.k_feature_names_,
        )

        logger.info(
            "Total feature selection runtime %.2fs",
            time.time() - start_time,
        )

        return sfs, X_transformed_df

    except Exception:
        logger.exception(
            "Sequential feature selection failed."
        )
        raise