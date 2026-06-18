import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


class DynamicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Encapsulates all churn feature generation logic natively inside a
    serializable, scikit-learn compatible transformer.
    """

    def __init__(self, selected_features: list = None):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self  # Stateless transformer

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        features_to_run = (
            self.selected_features
            if self.selected_features is not None
            else (self._get_all_binary_features() + self._get_all_continuous_features())
        )

        for feature in features_to_run:
            # Check both internal lookup prefixes automatically
            computation_method = getattr(self, f"_compute_bin_{feature}", None) or \
                                 getattr(self, f"_compute_cont_{feature}", None)
            
            if computation_method is not None:
                X[feature] = computation_method(X)

        return X

    def _get_all_binary_features(self) -> list:
        """Dynamically lists all binary feature calculation routines."""
        return [
            method.replace("_compute_bin_", "")
            for method in dir(self)
            if method.startswith("_compute_bin_")
        ]

    def _get_all_continuous_features(self) -> list:
        """Dynamically lists all continuous feature calculation routines."""
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
    k_features: int = 5
):
    """
    Combines engineering and preprocessing using DynamicFeatureEngineer class, 
    extracts transformed feature names, and applies mlxtend's SequentialFeatureSelector.
    """
    # 1. Deduce engineered features needed for this experiment layout
    all_layout_cols = (
        routing_config.get("passthrough", []) +
        routing_config.get("standard_scale", []) +
        routing_config.get("one_hot_encode", [])
    )
    
    # Check against all methods available in our custom transformer class
    dummy_engineer = DynamicFeatureEngineer()
    available_transformations = (
        dummy_engineer._get_all_binary_features() + 
        dummy_engineer._get_all_continuous_features()
    )
    needed_engineered = [col for col in all_layout_cols if col in available_transformations]
    
    # 2. FIXED: Use the class-based transformer instead of the old function version
    fe_step = DynamicFeatureEngineer(selected_features=needed_engineered)
    
    prep_step = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), routing_config.get("one_hot_encode", [])),
            ("num", StandardScaler(), routing_config.get("standard_scale", [])),
            ("pass", "passthrough", routing_config.get("passthrough", []))
        ],
        remainder="drop"
    )
    
    transform_pipe = Pipeline([("fe", fe_step), ("prep", prep_step)])
    
    # 3. Fit-Transform Data & Map Correct Post-Encoded String Headings
    X_transformed = transform_pipe.fit_transform(X, y)
    feature_names = transform_pipe.named_steps["prep"].get_feature_names_out()
    X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    
    # 4. Configure Sequential Selector Engine
    direction = "Forward" if forward else "Backward"
    sfs = SFS(
        clone(base_model),
        k_features=k_features,
        forward=forward,
        floating=False,
        scoring="average_precision", # Optimizing directly for PR-AUC
        cv=5,
        n_jobs=-1
    )
    
    print(f"\n--- Running {direction} Feature Selection on {base_model.__class__.__name__} ---")
    sfs.fit(X_transformed_df, y)
    
    print(f"Optimal Feature Subset Size: {len(sfs.k_feature_idx_)}")
    print(f"Optimal Feature Names: {sfs.k_feature_names_}")
    print(f"Best CV Score (PR-AUC): {sfs.k_score_:.4f}")
    
    return sfs, X_transformed_df
