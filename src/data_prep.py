import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_stratified_splits(df: pd.DataFrame, target_col: str, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 43):
    """
    Splits a DataFrame into stratified train, validation, and test sets.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # First split: Isolate the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )
    
    # Second split: Isolate train and validation sets from the remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=val_size, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_feature_splits(X_train, X_val, X_test, base_path: str):  
    """Saves feature splits to the specified directory, creating it if it doesn't exist."""
    os.makedirs(base_path, exist_ok=True)
    X_train.to_csv(f"{base_path}/X_train.csv", index=False)
    X_val.to_csv(f"{base_path}/X_val.csv", index=False)
    X_test.to_csv(f"{base_path}/X_test.csv", index=False)


def save_target_splits(y_train, y_val, y_test, base_path: str):
    """Saves target splits to the specified directory, creating it if it doesn't exist."""
    os.makedirs(base_path, exist_ok=True)
    y_train.to_csv(f"{base_path}/y_train.csv", index=False)
    y_val.to_csv(f"{base_path}/y_val.csv", index=False)
    y_test.to_csv(f"{base_path}/y_test.csv", index=False)

