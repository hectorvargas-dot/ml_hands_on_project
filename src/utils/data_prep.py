import os
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split


def create_splits(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 43,
):
    """
    Split a DataFrame into either:

    - train/test (if val_size is None)
    - train/validation/test (if val_size is provided)

    Returns:
        Train/test:
            X_train, X_test, y_train, y_test

        Train/val/test:
            X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        stratify=y,
        test_size=test_size,
        random_state=random_state,
    )

    # Simple train/test split
    if val_size is None or val_size == 0:
        return X_temp, X_test, y_temp, y_test

    # Train/validation/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        stratify=y_temp,
        test_size=val_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_feature_splits(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    base_path: str,
    X_val: pd.DataFrame | None = None,
) -> None:
    """Saves feature splits."""
    os.makedirs(base_path, exist_ok=True)

    X_train.to_csv(os.path.join(base_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(base_path, "X_test.csv"), index=False)

    if X_val is not None:
        X_val.to_csv(os.path.join(base_path, "X_val.csv"), index=False)


def save_target_splits(
    y_train: pd.Series,
    y_test: pd.Series,
    base_path: str,
    y_val: pd.Series | None = None,
) -> None:
    """Saves target splits."""
    os.makedirs(base_path, exist_ok=True)

    y_train.to_csv(os.path.join(base_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(base_path, "y_test.csv"), index=False)

    if y_val is not None:
        y_val.to_csv(os.path.join(base_path, "y_val.csv"), index=False)


VOLUME_PATH = "/Volumes/datacartel_dbx/havg_data/volumen"


def save_feature_splits_to_volume(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    X_val: pd.DataFrame | None = None,
    volume_path: str = VOLUME_PATH,
) -> None:
    """Saves feature splits to a Unity Catalog Volume.

    Writes X_train.csv and X_test.csv (and optionally X_val.csv) to the
    specified UC Volume path, overwriting any existing files.

    Args:
        X_train: Training feature DataFrame.
        X_test: Test feature DataFrame.
        X_val: Optional validation feature DataFrame.
        volume_path: UC Volume path to write to.
    """
    X_train.to_csv(os.path.join(volume_path, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(volume_path, "X_test.csv"), index=False)

    if X_val is not None:
        X_val.to_csv(os.path.join(volume_path, "X_val.csv"), index=False)


def save_target_splits_to_volume(
    y_train: pd.Series,
    y_test: pd.Series,
    y_val: pd.Series | None = None,
    volume_path: str = VOLUME_PATH,
) -> None:
    """Saves target splits to a Unity Catalog Volume.

    Writes y_train.csv and y_test.csv (and optionally y_val.csv) to the
    specified UC Volume path, overwriting any existing files.

    Args:
        y_train: Training target Series.
        y_test: Test target Series.
        y_val: Optional validation target Series.
        volume_path: UC Volume path to write to.
    """
    y_train.to_csv(os.path.join(volume_path, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(volume_path, "y_test.csv"), index=False)

    if y_val is not None:
        y_val.to_csv(os.path.join(volume_path, "y_val.csv"), index=False)
