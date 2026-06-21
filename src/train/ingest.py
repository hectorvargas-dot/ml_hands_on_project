import logging

import pandas as pd

from src.utils import data_prep as prep


logger = logging.getLogger(__name__)


def run_ingest(df: pd.DataFrame) -> None:
    """Runs the data ingestion pipeline.

    This function creates train/test splits from the input dataframe and
    persists the resulting feature and target datasets to disk.

    Args:
        df (pd.DataFrame): Input dataset containing features and target column
            (`Exited`).

    Raises:
        Exception: Re-raises any exception encountered during ingestion after
            logging the failure details.
    """
    feature_path = "data/processed/raw_features"
    target_path = "data/processed/target"

    logger.info("Starting ingestion process.")
    logger.info("Input dataframe shape: %s", df.shape)

    try:
        logger.info(
            "Creating train/test splits with test_size=0.2 and target_col='Exited'."
        )

        X_train, X_test, y_train, y_test = prep.create_splits(
            df=df,
            target_col="Exited",
            test_size=0.2,
        )

        logger.info(
            "Split completed successfully. "
            "X_train: %s, X_test: %s, y_train: %s, y_test: %s",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
        )

        logger.info("Saving feature splits to: %s", feature_path)
        prep.save_feature_splits(
            X_train,
            X_test,
            base_path=feature_path,
        )
        logger.info("Feature splits saved successfully.")

        logger.info("Saving target splits to: %s", target_path)
        prep.save_target_splits(
            y_train,
            y_test,
            base_path=target_path,
        )
        logger.info("Target splits saved successfully.")

        logger.info("Ingestion process completed successfully.")

    except Exception:
        logger.exception("Ingestion process failed.")
        raise
