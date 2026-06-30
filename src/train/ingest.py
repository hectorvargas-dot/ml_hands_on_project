import logging
import sys

import pandas as pd

# Ensure project root is in path for module imports
sys.path.insert(0, '/Workspace/Users/hector.vargas@wizeline.com/ml_hands_on_project')

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

        logger.info("Saving splits to UC Volume...")
        prep.save_feature_splits_to_volume(X_train, X_test)
        prep.save_target_splits_to_volume(y_train, y_test)
        logger.info("Splits saved to Volume successfully.")

        logger.info("Ingestion process completed successfully.")

    except Exception:
        logger.exception("Ingestion process failed.")
        raise


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    logging.basicConfig(level=logging.INFO)

    spark = SparkSession.builder.getOrCreate()

    # Read data using the churn_query_select_all SQL
    df = spark.sql("""
        SELECT
          `Satisfaction Score` AS satisfaction_score,
          `Card Type` AS card_type,
          `Point Earned` AS point_earned,
          * EXCEPT (`Satisfaction Score`, `Card Type`, `Point Earned`)
        FROM read_files(
          '/Volumes/datacartel_dbx/havg_data/volumen/train_test_datasset.csv',
          format => 'csv',
          header => true
        )
    """).toPandas()

    logger.info("Data loaded from Volume CSV. Shape: %s", df.shape)
    run_ingest(df)
