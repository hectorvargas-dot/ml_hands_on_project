import logging

from pyspark.sql import DataFrame

logger = logging.getLogger(__name__)


def save_feature_splits(
    X_train,
    X_test,
    base_path: str,
):
    X_train.write.format("delta").mode("overwrite").save(
        f"{base_path}/X_train"
    )

    X_test.write.format("delta").mode("overwrite").save(
        f"{base_path}/X_test"
    )


def save_target_splits(
    y_train,
    y_test,
    base_path: str,
):
    y_train.write.format("delta").mode("overwrite").save(
        f"{base_path}/y_train"
    )

    y_test.write.format("delta").mode("overwrite").save(
        f"{base_path}/y_test"
    )   


def create_splits(
    df: DataFrame,
    target_col: str,
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Split Spark DataFrame into train/test features and targets.
    """

    train_df, test_df = df.randomSplit(
        [1 - test_size, test_size],
        seed=seed,
    )

    X_train = train_df.drop(target_col)
    X_test = test_df.drop(target_col)

    y_train = train_df.select(target_col)
    y_test = test_df.select(target_col)

    return X_train, X_test, y_train, y_test


def run_ingest(df: DataFrame) -> None:
    """Runs the data ingestion pipeline.

    Creates train/test splits from a Spark DataFrame and
    persists the resulting feature and target datasets.

    Args:
        df (DataFrame): Input Spark DataFrame containing features and
            target column (`Exited`).

    Raises:
        Exception: Re-raises any exception encountered during ingestion
            after logging the failure details.
    """
    feature_path = "/Volumes/datacartel_dbx/havg_data/volumen/raw_features"
    target_path = "/Volumes/datacartel_dbx/havg_data/volumen/target"

    logger.info("Starting ingestion process.")
    logger.info("Input dataframe rows: %s", df.count())
    logger.info("Input dataframe columns: %s", len(df.columns))

    try:
        logger.info(
            "Creating train/test splits with test_size=0.2 and target_col='Exited'."
        )

        X_train, X_test, y_train, y_test = create_splits(
            df=df,
            target_col="Exited",
            test_size=0.2,
        )

        logger.info(
            "Split completed successfully. "
            "X_train rows: %s, X_test rows: %s, "
            "y_train rows: %s, y_test rows: %s",
            X_train.count(),
            X_test.count(),
            y_train.count(),
            y_test.count(),
        )

        logger.info("Saving feature splits to: %s", feature_path)
        save_feature_splits(
            X_train,
            X_test,
            base_path=feature_path,
        )
        logger.info("Feature splits saved successfully.")

        logger.info("Saving target splits to: %s", target_path)
        save_target_splits(
            y_train,
            y_test,
            base_path=target_path,
        )
        logger.info("Target splits saved successfully.")

        logger.info("Ingestion process completed successfully.")

    except Exception:
        logger.exception("Ingestion process failed.")
        raise