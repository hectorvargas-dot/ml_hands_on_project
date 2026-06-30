import mlflow
import pandas as pd
import logging
import sys
logger = logging.getLogger(__name__)


def run_predict_pipeline(
    df: pd.DataFrame,
    model_uri: str,
):
    mlflow.set_registry_uri("databricks-uc")
    pipeline = mlflow.sklearn.load_model(model_uri)

    customer_ids = df["CustomerId"].copy()

    # Remove identifier column if it was not used during training
    X = df.drop(columns=["CustomerId"])

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    results = [
        {
            "CustomerId": int(customer_id),
            "prediction": int(pred),
            "churn_probability": round(float(prob), 4)
        }
        for customer_id, pred, prob in zip(
            customer_ids,
            y_pred,
            y_proba
        )
    ]

    return results

if __name__ == "__main__":
    model_uri  = sys.argv[1]
    
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
          '/Volumes/datacartel_dbx/havg_data/volumen/X_val.csv',
          format => 'csv',
          header => true
        )
    """).toPandas()

    logger.info("Data loaded from Volume CSV. Shape: %s", df.shape)
    results = run_predict_pipeline(df, model_uri)
    print(results)
    # Save results to volume
    results_df = pd.DataFrame(results)
    results_df.to_csv('/Volumes/datacartel_dbx/havg_data/volumen/predictions.csv', index=False)