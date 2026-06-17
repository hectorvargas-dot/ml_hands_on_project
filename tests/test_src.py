import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import data_prep, mlflow_utils


def test_create_stratified_splits():
    # Generate dummy binary classification dataframe
    df = pd.DataFrame(
        {"feature_1": range(100), "feature_2": range(100, 200), "target": [0] * 50 + [1] * 50}
    )

    X_train, X_val, X_test, y_train, y_val, y_test = data_prep.create_stratified_splits(
        df, target_col="target", test_size=0.2, val_size=0.2, random_state=42
    )

    assert len(X_train) == 64
    assert len(X_val) == 16
    assert len(X_test) == 20
    assert len(y_train) == 64
    assert len(y_val) == 16
    assert len(y_test) == 20

    # Assert stratified splits
    assert (y_train == 0).sum() == 32
    assert (y_train == 1).sum() == 32
    assert (y_val == 0).sum() == 8
    assert (y_val == 1).sum() == 8
    assert (y_test == 0).sum() == 10
    assert (y_test == 1).sum() == 10


def test_save_splits(tmp_path):
    X_train = pd.DataFrame({"a": [1, 2]})
    X_val = pd.DataFrame({"a": [3, 4]})
    X_test = pd.DataFrame({"a": [5, 6]})

    base_path = os.path.join(tmp_path, "splits")
    data_prep.save_feature_splits(X_train, X_val, X_test, base_path)

    assert os.path.exists(os.path.join(base_path, "X_train.csv"))
    assert os.path.exists(os.path.join(base_path, "X_val.csv"))
    assert os.path.exists(os.path.join(base_path, "X_test.csv"))

    y_train = pd.Series([1, 2])
    y_val = pd.Series([3, 4])
    y_test = pd.Series([5, 6])

    data_prep.save_target_splits(y_train, y_val, y_test, base_path)

    assert os.path.exists(os.path.join(base_path, "y_train.csv"))
    assert os.path.exists(os.path.join(base_path, "y_val.csv"))
    assert os.path.exists(os.path.join(base_path, "y_test.csv"))


def test_mlflow_utils_integration(tmp_path):
    # Setup paths
    db_path = os.path.join(tmp_path, "test_mlflow.db")
    artifacts_dir = os.path.join(tmp_path, "mlartifacts")
    experiment_name = "test_experiment"

    # Init experiment
    mlflow_utils.init_mlflow_experiment(experiment_name, db_path, artifacts_dir)

    # Train dummy model & log evaluation
    import mlflow

    X_train = pd.DataFrame({"feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]})
    y_train = pd.Series([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    X_test = pd.DataFrame({"feature_1": [1.5, 5.5, 9.5]})
    y_test = pd.Series([0, 1, 1])

    pipeline = Pipeline([("preprocessing", StandardScaler()), ("model", LogisticRegression())])
    pipeline.fit(X_train, y_train)

    with mlflow.start_run(run_name="test_run"):
        metrics = mlflow_utils.log_model_evaluation(
            pipeline, X_test, y_test, input_features=["feature_1"], artifact_path="model"
        )

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

    # Get summary
    summary_df = mlflow_utils.get_experiment_summary(experiment_name)
    assert not summary_df.empty
    assert "run_id" in summary_df.columns
    assert "metrics.accuracy" in summary_df.columns

    # Remove run
    run_id = summary_df.iloc[0]["run_id"]
    mlflow_utils.remove_recent_runs(experiment_name, count=1)

    # Verify summary is empty now
    summary_df_after = mlflow_utils.get_experiment_summary(experiment_name)
    assert summary_df_after.empty or run_id not in summary_df_after["run_id"].values
