from src.train.ingest import run_ingest
from src.train.feature_engineering import run_feature_engineering
from src.train.model_optimization import run_model_optimization

def run_train_pipeline(df, experiment_name):
    run_ingest(df)
    run_feature_engineering()
    run_model_optimization(experiment_name)
