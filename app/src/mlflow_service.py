import os

import mlflow
from sklearn.base import BaseEstimator


class MLFlowClient:
    def __init__(self, username: str, password: str):
        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = password
        mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))

    def load_scikit_learn_model(self, model_path:str) -> BaseEstimator:
        return mlflow.sklearn.load_model(model_path)