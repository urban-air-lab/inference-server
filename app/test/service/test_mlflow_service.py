from unittest.mock import patch

from sklearn.base import BaseEstimator

from app.src.service.mlflow_service import MLFlowService


def test_ml_flow_service():
    with patch("app.src.service.mlflow_service.mlflow"):
        mlflow_service = MLFlowService("username", "password")

    assert isinstance(mlflow_service, MLFlowService)


def test_load_model():
    with patch("app.src.service.mlflow_service.mlflow") as mock_mlflow:
        mock_mlflow.sklearn.load_model.return_value = BaseEstimator()
        mlflow_service = MLFlowService("username", "password")
        model = mlflow_service.load_scikit_learn_model("model")

    assert isinstance(model, BaseEstimator)
