from unittest.mock import patch

from sklearn.base import BaseEstimator

from app.src.service.mlflow_service import MLFlowService


def test_ml_flow_service():
    mlflow_service = MLFlowService("username", "password")
    assert isinstance(mlflow_service, MLFlowService)

def test_load_model():
    mlflow_service = MLFlowService("username", "password")

    with patch("app.src.service.mlflow_service.mlflow") as mock_api:
        mock_api.sklearn.load_model.return_value = BaseEstimator()
        model = mlflow_service.load_scikit_learn_model("model")
        assert isinstance(model, BaseEstimator)



