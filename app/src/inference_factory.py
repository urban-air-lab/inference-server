import os
from typing import Dict

from dotenv import load_dotenv
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.logging import get_logger
from ual.mqtt.mqtt_client import MQTTClient

from app.src.sensor import SensorSourceStr
from app.src.service.inference_service import InferenceService
from app.src.service.mlflow_service import MLFlowService

load_dotenv()
logging = get_logger()


def create_inference_service(model_config: Dict) -> InferenceService:
    """
    Factory function to create an InferenceService from a model configuration.

    Args:
        model_config: Dictionary containing model configuration with keys:
                     - name: Model identifier (for logging)
                     - sensor_bucket: InfluxDB bucket name
                     - sensor_name: Sensor identifier
                     - mqtt_topic: MQTT topic for publishing predictions
                     - model_name: MLflow model name
                     - model_version: MLflow model version
                     - inputs: List of sensor fields to query
                     - targets: List of prediction targets
                     - start_time: (optional) For on-demand inference
                     - stop_time: (optional) For on-demand inference

    Returns:
        Configured InferenceService instance
    """
    logging.info(f"Creating inference service for model: {model_config.get('name', 'unnamed')}")

    influx_connector = InfluxDBConnector(
        os.getenv("INFLUX_URL"), os.getenv("INFLUX_TOKEN"), os.getenv("INFLUX_ORG")
    )

    mqtt_client = MQTTClient(
        os.getenv("MQTT_SERVER"),
        int(os.getenv("MQTT_PORT")),
        os.getenv("MQTT_USERNAME"),
        os.getenv("MQTT_PASSWORD"),
    )

    mlflow_client = MLFlowService(
        os.getenv("MLFLOW_USERNAME"), os.getenv("MLFLOW_PASSWORD")
    )

    sensor_source = SensorSourceStr(
        bucket=model_config["sensor_bucket"], sensor=model_config["sensor_name"]
    )

    inference_service = InferenceService(
        influx_connector, mqtt_client, mlflow_client, sensor_source, model_config
    )

    logging.info(
        f"Inference service created for {model_config.get('name')}: "
        f"sensor={model_config['sensor_name']}, "
        f"model={model_config['model_name']}:{model_config['model_version']}"
    )

    return inference_service


def validate_model_config(model_config: Dict) -> None:
    """
    Validate that a model configuration contains all required fields.

    Args:
        model_config: Model configuration dictionary to validate

    Raises:
        ValueError: If required fields are missing
    """
    required_fields = [
        "name",
        "sensor_bucket",
        "sensor_name",
        "mqtt_topic",
        "model_name",
        "model_version",
        "inputs",
        "targets",
    ]

    missing_fields = [field for field in required_fields if field not in model_config]

    if missing_fields:
        raise ValueError(
            f"Model config '{model_config.get('name', 'unnamed')}' is missing required fields: {missing_fields}"
        )

    logging.info(f"Model config '{model_config['name']}' validated successfully")
