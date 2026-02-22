import os
import pandas as pd
from dotenv import load_dotenv
from ual.get_config import get_config
from ual.logging import get_logger
from ual.influx import sensors
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
from ual.mqtt.mqtt_client import MQTTClient

from app.src.mlflow_service import MLFlowClient
from app.src.inference import InferenceService

load_dotenv()
logging = get_logger()


def run_on_demand_inference(config_path: str) -> None:
    """
    Run inference on-demand for a specified time range.

    Args:
        config_path: Path to the YAML configuration file containing:
                    - start_time: ISO 8601 UTC timestamp
                    - stop_time: ISO 8601 UTC timestamp
                    - inputs: List of sensor fields to query
                    - targets: List of prediction targets
                    - model_name: MLflow model name
                    - model_version: MLflow model version
    """
    logging.info(f"Loading configuration from {config_path}")
    config: dict = get_config(config_path)

    # Validate required config fields
    required_fields = ["start_time", "stop_time", "inputs", "model_name", "model_version"]
    for field in required_fields:
        if field not in config:
            logging.error(f"Missing required field '{field}' in configuration")
            raise ValueError(f"Missing required field '{field}' in configuration")

    # Initialize services
    influx_connector = InfluxDBConnector(
        os.getenv("INFLUX_URL"),
        os.getenv("INFLUX_TOKEN"),
        os.getenv("INFLUX_ORG")
    )

    mqtt_client = MQTTClient(
        os.getenv("MQTT_SERVER"),
        int(os.getenv("MQTT_PORT")),
        os.getenv("MQTT_USERNAME"),
        os.getenv("MQTT_PASSWORD")
    )

    mlflow_client = MLFlowClient(
        os.getenv("MLFLOW_USERNAME"),
        os.getenv("MLFLOW_PASSWORD")
    )

    sensor_source = SensorSource(
        bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET,
        sensor=sensors.UALSensors.UAL_3
    )

    # Create inference service
    inference_service = InferenceService(
        influx_connector,
        mqtt_client,
        mlflow_client,
        sensor_source,
        config
    )

    # Run inference for the configured time range
    logging.info(f"Starting on-demand inference from {config['start_time']} to {config['stop_time']}")
    inference_service.initial_inference()
    logging.info("On-demand inference completed successfully")


if __name__ == "__main__":
    # Configure the path to your inference config file here
    CONFIG_PATH: str = "./run_config.yaml"

    run_on_demand_inference(CONFIG_PATH)
