import os
from datetime import datetime

import numpy as np
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from sklearn.base import BaseEstimator
from ual.data_processor import DataProcessor
from ual.get_config import get_config
from ual.influx import sensors
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
from ual.logging import get_logger
from ual.mqtt.mqtt_client import MQTTClient

from app.src.inference_factory import create_inference_service, validate_model_config
from app.src.mlflow_service import MLFlowClient
from app.src.time_service import get_last_hour, get_next_full_hour

load_dotenv()
logging = get_logger()


class InferenceService:
    def __init__(
        self,
        influx_connector: InfluxDBConnector,
        mqtt_client: MQTTClient,
        mlflow_client,
        sensor_source: SensorSource,
        config: dict,
    ):
        self.connection: InfluxDBConnector = influx_connector
        self.mqtt_client: MQTTClient = mqtt_client
        self.mlflow_client: MLFlowClient = mlflow_client
        self.sensor_source: SensorSource = sensor_source
        self.config: dict = config

    def initial_inference(self) -> None:
        logging.info("Initial inference started.")
        inputs_query: str = (
            InfluxQueryBuilder()
            .set_bucket(self.sensor_source.get_bucket())
            .set_range(self.config["start_time"], self.config["stop_time"])
            .set_topic(self.sensor_source.get_sensor())
            .set_fields(self.config["inputs"])
            .build()
        )
        input_data: pd.DataFrame = self.connection.query_dataframe(inputs_query)
        self.run_inference(input_data)
        logging.info(
            f"Initial inference complete for {self.config['start_time']} - {self.config['stop_time']}"
        )

    def hourly_inference(self) -> None:
        start_of_hour, end_of_hour = get_last_hour()
        logging.info(f"Inference of hour: {start_of_hour} - {end_of_hour}")

        inputs_query: str = (
            InfluxQueryBuilder()
            .set_bucket(self.sensor_source.get_bucket())
            .set_range(start_of_hour, end_of_hour)
            .set_topic(self.sensor_source.get_sensor())
            .set_fields(self.config["inputs"])
            .build()
        )
        input_data: pd.DataFrame = self.connection.query_dataframe(inputs_query)
        self.run_inference(input_data)
        logging.info(f"Inference complete for hour: {start_of_hour} - {end_of_hour}")

    def run_inference(self, inputs: pd.DataFrame) -> None:
        data_processor: DataProcessor = (
            DataProcessor(inputs)
            .to_hourly()
            .remove_nan()
            .calculate_w_a_difference(["NO", "NO2", "O3"])
            .align_dataframes_by_time()
        )

        model: BaseEstimator = self.mlflow_client.load_scikit_learn_model(
            f"models:/{self.config['model_name']}/{self.config['model_version']}"
        )

        prediction: np.ndarray = model.predict(data_processor.get_inputs())

        dataframe_predictions: pd.DataFrame = pd.DataFrame(prediction, columns=["NO2"])
        dataframe_predictions["timestamp"] = (
            data_processor.get_inputs().index.astype("int64") // 10**9
        )

        data: list[dict] = dataframe_predictions.to_dict(orient="records")
        for element in data:
            self.mqtt_client.publish_data(element, self.config["mqtt_topic"])
        # self.mqtt_client.stop() ## TODO: Wird hier direkt die verbindung beendet? Nochmal gegen checken mit joshkas client implementierung


if __name__ == "__main__":
    # Load models registry
    registry_config: dict = get_config("./models_registry.yaml")
    models = registry_config.get("models", [])

    if not models:
        logging.error("No models found in models_registry.yaml")
        exit(1)

    logging.info(f"Found {len(models)} model(s) in registry")

    # Validate all model configurations
    for model_config in models:
        validate_model_config(model_config)

    # Create inference services for all models
    services = []
    for model_config in models:
        service = create_inference_service(model_config)
        services.append((model_config["name"], service))

    next_full_hour: datetime = get_next_full_hour()
    logging.info(f"Starting hourly inference schedulers. Next run at: {next_full_hour}")

    # Create schedulers for each service
    schedulers = []
    for i, (model_name, service) in enumerate(services):
        # Use BackgroundScheduler for all but the last service
        if i < len(services) - 1:
            scheduler = BackgroundScheduler()
        else:
            # Use BlockingScheduler for the last service to keep main thread alive
            scheduler = BlockingScheduler()

        scheduler.add_job(
            service.hourly_inference,
            "interval",
            hours=1,
            next_run_time=next_full_hour,
            id=f"hourly_inference_{model_name}",
        )
        schedulers.append((model_name, scheduler))

    # Start all schedulers
    for model_name, scheduler in schedulers:
        logging.info(f"Starting scheduler for model: {model_name}")
        scheduler.start()

    # The last scheduler is blocking, so we'll only reach here on shutdown
    logging.info("All schedulers stopped")
