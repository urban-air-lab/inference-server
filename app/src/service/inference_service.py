import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ual.data_processor import DataProcessor
from ual.get_config import logging
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
from ual.mqtt.mqtt_client import MQTTClient

from app.src.service.mlflow_service import MLFlowService
from app.src.service.time_service import get_last_hour


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
        self.mlflow_client: MLFlowService = mlflow_client
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
        logging.info("DEBUG: " + inputs_query)
        logging.info(f"query data from {self.sensor_source.get_bucket()}/{self.sensor_source.get_sensor()}")
        input_data: pd.DataFrame = self.connection.query_dataframe(inputs_query)
        self._run_inference(input_data)
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
        logging.info(f"query data from {self.sensor_source.get_bucket()}/{self.sensor_source.get_sensor()}")
        input_data: pd.DataFrame = self.connection.query_dataframe(inputs_query)
        self._run_inference(input_data)
        logging.info(f"Inference complete for hour: {start_of_hour} - {end_of_hour}")

    def _run_inference(self, inputs: pd.DataFrame) -> None:
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

        dataframe_predictions: pd.DataFrame = pd.DataFrame(prediction, columns=self.config["targets"])
        dataframe_predictions["timestamp"] = (
            data_processor.get_inputs().index.astype("int64") // 10**9
        )

        data: list[dict] = dataframe_predictions.to_dict(orient="records")
        for element in data:
            self.mqtt_client.publish_data(element, self.config["mqtt_topic"])
