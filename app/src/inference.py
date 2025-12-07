import os
from datetime import datetime
import numpy as np
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv
from ual.data_processor import DataProcessor
from ual.get_config import get_config
from ual.logging import get_logger
from ual.influx import sensors
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
from sklearn.base import BaseEstimator
from ual.mqtt.mqtt_client import MQTTClient

from app.src.mlflow_service import MLFlowClient
from app.src.time_service import get_last_hour, get_next_full_hour

load_dotenv()
logging = get_logger()


class InferenceService:
    def __init__(self,
                 influx_connector: InfluxDBConnector,
                 mqtt_client: MQTTClient,
                 mlflow_client,
                 sensor_source: SensorSource,
                 config: dict):
        self.connection: InfluxDBConnector = influx_connector
        self.mqtt_client: MQTTClient = mqtt_client
        self.mlflow_client: MLFlowClient = mlflow_client
        self.sensor_source: SensorSource = sensor_source
        self.config: dict = config

    def initial_inference(self) -> None:
        logging.info("Initial inference started.")
        inputs_query: str = InfluxQueryBuilder() \
            .set_bucket(self.sensor_source.get_bucket()) \
            .set_range(self.config["start_time"], self.config["stop_time"]) \
            .set_topic(self.sensor_source.get_sensor()) \
            .set_fields(self.config["inputs"]) \
            .build()
        input_data: pd.DataFrame = self.connection.query_dataframe(inputs_query)
        self.run_inference(input_data)
        logging.info(f'Initial inference complete for {self.config["start_time"]} - {self.config["stop_time"]}')

    def hourly_inference(self) -> None:
        start_of_hour, end_of_hour = get_last_hour()
        logging.info(f"Inference of hour: {start_of_hour} - {end_of_hour}")

        inputs_query: str = InfluxQueryBuilder() \
            .set_bucket(self.sensor_source.get_bucket()) \
            .set_range(start_of_hour, end_of_hour) \
            .set_topic(self.sensor_source.get_sensor()) \
            .set_fields(self.config["inputs"]) \
            .build()
        input_data: pd.DataFrame = self.connection.query_dataframe(inputs_query)
        self.run_inference(input_data)
        logging.info(f'Inference complete for hour: {start_of_hour} - {end_of_hour}')

    def run_inference(self, inputs: pd.DataFrame) -> None:
        data_processor: DataProcessor = (DataProcessor(inputs)
                                         .to_hourly()
                                         .remove_nan()
                                         .calculate_w_a_difference(['NO', 'NO2', 'O3'])
                                         .align_dataframes_by_time())

        model: BaseEstimator = self.mlflow_client.load_scikit_learn_model(f"models:/{self.config['model_name']}/{self.config['model_version']}")

        prediction: np.ndarray = model.predict(data_processor.get_inputs())

        dataframe_predictions: pd.DataFrame = pd.DataFrame(prediction, columns=["NO2"])
        dataframe_predictions["timestamp"] = data_processor.get_inputs().index.astype('int64') // 10 ** 9

        data: list[dict] = dataframe_predictions.to_dict(orient='records')
        self.mqtt_client: MQTTClient = MQTTClient(os.getenv("MQTT_SERVER"), int(os.getenv("MQTT_PORT")),
                                             os.getenv("MQTT_USERNAME"), os.getenv("MQTT_PASSWORD"))
        for element in data:
            self.mqtt_client.publish_data(element, "sensors/ual-hour-inference/ual-3")
        self.mqtt_client.stop()


if __name__ == "__main__":
    inference_service:InferenceService = InferenceService(InfluxDBConnector(os.getenv("INFLUX_URL"), os.getenv("INFLUX_TOKEN"), os.getenv("INFLUX_ORG")),
                                                          MQTTClient(os.getenv("MQTT_SERVER"), int(os.getenv("MQTT_PORT")), os.getenv("MQTT_USERNAME"), os.getenv("MQTT_PASSWORD")),
                                                          MLFlowClient(os.getenv("MLFLOW_USERNAME"), os.getenv("MLFLOW_PASSWORD")),
                                                          SensorSource(bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET, sensor=sensors.UALSensors.UAL_3),
                                                          get_config("./run_config.yaml"))
    inference_service.initial_inference()

    next_full_hour: datetime = get_next_full_hour()
    logging.info(f"Inference at next full hour: {next_full_hour}")

    scheduler = BlockingScheduler()
    scheduler.add_job(inference_service.hourly_inference, 'interval', hours=1, next_run_time=next_full_hour)
    logging.info("Starting scheduler...")
    scheduler.start()


