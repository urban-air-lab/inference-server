import os
from datetime import datetime, timedelta, timezone
from typing import Tuple

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
import mlflow
from sklearn.base import BaseEstimator
from ual.mqtt.mqtt_client import MQTTClient

load_dotenv()
logging = get_logger()

def get_last_hour() -> Tuple[str, str]:
    now: datetime = datetime.now()
    now: datetime = now.replace(minute=0, second=0, microsecond=0)

    start_of_hour: datetime = now - timedelta(hours=1)
    start_of_hour_utc = start_of_hour.replace(tzinfo=timezone.utc)
    start_of_hour_iso_utc = start_of_hour_utc.isoformat().replace("+00:00", "Z")

    end_of_hour: datetime = now
    end_of_hour_utc = end_of_hour.replace(tzinfo=timezone.utc)
    end_of_hour_iso_utc = end_of_hour_utc.isoformat().replace("+00:00", "Z")
    return start_of_hour_iso_utc, end_of_hour_iso_utc

def get_next_full_hour() -> datetime:
    now = datetime.now()
    now = now.replace(minute=0, second=0, microsecond=0)
    return now + timedelta(hours=1, minutes=1)  # 1 minute extra to be assured last sensor date arrived


class InferenceService:
    def __init__(self, influx_url: str,
                 influx_token: str,
                 influx_org: str,
                 sensor_source: SensorSource,
                 config: dict):
        self.connection: InfluxDBConnector = InfluxDBConnector(influx_url, influx_token,influx_org)
        self.sensor_source = sensor_source
        self.config = config

    def initial_inference(self):
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

    def hourly_inference(self):
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

        os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
        os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
        mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))
        model: BaseEstimator = mlflow.sklearn.load_model(f"models:/{run_config['model_name']}/{run_config['model_version']}")

        prediction: np.ndarray = model.predict(data_processor.get_inputs())

        dataframe_predictions: pd.DataFrame = pd.DataFrame(prediction, columns=["NO2"])
        dataframe_predictions["timestamp"] = data_processor.get_inputs().index.astype('int64') // 10 ** 9

        data: list[dict] = dataframe_predictions.to_dict(orient='records')
        mqtt_client: MQTTClient = MQTTClient(os.getenv("MQTT_SERVER"), int(os.getenv("MQTT_PORT")),
                                             os.getenv("MQTT_USERNAME"), os.getenv("MQTT_PASSWORD"))
        for element in data:
            mqtt_client.publish_data(element, "sensors/ual-hour-inference/ual-3")
        mqtt_client.stop()


if __name__ == "__main__":
    run_config: dict = get_config("./run_config.yaml")
    ual_source = SensorSource(bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET,
                              sensor=sensors.UALSensors.UAL_3)

    inference_service:InferenceService = InferenceService(os.getenv("INFLUX_URL"),
                                                          os.getenv("INFLUX_TOKEN"),
                                                          os.getenv("INFLUX_ORG"),
                                                          ual_source,
                                                          run_config)
    inference_service.initial_inference()

    next_full_hour: datetime = get_next_full_hour()
    logging.info(f"Inference at next full hour: {next_full_hour}")

    scheduler = BlockingScheduler()
    scheduler.add_job(inference_service.hourly_inference, 'interval', hours=1, next_run_time=next_full_hour)
    logging.info("Starting scheduler...")
    scheduler.start()


