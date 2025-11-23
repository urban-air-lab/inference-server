import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from ual.data_processor import DataProcessor
from ual.get_config import get_config
from ual.influx import sensors
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource
import mlflow
from sklearn.base import BaseEstimator
from ual.mqtt.mqtt_client import MQTTClient

load_dotenv()

ual_source = SensorSource(bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET,
                          sensor=sensors.UALSensors.UAL_3)

run_config: dict = get_config("./run_config.yaml")
run_config["ual_bucket"] = ual_source.get_bucket()
run_config["ual_sensor"] = ual_source.get_sensor()

connection: InfluxDBConnector = InfluxDBConnector(os.getenv("INFLUX_URL"), os.getenv("INFLUX_TOKEN"),
                                                  os.getenv("INFLUX_ORG"))

inputs_query: str = InfluxQueryBuilder() \
    .set_bucket(ual_source.get_bucket()) \
    .set_range(run_config["start_time"], run_config["stop_time"]) \
    .set_topic(ual_source.get_sensor()) \
    .set_fields(run_config["inputs"]) \
    .build()
input_data: pd.DataFrame = connection.query_dataframe(inputs_query)

data_processor: DataProcessor = (DataProcessor(input_data)
                                 .to_hourly()
                                 .remove_nan()
                                 .calculate_w_a_difference(['NO', 'NO2', 'O3'])
                                 .align_dataframes_by_time())


os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_URL"))

model_name: str = "NO2_ual-3"
model_version: str = "1"
model: BaseEstimator = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")

prediction: np.ndarray = model.predict(data_processor.get_inputs())

dataframe_predictions = pd.DataFrame(prediction, columns=["NO"])
dataframe_predictions["timestamp"] = data_processor.get_inputs().index.astype('int64') // 10**9

data = dataframe_predictions.to_dict(orient='records')
mqtt_client = MQTTClient(os.getenv("MQTT_SERVER"), int(os.getenv("MQTT_PORT")), os.getenv("MQTT_USERNAME"), os.getenv("MQTT_PASSWORD"))
for element in data:
    mqtt_client.publish_data(element, "sensors/ual-hour-inference/ual-3")
mqtt_client.stop()

