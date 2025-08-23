import logging
import os
import sys
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from ual.data_processor import DataProcessor
from ual.get_config import get_config
from ual.influx import sensors
from ual.influx.influx_buckets import InfluxBuckets
from ual.mqtt.mqtt_client import MQTTClient
from app.influxdb_service import InfluxDBService
from app.sensor_source import SensorSource

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)


class SlidingWindowsInference:
    def __init__(self, configuration: dict, ual_source: SensorSource, lubw_source: SensorSource):
        self.configuration: dict = configuration

        self.ual_source = ual_source
        self.lubw_source = lubw_source
        self.connection = InfluxDBService(os.getenv("INFLUX_URL"), os.getenv("INFLUX_TOKEN"), os.getenv("INFLUX_ORG"))
        self.mqtt = MQTTClient(os.getenv("MQTT_SERVER"), int(os.getenv("MQTT_PORT")), os.getenv("MQTT_USERNAME"),
                               os.getenv("MQTT_PASSWORD"))

        self.interval_start_time, self.interval_end_time = self._create_time_interval(self.configuration["start_time"])

        self.inputs: pd.DataFrame | None = None
        self.targets: pd.DataFrame | None = None
        self.processor: DataProcessor | None = None
        self.model: RandomForestRegressor | None = None

    async def start(self):
        logging.info("Starting sliding window loop.")
        while True:
            try:
                logging.info("Running training step.")
                await self.train_step()
                logging.info("Running inference step.")
                await self.inference_step()
            except Exception as e:
                logging.exception(f"Error during inference loop: {e}")
            await asyncio.sleep(self.configuration.get("loop_interval_seconds", 60))

    async def train_step(self):
        logging.info(f"Training on interval: {self.interval_start_time} to {self.interval_end_time}")

        self.inputs: pd.DataFrame = await self.connection.get_influx_data(
            self.interval_start_time,
            self.interval_end_time,
            self.ual_source,
            self.configuration["inputs"]
        )
        logging.info(f"Loaded {len(self.inputs)} input rows.")

        self.targets: pd.DataFrame = await self.connection.get_influx_data(
            self.interval_start_time,
            self.interval_end_time,
            self.lubw_source,
            self.configuration["targets"]
        )
        logging.info(f"Loaded {len(self.targets)} target rows.")

        if self.inputs.shape[0] < self.configuration["interval"]:
            logging.info("Waiting for missing input data...")
            self.inputs = await self._check_data_completion(self.inputs)

        self.processor = (DataProcessor(self.inputs, self.targets)
                          .to_hourly()
                          .remove_nan()
                          .calculate_w_a_difference()
                          .align_dataframes_by_time())

        logging.info("Training RandomForestRegressor...")
        model: RandomForestRegressor = RandomForestRegressor()
        model.fit(self.processor.get_inputs(),
                  self.processor.get_target(self.configuration["targets"]))  # TODO: Auslagern in Thread?
        self.model = model
        logging.info("Model trained successfully.")

    async def _check_data_completion(self, interval_inputs):
        missing_inputs = self.configuration["interval"] - len(interval_inputs)
        seconds_to_wait = missing_inputs * 60  # minute steps
        logging.warning(f"Waiting {seconds_to_wait}s for {missing_inputs} missing input rows.")
        await asyncio.sleep(seconds_to_wait)

        interval_inputs: pd.DataFrame = await self.connection.get_influx_data(
            self.interval_start_time,
            self.interval_end_time,
            self.ual_source,
            self.configuration["inputs"]
        )
        logging.info(f"Re-checked input rows: now {len(interval_inputs)} rows.")
        return interval_inputs

    async def inference_step(self):
        self.interval_start_time = self.interval_end_time
        self.interval_start_time, self.interval_end_time = self._create_time_interval(self.interval_start_time)
        logging.info(f"Inference on new interval: {self.interval_start_time} to {self.interval_end_time}")

        current_inputs: pd.DataFrame = await self.connection.get_influx_data(
            self.interval_start_time,
            self.interval_end_time,
            self.ual_source,
            self.configuration["inputs"]
        )
        logging.info(f"Initial inference inputs: {len(current_inputs)} rows")
        await self._inference(current_inputs)

        # while True:
        #     new_inputs = await self._wait_for_next_input(current_inputs)
        #
        #     await self._inference(current_inputs)
        #
        #     current_inputs = new_inputs
        #
        if current_inputs.shape[0] >= self.configuration["interval"]:
            logging.info("Inference window complete.")

    async def _inference(self, current_inputs):
        self.processor = (DataProcessor(current_inputs)
                          .to_hourly()
                          .remove_nan()
                          .calculate_w_a_difference()
                          .align_dataframes_by_time())
        prediction: np.ndarray = self.model.predict(self.processor.get_inputs())
        results = self._create_results(prediction, self.processor.get_inputs().index)
        self.mqtt.publish_dataframe(results, f'sensors/ual-hour-inference/test')
        logging.info(f"Publishing {len(results)} predictions.")

    def _create_time_interval(self, start: str) -> (str, str):
        start_dt: datetime = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
        end_dt: datetime = start_dt + timedelta(minutes=self.configuration["interval"])

        return start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    async def _wait_for_next_input(self, current_inputs: pd.DataFrame) -> pd.DataFrame:
        while True:
            new_inputs = await self.connection.get_influx_data(
                self.interval_start_time,
                self.interval_end_time,
                self.ual_source,
                self.configuration["inputs"]
            )
            if len(new_inputs) > len(current_inputs):
                logging.info(f"New input row detected. Total rows: {len(new_inputs)}")
                return new_inputs
            logging.info("No new data yet, sleeping...")
            await asyncio.sleep(60)

    def _create_results(self, predictions: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame(data=predictions.flatten(), columns=[self.configuration["targets"][0]])
        df["timestamp"] = (index.astype("int64") // 1_000_000_000).astype(np.int64)
        return df


if __name__ == "__main__":
    ual_source = SensorSource(bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET,
                              sensor=sensors.UALSensors.UAL_3)
    lubw_source = SensorSource(bucket=InfluxBuckets.LUBW_HOUR_BUCKET,
                               sensor=sensors.LUBWSensors.DEBW015)
    inference = SlidingWindowsInference(get_config("./run_config.yaml"), ual_source, lubw_source)
    asyncio.run(inference.start())
