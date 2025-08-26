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
        self.model: RandomForestRegressor | None = None

    async def start(self):
        if not self._interval_is_hourly():
            raise ValueError("interval must be a multiple of 60, since minutes are transformed to hours")
        logging.info("Starting sliding window loop.")
        while True:
            try:
                logging.info("Running training step.")
                await self.train_step()
                logging.info("Running inference step.")
                await self.inference_step()
            except Exception as e:
                logging.exception(f"Error during inference loop: {e}")
            logging.info("Finish loop, start again in 10s")
            await asyncio.sleep(10)

    async def train_step(self):
        logging.info(f"Training on interval: {self.interval_start_time} to {self.interval_end_time}")

        try:
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
        except Exception as e:
            logging.info(f"No train data in interval {self.interval_start_time} : {self.interval_end_time}")
            logging.info(f"skip interval")
            return

        if self._inputs_small_then_interval():
            progress: bool = await self._check_for_progress(self.inputs)
            if progress:
                logging.info("Waiting for missing input data...")
                self.inputs = await self._wait_for_data_completion(self.inputs)
            else:
                logging.info(f"No new data in train interval {self.interval_start_time} : {self.interval_end_time}")
                return

        processor = (DataProcessor(self.inputs, self.targets)
                     .to_hourly()
                     .remove_nan()
                     .calculate_w_a_difference()
                     .align_dataframes_by_time())

        logging.info("Training RandomForestRegressor...")
        self.model: RandomForestRegressor = RandomForestRegressor()
        self.model.fit(processor.get_inputs(),
                       processor.get_target(self.configuration["targets"]))  # TODO: Auslagern in Thread?

        logging.info("Model trained successfully.")

    async def inference_step(self):
        await self._move_time_window()
        logging.info(f"Inference on new interval: {self.interval_start_time} to {self.interval_end_time}")

        expected_hours = await self._get_interval_in_hours()
        published_hours = pd.DatetimeIndex([])

        try:
            current_inputs: pd.DataFrame = await self.connection.get_influx_data(
                self.interval_start_time,
                self.interval_end_time,
                self.ual_source,
                self.configuration["inputs"]
            )
            logging.info(f"Initial inference inputs: {len(current_inputs)} rows")
        except Exception as e:
            logging.info(f"No inference data in interval {self.interval_start_time} : {self.interval_end_time}")
            logging.info(f"skip interval")
            return

        while True:
            processor = (DataProcessor(current_inputs)
                         .to_hourly()
                         .remove_nan()
                         .calculate_w_a_difference()
                         .align_dataframes_by_time())
            current_inputs_hourly = processor.get_inputs()
            hours_ready_to_predict = current_inputs_hourly.index.difference(published_hours)

            if not hours_ready_to_predict.empty:
                hours_ready_to_predict = hours_ready_to_predict.intersection(current_inputs_hourly.index)

                if not hours_ready_to_predict.empty:
                    y_pred = self.model.predict(current_inputs_hourly.loc[hours_ready_to_predict])
                    results = self._create_results(y_pred, current_inputs_hourly.loc[hours_ready_to_predict].index)
                    logging.info(
                        f"Publishing {len(results)} predictions for {len(hours_ready_to_predict)} completed hour(s).")
                    self.mqtt.publish_dataframe(results,
                                                f'sensors/ual-hour-inference/{self.ual_source.get_bucket()}')
                    published_hours = published_hours.union(hours_ready_to_predict)

            if current_inputs_hourly.shape[0] >= expected_hours:
                logging.info("Inference window complete.")
                break

            progress: bool = await self._check_for_progress(current_inputs)
            if progress:
                current_inputs = await self._wait_for_next_full_hour(
                    current_inputs_hourly=current_inputs_hourly,
                    published_hours=published_hours,
                )
            else:
                logging.info(
                    f"No new data in this inference interval {self.interval_start_time} : {self.interval_end_time}")
                return

    async def _move_time_window(self):
        self.interval_start_time = self.interval_end_time
        self.interval_start_time, self.interval_end_time = self._create_time_interval(self.interval_start_time)

    async def _wait_for_data_completion(self, interval_inputs):
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

    async def _wait_for_next_full_hour(self, current_inputs_hourly: pd.DataFrame,
                                       published_hours: pd.DatetimeIndex) -> pd.DataFrame | None:

        while True:
            new_inputs = await self.connection.get_influx_data(
                self.interval_start_time, self.interval_end_time, self.ual_source, self.configuration["inputs"]
            )
            new_inputs_hourly = new_inputs.resample("h").mean()

            if len(new_inputs_hourly) > len(current_inputs_hourly):
                newly_full = new_inputs.index.difference(published_hours)
                if not newly_full.empty:
                    logging.info(f"Detected {len(newly_full)} newly completed hour(s): {list(newly_full)}")
                    return new_inputs

            logging.info("No newly completed hour yet, sleeping 60s...")
            await asyncio.sleep(60)

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

    async def _check_for_progress(self, current_inputs: pd.DataFrame) -> bool:
        logging.info("check for progress in sensor data")
        progress_counter = 0
        while progress_counter <= 2:
            new_inputs: pd.DataFrame = await self.connection.get_influx_data(
                self.interval_start_time,
                self.interval_end_time,
                self.ual_source,
                self.configuration["inputs"]
            )
            if not len(new_inputs) > len(current_inputs):
                await asyncio.sleep(60)
                logging.info("no progress so far")
                progress_counter += 1
            else:
                logging.info("progress in sensor data - proceeding")
                return True
        logging.info("no progress in sensor data")
        return False

    def _create_time_interval(self, start: str) -> (str, str):
        start_dt: datetime = datetime.strptime(start, "%Y-%m-%dT%H:%M:%SZ")
        end_dt: datetime = start_dt + timedelta(minutes=self.configuration["interval"])

        return start_dt.strftime("%Y-%m-%dT%H:%M:%SZ"), end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _create_results(self, predictions: np.ndarray, index: pd.DatetimeIndex) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame(data=predictions.flatten(), columns=[self.configuration["targets"][0]])
        df["timestamp"] = (index.astype("int64") // 1_000_000_000).astype(np.int64)
        return df

    async def _get_interval_in_hours(self) -> int:
        return int(self.configuration["interval"]) // 60

    def _inputs_small_then_interval(self) -> bool:
        return self.inputs.shape[0] < self.configuration["interval"]

    def _interval_is_hourly(self) -> bool:
        return self.configuration["interval"] % 60 == 0


if __name__ == "__main__":
    ual_source = SensorSource(bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET,
                              sensor=sensors.UALSensors.UAL_3)
    lubw_source = SensorSource(bucket=InfluxBuckets.LUBW_HOUR_BUCKET,
                               sensor=sensors.LUBWSensors.DEBW015)
    inference = SlidingWindowsInference(get_config("./run_config.yaml"), ual_source, lubw_source)
    asyncio.run(inference.start())
