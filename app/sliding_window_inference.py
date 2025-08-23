import os
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from ual.data_processor import DataProcessor
from ual.mqtt.mqtt_client import MQTTClient
from app.influxdb_service import InfluxDBService
from app.sensor_source import SensorSource


class SlidingWindowsInference:
    def __init__(self, configuration: dict, ual_source: SensorSource, lubw_source: SensorSource):
        self.configuration: dict = configuration

        self.ual_source = ual_source
        self.lubw_source = lubw_source
        self.connection = InfluxDBService()
        self.mqtt = MQTTClient(os.getenv("MQTT_SERVER"), int(os.getenv("MQTT_PORT")), os.getenv("MQTT_USERNAME"),
                               os.getenv("MQTT_PASSWORD"))

        self.interval_start_time: str | None = None
        self.interval_end_time: str | None = None
        self.inputs: pd.DataFrame | None = None
        self.targets: pd.DataFrame | None = None
        self.processor: DataProcessor | None = None
        self.model: RandomForestRegressor | None = None

    async def start(self):
        while True:
            try:
                await self.train_step()
                await self.inference_step()
            except Exception as e:
                print(f"Error during inference: {e}")
            await asyncio.sleep(self.configuration.get("loop_interval_seconds", 60))

    async def train_step(self):
        self._create_time_interval()
        self.inputs: pd.DataFrame = self.connection.get_inputs_by_dates(self.interval_start_time,
                                                                        self.interval_end_time,
                                                                        self.configuration)
        self.targets: pd.DataFrame = self.connection.get_targets_by_dates(self.interval_start_time,
                                                                          self.interval_end_time,
                                                                          self.configuration)
        if self.inputs != self.configuration["interval"]:
            self.inputs = await self._check_data_completion(self.inputs)
        if self.targets == self.configuration["interval"]:
            self.targets = await self._check_data_completion(self.targets)
        self.processor = (DataProcessor(self.inputs, self.targets)
                          .to_hourly()
                          .remove_nan()
                          .calculate_w_a_difference()
                          .align_dataframes_by_time())
        model: RandomForestRegressor = RandomForestRegressor()
        model.fit(self.processor.get_inputs(), self.processor.get_target(self.configuration["targets"]))

    async def _check_data_completion(self, interval_inputs):
        missing_inputs = self.configuration["interval"] - interval_inputs
        seconds_to_wait = missing_inputs * 60  # minute steps
        await asyncio.sleep(seconds_to_wait)
        interval_inputs: pd.DataFrame = self.connection.get_inputs_by_dates(self.interval_start_time,
                                                                            self.interval_end_time,
                                                                            self.configuration)
        return interval_inputs

    async def inference_step(self):
        self.interval_start_time = self.interval_end_time
        self._create_time_interval()
        current_inputs: pd.DataFrame = self.connection.get_inputs_by_dates(
            self.interval_start_time,
            self.interval_end_time,
            self.configuration
        )
        while True:
            new_inputs = await self._wait_for_next_input(current_inputs)

            self.processor = (DataProcessor(current_inputs, None)
                              .to_hourly()
                              .remove_nan()
                              .calculate_w_a_difference()
                              .align_dataframes_by_time())

            prediction: np.ndarray = self.model.predict(self.processor.get_inputs())
            results = self._create_results(self.processor, prediction, self.configuration)
            self.mqtt.publish_dataframe(results, f'sensors/ual-hour-inference/{self.configuration["ual_bucket"]}')

            current_inputs = new_inputs
            if current_inputs <= self.configuration["interval"]:
                break

    def _create_time_interval(self):
        self.interval_start_time: datetime = datetime.strptime(self.configuration["start_time"], "%Y-%m-%dT%H:%M:%SZ")
        self.interval_end_time: datetime = self.interval_start_time + timedelta(minutes=self.configuration["interval"])

    async def _wait_for_next_input(self, current_inputs: pd.DataFrame) -> pd.DataFrame:
        while True:
            # replace by mqtt event
            new_inputs = self.connection.get_inputs_by_dates(
                self.interval_start_time,
                self.interval_end_time,
                self.configuration
            )
            if len(new_inputs) > len(current_inputs):
                return new_inputs
            await asyncio.sleep(60)

    def _create_results(self, next_data_processor, predictions, run_config):
        df: pd.DataFrame = pd.DataFrame(data=predictions.flatten(), columns=[run_config["targets"][0]])
        df["timestamp"] = next_data_processor.get_target(run_config["targets"]).index.astype('int64') // 1_000_000_000
        return df
