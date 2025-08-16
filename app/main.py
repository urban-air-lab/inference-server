import numpy as np
from dotenv import load_dotenv
from ual.influx import sensors
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.get_config import get_config
from ual.data_processor import DataProcessor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from itertools import pairwise

from ual.mqtt.mqtt_client import MQTTClient

load_dotenv()


def main():
    run_config: dict = get_config("./run_config.yaml")
    run_config["ual_bucket"] = InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value
    run_config["ual_sensor"] = sensors.UALSensors.UAL_3.value
    run_config["lubw_bucket"] = InfluxBuckets.LUBW_HOUR_BUCKET.value
    run_config["lubw_sensor"] = sensors.LUBWSensors.DEBW015.value

    connection = InfluxDBConnector()

    bucket_timestamps: pd.DatetimeIndex = get_timestamps_of_bucket(connection, run_config)
    interval_length: int = 10000
    intervals: list[pd.DatetimeIndex] = slice_datetime_index(bucket_timestamps, interval_length)
    ISO_format = "%Y-%m-%dT%H:%M:%SZ"

    for this_step, next_step in pairwise(intervals):
        if len(this_step) != interval_length or len(next_step) != interval_length:
            diff1: int = interval_length - len(this_step)
            diff2: int = interval_length - len(next_step)
            diff: int = max(diff1, diff2)
            # wait for diff time
            # schedular now + diff minutes

        this_inputs: pd.DataFrame = get_inputs_by_dates(connection, this_step[0].strftime(ISO_format),
                                                        this_step[-1].strftime(ISO_format), run_config)
        this_targets: pd.DataFrame = get_targets_by_dates(connection, this_step[0].strftime(ISO_format),
                                                          this_step[-1].strftime(ISO_format), run_config)

        this_data_processor: DataProcessor = (DataProcessor(this_inputs, this_targets)
                                              .to_hourly()
                                              .remove_nan()
                                              .calculate_w_a_difference()
                                              .align_dataframes_by_time())

        model: RandomForestRegressor = RandomForestRegressor()
        model.fit(this_data_processor.get_inputs(), this_data_processor.get_target(run_config["targets"]))

        next_inputs: pd.DataFrame = get_inputs_by_dates(connection, next_step[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                        next_step[-1].strftime("%Y-%m-%dT%H:%M:%SZ"), run_config)
        next_targets: pd.DataFrame = get_targets_by_dates(connection, next_step[0].strftime(ISO_format),
                                                          next_step[-1].strftime(ISO_format), run_config)

        next_data_processor: DataProcessor = (DataProcessor(next_inputs, next_targets)
                                              .to_hourly()
                                              .remove_nan()
                                              .calculate_w_a_difference()
                                              .align_dataframes_by_time())

        predictions: np.ndarray = model.predict(next_data_processor.get_inputs())
        df: pd.DataFrame = pd.DataFrame(data=predictions.flatten(), columns=[run_config["targets"][0]])
        df["timestamp"] = next_data_processor.get_target(run_config["targets"]).index.astype('int64') // 1_000_000_000

        print(df)

        mqtt_client = MQTTClient()
        mqtt_client.publish_dataframe(df, f'sensors/ual-hour-inference/{run_config["ual_bucket"]}')


def get_timestamps_of_bucket(connection: InfluxDBConnector, run_config: dict) -> pd.DatetimeIndex:
    timestamp_query = InfluxQueryBuilder() \
        .set_bucket(run_config["ual_bucket"]) \
        .set_range_to_start_0() \
        .set_topic(run_config["ual_sensor"]) \
        .set_fields(run_config["inputs"]) \
        .build()
    return connection.query_dataframe(timestamp_query).index


def get_inputs_by_dates(connection: InfluxDBConnector, start: str, stop: str, run_config: dict) -> pd.DataFrame:
    return get_influx_data_by_dates(connection, start, stop, run_config["ual_bucket"], run_config["ual_sensor"],
                                    run_config["inputs"])


def get_targets_by_dates(connection: InfluxDBConnector, start: str, stop: str, run_config: dict) -> pd.DataFrame:
    return get_influx_data_by_dates(connection, start, stop, run_config["lubw_bucket"], run_config["lubw_sensor"],
                                    run_config["targets"])


def get_influx_data_by_dates(connection: InfluxDBConnector,
                             start: str,
                             stop: str,
                             bucket: str,
                             topic: str,
                             field: str) -> pd.DataFrame:
    query = InfluxQueryBuilder() \
        .set_bucket(bucket) \
        .set_range(start, stop, True) \
        .set_topic(topic) \
        .set_fields(field) \
        .build()
    return connection.query_dataframe(query)


def slice_datetime_index(dt_index: pd.DatetimeIndex, chunk_size: int) -> list:
    return [dt_index[i:i + chunk_size] for i in range(0, len(dt_index), chunk_size)]


if __name__ == "__main__":
    main()
