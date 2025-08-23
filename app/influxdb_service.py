import os
import pandas as pd
from dotenv import load_dotenv
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder

load_dotenv()


class InfluxDBService:
    def __init__(self):
        self.connection = InfluxDBConnector(os.getenv("INFLUX_URL"), os.getenv("INFLUX_TOKEN"), os.getenv("INFLUX_ORG"))

    def get_inputs_by_dates(self, start: str, stop: str, run_config: dict) -> pd.DataFrame:
        return self._get_influx_data_by_dates(start, stop, run_config["ual_bucket"], run_config["ual_sensor"],
                                              run_config["inputs"])

    def get_targets_by_dates(self, start: str, stop: str, run_config: dict) -> pd.DataFrame:
        return self._get_influx_data_by_dates(start, stop, run_config["lubw_bucket"], run_config["lubw_sensor"],
                                              run_config["targets"])

    def _get_influx_data_by_dates(self,
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
        return self.connection.query_dataframe(query)
