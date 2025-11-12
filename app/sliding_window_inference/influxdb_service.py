import pandas as pd
from dotenv import load_dotenv
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.influx.sensors import SensorSource

load_dotenv()


class InfluxDBService:
    def __init__(self, url: str, token: str, org: str):
        self.connection = InfluxDBConnector(url, token, org)

    async def get_influx_data(self,
                              start: str,
                              stop: str,
                              source: SensorSource,
                              field: str) -> pd.DataFrame:
        query = InfluxQueryBuilder() \
            .set_bucket(source.get_bucket()) \
            .set_range(start, stop, True) \
            .set_topic(source.get_sensor()) \
            .set_fields(field) \
            .build()
        return self.connection.query_dataframe(query)
