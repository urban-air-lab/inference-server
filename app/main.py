from dotenv import load_dotenv
from ual.influx import sensors
from ual.influx.Influx_db_connector import InfluxDBConnector
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.influx_query_builder import InfluxQueryBuilder
from ual.get_config import get_config

load_dotenv()

run_config = get_config("./run_config.yaml")
run_config["ual_bucket"] = InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value
run_config["ual_sensor"] = sensors.UALSensors.UAL_3.value
run_config["lubw_bucket"] = InfluxBuckets.LUBW_HOUR_BUCKET.value
run_config["lubw_sensor"] = sensors.LUBWSensors.DEBW015.value

connection = InfluxDBConnector()

inputs_query = InfluxQueryBuilder() \
    .set_bucket(run_config["ual_bucket"]) \
    .set_range(run_config["start_time"], run_config["stop_time"]) \
    .set_topic(run_config["ual_sensor"]) \
    .set_fields(run_config["inputs"]) \
    .build()
input_data = connection.query_dataframe(inputs_query)
print(input_data.index)
