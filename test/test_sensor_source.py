from ual.influx import sensors
from ual.influx.influx_buckets import InfluxBuckets

from app.sensor_source import SensorSource


def test_sensor_source():
    sensor_source = SensorSource(bucket=InfluxBuckets.UAL_MINUTE_CALIBRATION_BUCKET.value,
                                 sensor=sensors.UALSensors.UAL_3.value)
    assert sensor_source.bucket == "ual-minute-calibration"
    assert sensor_source.sensor == "ual-3"
