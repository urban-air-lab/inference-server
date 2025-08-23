from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.sensors import UALSensors, LUBWSensors


class SensorSource:
    def __init__(self, bucket: InfluxBuckets, sensor: UALSensors | LUBWSensors):
        self.bucket: InfluxBuckets = bucket
        self.sensor: UALSensors | LUBWSensors = sensor

    def get_bucket(self) -> str:
        return self.bucket.value

    def get_sensor(self) -> str:
        return self.sensor.value
