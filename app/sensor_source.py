from dataclasses import dataclass

from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.sensors import UALSensors, LUBWSensors


@dataclass()
class SensorSource:
    def __init__(self, bucket: InfluxBuckets, sensor: UALSensors | LUBWSensors):
        self.bucket: InfluxBuckets = bucket
        self.sensor: UALSensors | LUBWSensors = sensor
