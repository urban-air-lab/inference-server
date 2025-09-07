from ual.get_config import get_config
from ual.influx.influx_buckets import InfluxBuckets
from ual.influx.sensors import UALSensors, LUBWSensors, SensorSource
from app.sliding_window_inference import SlidingWindowsInference


def test_sliding_windows_inference_init():
    ual_sensor_source = SensorSource(InfluxBuckets.TEST_BUCKET, UALSensors.UAL_1)
    lubw_sensor_source = SensorSource(InfluxBuckets.TEST_BUCKET, LUBWSensors.DEBW015)
    sliding_windows_inference = SlidingWindowsInference(get_config("./resources/run_config.yaml"),
                                                        ual_sensor_source,
                                                        lubw_sensor_source)
    assert sliding_windows_inference.interval_start_time is "2024-11-16T00:00:00Z"
    assert sliding_windows_inference.interval_end_time is "2024-11-23T12:00:00Z"


def test_interval_is_hourly_correct():
    ual_sensor_source = SensorSource(InfluxBuckets.TEST_BUCKET, UALSensors.UAL_1)
    lubw_sensor_source = SensorSource(InfluxBuckets.TEST_BUCKET, LUBWSensors.DEBW015)
    sliding_windows_inference = SlidingWindowsInference(get_config("./resources/interval_config.yaml"),
                                                        ual_sensor_source,
                                                        lubw_sensor_source)
    assert sliding_windows_inference._interval_is_hourly() is True


def test_interval_is_hourly_wrong():
    ual_sensor_source = SensorSource(InfluxBuckets.TEST_BUCKET, UALSensors.UAL_1)
    lubw_sensor_source = SensorSource(InfluxBuckets.TEST_BUCKET, LUBWSensors.DEBW015)
    sliding_windows_inference = SlidingWindowsInference(get_config("./resources/wrong_interval_config.yaml"),
                                                        ual_sensor_source,
                                                        lubw_sensor_source)
    assert sliding_windows_inference._interval_is_hourly() is False
