class SensorSourceStr:
    def __init__(self, bucket: str, sensor: str):
        self.bucket: str = bucket
        self.sensor: str = sensor

    def get_bucket(self) -> str:
        return self.bucket

    def get_sensor(self) -> str:
        return self.sensor