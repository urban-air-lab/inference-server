from datetime import datetime, timedelta, timezone
from typing import Tuple


def get_last_hour() -> Tuple[str, str]:
    now: datetime = datetime.now()
    now: datetime = now.replace(minute=0, second=0, microsecond=0)

    start_of_hour: datetime = now - timedelta(hours=1)
    start_of_hour_utc = start_of_hour.replace(tzinfo=timezone.utc)
    start_of_hour_iso_utc = start_of_hour_utc.isoformat().replace("+00:00", "Z")

    end_of_hour: datetime = now
    end_of_hour_utc = end_of_hour.replace(tzinfo=timezone.utc)
    end_of_hour_iso_utc = end_of_hour_utc.isoformat().replace("+00:00", "Z")
    return start_of_hour_iso_utc, end_of_hour_iso_utc

def get_next_full_hour() -> datetime:
    now = datetime.now()
    now = now.replace(minute=0, second=0, microsecond=0)
    return now + timedelta(hours=1, minutes=1)  # 1 minute extra to be assured last sensor date arrived