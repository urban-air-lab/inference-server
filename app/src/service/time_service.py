from datetime import datetime, timedelta


def get_next_full_hour() -> datetime:
    now = datetime.now()
    now = now.replace(minute=0, second=0, microsecond=0)
    return now + timedelta(
        hours=1, minutes=1
    )  # 1 minute extra to be assured last sensor date arrived
