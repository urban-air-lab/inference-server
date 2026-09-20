from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from app.src.service.time_service import get_next_full_hour


def test_get_next_full_hour():
    # Arrange
    fixed_now = datetime(2025, 6, 15, 14, 37, 22, 0, ZoneInfo("Europe/Berlin"))
    expected_next = datetime(2025, 6, 15, 15, 1, 0, 0, ZoneInfo("Europe/Berlin"))

    # Act
    with patch("app.src.service.time_service.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_now
        result = get_next_full_hour()

    # Assert
    assert result == expected_next
