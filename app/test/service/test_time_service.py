from datetime import datetime
from unittest.mock import patch

from app.src.service.time_service import get_next_full_hour


def test_get_next_full_hour():
    # Arrange
    fixed_now = datetime(2025, 6, 15, 14, 37, 22)
    expected_next = datetime(2025, 6, 15, 15, 1, 0)

    # Act
    with patch("app.src.service.time_service.datetime") as mock_datetime:
        mock_datetime.now.return_value = fixed_now
        result = get_next_full_hour()

    # Assert
    assert result == expected_next
