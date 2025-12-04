from app.src.inference import get_next_full_hour


def test_get_next_full_hour():
    time = get_next_full_hour()
    assert True