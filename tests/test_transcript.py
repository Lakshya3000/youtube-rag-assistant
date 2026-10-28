import pytest
from src.transcript import extract_video_id


def test_plain_video_id():
    assert extract_video_id("Gfr50f6ZBvo") == "Gfr50f6ZBvo"


def test_watch_url():
    assert extract_video_id("https://www.youtube.com/watch?v=Gfr50f6ZBvo&t=10") == "Gfr50f6ZBvo"


def test_short_url():
    assert extract_video_id("https://youtu.be/Gfr50f6ZBvo") == "Gfr50f6ZBvo"


def test_shorts_url():
    assert extract_video_id("https://www.youtube.com/shorts/Gfr50f6ZBvo") == "Gfr50f6ZBvo"


def test_invalid_url():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/video")
