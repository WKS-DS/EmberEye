"""Tests for the shared domain types (embereye/models/types.py)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from embereye.models.types import Detection, DetectionClass, PixelBBox, SensorType, Telemetry


def test_bbox_from_polygon_contains_every_point():
    polygon = np.array([[10.2, 5.0], [30.0, 7.5], [25.6, 40.1], [12.0, 33.3]], dtype=np.float32)
    box = PixelBBox.from_polygon(polygon)

    assert (box.x1, box.y1, box.x2, box.y2) == (10, 5, 30, 41)
    assert box.x1 <= polygon[:, 0].min() and polygon[:, 0].max() <= box.x2
    assert box.y1 <= polygon[:, 1].min() and polygon[:, 1].max() <= box.y2


def test_bbox_from_polygon_on_whole_pixels_is_exact():
    box = PixelBBox.from_polygon([[0, 0], [4, 0], [4, 3], [0, 3]])
    assert (box.x1, box.y1, box.x2, box.y2) == (0, 0, 4, 3)
    assert box.area == 12


def test_bbox_from_empty_polygon_is_an_error():
    with pytest.raises(ValueError):
        PixelBBox.from_polygon(np.zeros((0, 2)))


def test_new_fields_are_optional_and_keep_old_calls_working():
    telemetry = Telemetry(heading_deg=90.0)
    assert telemetry.gimbal_roll is None

    polygon = np.array([[1.0, 1.0], [5.0, 1.0], [5.0, 4.0]])
    detection = Detection(
        ts=datetime(2026, 1, 1, tzinfo=timezone.utc),
        detection_class=DetectionClass.FIRE,
        confidence=0.9,
        bbox_px=PixelBBox.from_polygon(polygon),
        sensor=SensorType.RGB,
        polygon_px=polygon,
    )
    assert detection.track_id is None
    assert detection.polygon_px is polygon
