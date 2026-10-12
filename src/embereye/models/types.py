"""
types.py — Shared domain types.

These are the objects that flow between the adapter, the pipeline, and
the dashboard. Every module speaks this vocabulary. Nothing in here knows
or cares what drone the data came from.

Using `@dataclass` (with slots for a small speed bump) keeps them simple
and self-documenting. Swap for Pydantic later if we need runtime validation
on the network layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enumerated categories
# ---------------------------------------------------------------------------
class DetectionClass(str, Enum):
    """What a detection is. Keep this list small and stable."""
    FIRE            = "fire"           # active flame
    HOTSPOT         = "hotspot"        # elevated heat, not clearly flame
    SMOKE           = "smoke"
    PERSON          = "person"
    VEHICLE         = "vehicle"
    STRUCTURE       = "structure"
    UNKNOWN         = "unknown"


class SensorType(str, Enum):
    RGB      = "rgb"
    THERMAL  = "thermal"


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class GeoPoint:
    """A latitude/longitude/altitude in WGS84. Altitude is meters above sea level."""
    lat: float
    lon: float
    alt_m: Optional[float] = None


@dataclass(slots=True)
class PixelBBox:
    """Bounding box in image pixel coordinates. (0,0) is top-left."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return max(0, (self.x2 - self.x1)) * max(0, (self.y2 - self.y1))


# ---------------------------------------------------------------------------
# Frames — what an adapter emits
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Telemetry:
    """Drone state at the moment a frame was captured. Every field optional
    because different platforms expose different things."""
    position:       Optional[GeoPoint] = None
    heading_deg:    Optional[float]    = None   # 0 = North, 90 = East
    gimbal_pitch:   Optional[float]    = None   # 0 = forward, -90 = straight down
    gimbal_yaw:     Optional[float]    = None
    battery_pct:    Optional[float]    = None


@dataclass(slots=True)
class Frame:
    """A single frame from a single sensor at a single moment.

    Adapters yield these. The pipeline consumes them. That's the whole
    contract between drone-land and pipeline-land.
    """
    ts:         datetime              # when this frame was captured
    sensor:     SensorType            # RGB or THERMAL
    image:      np.ndarray            # HxWx3 (RGB) or HxW (thermal, float32 in Celsius if radiometric)
    telemetry:  Telemetry             # drone state at capture time
    fov_deg:    Optional[float] = None  # horizontal field of view
    frame_id:   Optional[str]   = None  # unique id if the platform provides one


# ---------------------------------------------------------------------------
# Detections — what the pipeline produces from frames
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class Detection:
    """A single detected object in a single frame."""
    ts:             datetime
    detection_class: DetectionClass
    confidence:     float                        # 0.0 to 1.0
    bbox_px:        PixelBBox
    sensor:         SensorType
    world_position: Optional[GeoPoint] = None    # filled in by georef stage
    temperature_c:  Optional[float]    = None    # from thermal, if available
    frame_id:       Optional[str]      = None
    track_id:       Optional[str]      = None    # filled in by tracker stage


@dataclass(slots=True)
class Track:
    """A single entity followed across many frames."""
    track_id:       str
    detection_class: DetectionClass
    first_seen:     datetime
    last_seen:      datetime
    detections:     list[Detection] = field(default_factory=list)

    @property
    def latest_position(self) -> Optional[GeoPoint]:
        for det in reversed(self.detections):
            if det.world_position is not None:
                return det.world_position
        return None


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------
def now_utc() -> datetime:
    """One place for the current time so tests can monkey-patch it."""
    return datetime.now(timezone.utc)
