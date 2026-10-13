"""
camera.py — The simulated drone camera: where it is, where it points, and
the flight path it follows.

Three ideas live here:

- CameraModel: the lens. Image size and field of view, plus the maths that
  links a point in the world to a pixel in the image (and back).
- CameraPose:  the exact position and orientation of the camera for one
  frame. In the sim this is known perfectly — it is the "answer key" for
  testing georeferencing.
- CameraPath:  a scripted flight. Ask it where the camera is at time t.
  OrbitPath circles a point while climbing and descending.

Angle conventions (same as models/types.py):
- heading: degrees clockwise from north. 0 = N, 90 = E.
- pitch:   0 = looking at the horizon, -90 = looking straight down.

Pixel convention (same as OpenCV): x counts right, y counts DOWN, and
(0, 0) is the centre of the top-left pixel.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


def look_rotation(heading_deg: float, pitch_deg: float) -> np.ndarray:
    """Orientation of a camera with the given heading and pitch (no roll).

    Returns a 3x3 array whose COLUMNS are the camera's own axes written in
    world (east, north, up) coordinates:

        column 0: the camera's right
        column 1: the camera's up
        column 2: the camera's back (it looks along MINUS this axis —
                  the convention MuJoCo and OpenGL use)
    """
    heading = math.radians(heading_deg)
    pitch   = math.radians(pitch_deg)

    forward = np.array([
        math.sin(heading) * math.cos(pitch),    # east
        math.cos(heading) * math.cos(pitch),    # north
        math.sin(pitch),                        # up (negative when looking down)
    ])
    # "Right" stays level because a gimbal keeps the horizon flat (no roll).
    right = np.array([math.cos(heading), -math.sin(heading), 0.0])
    back  = -forward
    up    = np.cross(back, right)

    return np.column_stack([right, up, back])


# ---------------------------------------------------------------------------
# Pose — where the camera is for one frame
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True, eq=False)
class CameraPose:
    """Exact camera position and orientation for one frame."""

    # Position in local metres (see geo.py). `up_m` is height above the
    # local datum, i.e. above the level the flat ground sits at.
    east_m:      float
    north_m:     float
    up_m:        float

    heading_deg: float          # 0 = N, 90 = E
    pitch_deg:   float          # 0 = horizon, -90 = straight down
    rotation:    np.ndarray     # 3x3, see look_rotation()

    # The same position the way a real drone would report it.
    lat:         float
    lon:         float
    alt_msl_m:   float          # altitude above sea level
    alt_agl_m:   float          # altitude above the ground directly below

    @property
    def position(self) -> np.ndarray:
        return np.array([self.east_m, self.north_m, self.up_m])


# ---------------------------------------------------------------------------
# Lens model
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CameraModel:
    """An ideal (distortion-free) camera: image size and field of view."""

    width:     int   = 640
    height:    int   = 480
    fov_v_deg: float = 50.0     # vertical field of view

    @property
    def focal_px(self) -> float:
        """Focal length measured in pixels.

        It is the distance from the lens to an imaginary screen on which
        one pixel is one unit wide. Half the image height sits at half the
        vertical field of view, which gives this formula.
        """
        return (self.height / 2) / math.tan(math.radians(self.fov_v_deg) / 2)

    @property
    def fov_h_deg(self) -> float:
        """Horizontal field of view (wider than vertical for a wide image)."""
        return 2 * math.degrees(math.atan((self.width / 2) / self.focal_px))

    @property
    def cx(self) -> float:
        """x of the image centre. Pixel centres are whole numbers, so the
        centre of a 640-wide image is at 319.5."""
        return (self.width - 1) / 2

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2

    def project(self, pose: CameraPose, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World points -> pixel coordinates.

        points_world: (N, 3) array of (east, north, up).
        Returns (pixels, in_front): pixels is (N, 2) of (x, y);
        in_front is True where the point is ahead of the camera. Pixels can
        fall outside the image — check against width/height if that matters.
        """
        points_world = np.atleast_2d(np.asarray(points_world, dtype=float))
        relative = points_world - pose.position

        # Re-express each point along the camera's own right/up/back axes.
        in_camera = relative @ pose.rotation
        depth = -in_camera[:, 2]            # distance in front of the lens
        in_front = depth > 1e-9

        safe_depth = np.where(in_front, depth, np.nan)
        x = self.cx + self.focal_px * in_camera[:, 0] / safe_depth
        y = self.cy - self.focal_px * in_camera[:, 1] / safe_depth   # minus: pixel y counts down
        return np.column_stack([x, y]), in_front

    def pixel_rays(self, pose: CameraPose, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Pixel coordinates -> the direction each pixel looks, in world axes.

        The exact reverse of project(). Returns an array shaped like x with
        an extra last axis of 3: (east, north, up) unit vectors.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        in_camera = np.stack([
            (x - self.cx) / self.focal_px,
            -(y - self.cy) / self.focal_px,
            -np.ones_like(x),
        ], axis=-1)
        in_world = in_camera @ pose.rotation.T
        return in_world / np.linalg.norm(in_world, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Flight paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PathPoint:
    """Where a scripted path puts the camera at one moment."""
    east_m:      float
    north_m:     float
    up_m:        float
    heading_deg: float
    pitch_deg:   float


class CameraPath(ABC):
    """A scripted flight: a rule that gives the camera pose at any time."""

    @abstractmethod
    def at(self, time_s: float) -> PathPoint:
        """Camera position and aim `time_s` seconds into the flight."""
        raise NotImplementedError


@dataclass
class OrbitPath(CameraPath):
    """Circle a point on the ground, always aiming at it, while the
    altitude rises and falls smoothly between two limits.

    Because the altitude changes, the camera's downward pitch changes too,
    so a single flight covers a range of viewing angles.
    """

    radius_m:          float = 350.0
    period_s:          float = 240.0    # time for one full lap
    altitude_min_m:    float = 120.0    # heights are `up_m`: above the local datum
    altitude_max_m:    float = 220.0
    altitude_period_s: float = 150.0    # time for one full up-and-down cycle
    center_east_m:     float = 0.0
    center_north_m:    float = 0.0
    target_up_m:       float = 0.0      # height of the point the camera aims at
    start_bearing_deg: float = 180.0    # where on the circle the flight starts, seen from the centre (180 = due south)
    clockwise:         bool  = True     # as seen from above

    def at(self, time_s: float) -> PathPoint:
        # Bearing from the centre out to the drone.
        laps = time_s / self.period_s
        direction = 1.0 if self.clockwise else -1.0
        bearing_deg = (self.start_bearing_deg + direction * 360.0 * laps) % 360.0
        bearing = math.radians(bearing_deg)

        east  = self.center_east_m  + self.radius_m * math.sin(bearing)
        north = self.center_north_m + self.radius_m * math.cos(bearing)

        # A sine wave between the two altitude limits, starting at the midpoint.
        mid   = (self.altitude_min_m + self.altitude_max_m) / 2
        swing = (self.altitude_max_m - self.altitude_min_m) / 2
        up = mid + swing * math.sin(2 * math.pi * time_s / self.altitude_period_s)

        # Aim at the centre: face back along the bearing, and tilt down by
        # the angle whose "opposite" is the height and "adjacent" the radius.
        heading_deg = (bearing_deg + 180.0) % 360.0
        pitch_deg = -math.degrees(math.atan2(up - self.target_up_m, self.radius_m))

        return PathPoint(east, north, up, heading_deg, pitch_deg)
