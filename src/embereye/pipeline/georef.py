"""
georef.py — Which point on the ground each pixel shows.

GeoRef uses only what a drone reports with each frame: its GPS position,
its altitude above sea level, its heading, the gimbal's pitch (and yaw),
and the camera's field of view. From those it rebuilds the camera, turns
each pixel into the direction that pixel looks, and follows that direction
down until it meets the ground.

This first version assumes the ground is flat: one level, `ground_up_m`
metres above the local datum (see geo.LocalFrame). On a hill that is wrong
by tens of metres (docs/ARCHITECTURE.md, GeoRef), so following the rays
down to the real terrain comes next.

Not handled yet:
- Gimbal roll. The sim has none, and a frame that reports any is refused
  rather than placed slightly wrong.
- Gimbal yaw is taken to be measured from the drone's heading, so the
  camera points at heading + gimbal yaw. The real drone's adapter must
  confirm that.

The camera maths (CameraModel, CameraPose, look_rotation) lives in
sim/camera.py. It is plain geometry with no MuJoCo in it.
"""

from __future__ import annotations

import math

import numpy as np

from embereye.geo import LocalFrame
from embereye.models.types import Frame
from embereye.sim.camera import CameraModel, CameraPose, look_rotation


def camera_from_frame(frame: Frame) -> CameraModel:
    """The camera's lens, from the frame's image size and field of view.

    Frames carry the HORIZONTAL field of view and CameraModel takes the
    vertical one. With square pixels both come from the same focal length.
    """
    if frame.fov_deg is None:
        raise ValueError("The frame has no field of view (fov_deg), so its pixels can't be placed on the ground.")
    height, width = frame.image.shape[:2]
    focal_px = (width / 2) / math.tan(math.radians(frame.fov_deg) / 2)
    fov_v_deg = 2 * math.degrees(math.atan((height / 2) / focal_px))
    return CameraModel(width=width, height=height, fov_v_deg=fov_v_deg)


class GeoRef:
    """Place a frame's pixels on flat ground.

    georef = GeoRef(scenario.local_frame)
    points = georef.ground_points(frame, xs, ys)  # (..., 3): east, north, up in metres
    """

    def __init__(self, local_frame: LocalFrame, ground_up_m: float = 0.0) -> None:
        self.local_frame = local_frame
        self.ground_up_m = ground_up_m  # height of the ground above the local datum

    def pose(self, frame: Frame) -> CameraPose:
        """Where the camera was and which way it pointed, from the frame's telemetry."""
        telemetry = frame.telemetry
        position = telemetry.position
        if position is None or position.alt_m is None:
            raise ValueError("The frame's telemetry has no GPS position with an altitude.")
        if telemetry.heading_deg is None or telemetry.gimbal_pitch is None:
            raise ValueError("The frame's telemetry needs heading_deg and gimbal_pitch.")
        if telemetry.gimbal_roll not in (None, 0.0):
            raise NotImplementedError(f"Gimbal roll ({telemetry.gimbal_roll} deg) is not handled yet.")

        heading_deg = (telemetry.heading_deg + (telemetry.gimbal_yaw or 0.0)) % 360
        pitch_deg = telemetry.gimbal_pitch
        east, north = self.local_frame.to_local(position.lat, position.lon)
        up = position.alt_m - self.local_frame.origin_alt_m

        return CameraPose(
            east_m=float(east),
            north_m=float(north),
            up_m=up,
            heading_deg=heading_deg,
            pitch_deg=pitch_deg,
            rotation=look_rotation(heading_deg, pitch_deg),
            lat=position.lat,
            lon=position.lon,
            alt_msl_m=position.alt_m,
            alt_agl_m=up - self.ground_up_m,
        )

    def ground_points(self, frame: Frame, x, y) -> np.ndarray:
        """The ground point each pixel (x, y) shows, as (east, north, up) in local metres.

        x and y are pixel coordinates (OpenCV convention, see sim/camera.py):
        single numbers, or arrays of the same shape. The result has that
        shape plus a last axis of 3. Pixels that look level or upwards (sky)
        never reach the ground and get NaN.
        """
        camera = camera_from_frame(frame)
        pose = self.pose(frame)
        rays = camera.pixel_rays(pose, x, y)

        # The ground is `height` below the camera and each ray drops
        # `descent` metres per metre travelled, so it meets the ground
        # height / descent metres out.
        height = pose.up_m - self.ground_up_m
        descent = -rays[..., 2]
        reaches_ground = (descent > 1e-9) & (height > 0)
        distance = np.where(reaches_ground, height / np.where(reaches_ground, descent, 1.0), np.nan)
        return pose.position + distance[..., None] * rays
