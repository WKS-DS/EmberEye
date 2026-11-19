"""
georef.py — Which point on the ground each pixel shows.

GeoRef uses only what a drone reports with each frame: its GPS position,
its altitude above sea level, its heading, the gimbal's pitch (and yaw),
and the camera's field of view. From those it rebuilds the camera, turns
each pixel into the direction that pixel looks, and follows that direction
down until it meets the ground.

The ground is either flat, at one level `ground_up_m` metres above the
local datum (see geo.LocalFrame), or terrain: a prediction.terrain.Terrain
elevation grid laid out on a geo.GroundGrid. With terrain, each point on
the ground takes the elevation of its cell, so the ground is made of
flat-topped cells, exactly like the simulator's tiles. Beyond the grid it
is the lowest ground level, as in the simulator.

Not handled yet:
- Gimbal roll. The sim has none, and a frame that reports any is refused
  rather than placed slightly wrong.
- Gimbal yaw is taken to be measured from the drone's heading, so the
  camera points at heading + gimbal yaw. The real drone's adapter must
  confirm that.
- Pixels that look level or upwards get NaN, even if they show a hillside
  higher than the camera. That only happens when the drone flies lower
  than the terrain around it; every sim flight path stays above it.
- A real elevation file (USGS DEM) will need smoothing between cells
  (bilinear interpolation) instead of flat-topped cells.

The camera maths (CameraModel, CameraPose, look_rotation) lives in
sim/camera.py. It is plain geometry with no MuJoCo in it.
"""

from __future__ import annotations

import math

import numpy as np

from vigil.geo import GroundGrid, LocalFrame
from vigil.models.types import Frame
from vigil.prediction.terrain import Terrain
from vigil.sim.camera import CameraModel, CameraPose, look_rotation

# A ray that hits the side of a cell is placed this far inside that cell, so
# looking the point up on the grid finds the cell it hit, not its neighbour.
SIDE_HIT_INSET_M = 1e-6


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
    """Place a frame's pixels on the ground: flat ground, or terrain.

    georef = GeoRef(scenario.local_frame)  # flat ground at up = 0
    georef = GeoRef(scenario.local_frame, terrain=scenario.terrain, grid=scenario.grid)
    points = georef.ground_points(frame, xs, ys)  # (..., 3): east, north, up in metres
    """

    def __init__(
        self,
        local_frame: LocalFrame,
        ground_up_m: float = 0.0,
        terrain: Terrain | None = None,
        grid: GroundGrid | None = None,
    ) -> None:
        if (terrain is None) != (grid is None):
            raise ValueError("Give terrain and grid together: the grid says where the terrain's cells are.")
        if terrain is not None:
            if terrain.shape != grid.shape or terrain.cell_size_m != grid.cell_size_m:
                raise ValueError("The terrain and the grid need the same shape and cell size.")
            if ground_up_m != 0.0:
                raise ValueError("Give either ground_up_m (flat ground) or terrain, not both.")
        self.local_frame = local_frame
        self.ground_up_m = ground_up_m  # height of flat ground above the local datum
        self.terrain = terrain
        self.grid = grid

    def ground_up_at(self, east_m, north_m) -> np.ndarray:
        """Height of the ground at these points, in metres above the local datum."""
        if self.terrain is None:
            return np.full(np.shape(east_m), self.ground_up_m)
        rows, cols, on_grid = self.grid.cells_at(east_m, north_m)
        elevation = self.terrain.elevation
        # The clip only keeps the lookup inside the array; off-grid points take the lowest level.
        on_grid_elevation = elevation[np.clip(rows, 0, self.grid.rows - 1), np.clip(cols, 0, self.grid.cols - 1)]
        return np.where(on_grid, on_grid_elevation, elevation.min())

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
            alt_agl_m=up - float(self.ground_up_at(east, north)),
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
        distance = self._distance_to_ground(pose.position, rays.reshape(-1, 3)).reshape(rays.shape[:-1])
        return pose.position + distance[..., None] * rays

    def _distance_to_ground(self, start: np.ndarray, rays: np.ndarray) -> np.ndarray:
        """How far each ray (N, 3) travels from `start` before it first meets the ground.

        NaN for rays that never do: rays that look level or upwards, and every
        ray if the camera is underground.
        """
        distance = np.full(len(rays), np.nan)
        if start[2] <= self.ground_up_at(start[0], start[1]):
            return distance
        down = -rays[:, 2] > 1e-9
        if self.terrain is None:
            # Each ray drops -ray[2] metres per metre travelled, so it covers
            # the camera's height above the ground in height / drop metres.
            distance[down] = (start[2] - self.ground_up_m) / -rays[down, 2]
        else:
            distance[down] = self._distance_to_terrain(start, rays[down])
        return distance

    def _distance_to_terrain(self, start: np.ndarray, rays: np.ndarray) -> np.ndarray:
        """Like _distance_to_ground, for rays (N, 3) that all go down, over terrain.

        Each ray is walked across the grid one cell at a time, from the cell
        edge where it enters a cell to the edge where it leaves. The first
        cell where it has dropped below the cell's top is the one it hits:
        on the top, or on the side if it entered the cell already below the
        top. This is exact for flat-topped cells; stepping a fixed distance
        along the ray is not, because a ray can dip below the corner of a
        cell for much less than any step.
        """
        grid, elevation = self.grid, self.terrain.elevation
        cell = grid.cell_size_m
        drop = -rays[:, 2]  # metres down per metre along the ray

        # The ray is in the air until it passes the highest ground level, and
        # on or under the ground once it reaches the lowest. Off the grid the
        # ground IS the lowest level, so a ray that meets no cell lands there.
        reaches_highest = np.maximum(start[2] - elevation.max(), 0.0) / drop
        reaches_lowest = (start[2] - elevation.min()) / drop
        enters_grid, leaves_grid = self._span_over_grid(start, rays)
        walk_from = np.maximum(reaches_highest, enters_grid)
        walk_to = np.minimum(reaches_lowest, leaves_grid)
        result = reaches_lowest.copy()

        # The cell each walk starts in. The point is taken a hair along the
        # ray, so a walk that starts exactly on a cell edge starts in the
        # cell it is entering.
        walking = walk_from < walk_to
        first = start + (np.where(walking, walk_from, 0.0) + 1e-9)[:, None] * rays
        row, col, _ = grid.cells_at(first[:, 0], first[:, 1])
        row, col = np.clip(row, 0, grid.rows - 1), np.clip(col, 0, grid.cols - 1)

        # Distance along the ray to the next column edge and the next row
        # edge, and between one edge and the next. Rows count southward.
        east_step = np.where(rays[:, 0] > 0, 1, -1)
        row_step = np.where(rays[:, 1] > 0, -1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            next_east = -grid.width_m / 2 + (col + (rays[:, 0] > 0)) * cell
            next_north = grid.height_m / 2 - (row + (rays[:, 1] < 0)) * cell
            to_col_edge = np.where(rays[:, 0] != 0, (next_east - start[0]) / rays[:, 0], np.inf)
            to_row_edge = np.where(rays[:, 1] != 0, (next_north - start[1]) / rays[:, 1], np.inf)
            col_spacing = np.where(rays[:, 0] != 0, cell / np.abs(rays[:, 0]), np.inf)
            row_spacing = np.where(rays[:, 1] != 0, cell / np.abs(rays[:, 1]), np.inf)

        entered_cell_at = walk_from.copy()
        while walking.any():
            i = np.flatnonzero(walking)
            leaves_cell_at = np.minimum(np.minimum(to_col_edge[i], to_row_edge[i]), walk_to[i])
            top = elevation[row[i], col[i]]
            below_top = start[2] - drop[i] * leaves_cell_at <= top

            # Hits: the top of the cell, or its side if the ray came in below the top.
            hit = i[below_top]
            meets_top_at = (start[2] - top[below_top]) / drop[hit]
            result[hit] = np.maximum(meets_top_at, entered_cell_at[hit] + SIDE_HIT_INSET_M)
            walking[hit] = False

            # Rays that reach the end of their walk without a hit land on the lowest level.
            walking[i[~below_top & (leaves_cell_at >= walk_to[i])]] = False

            # The rest move into the next cell, across whichever edge comes first.
            moving = i[walking[i]]
            across_col = to_col_edge[moving] <= to_row_edge[moving]
            c, r = moving[across_col], moving[~across_col]
            entered_cell_at[c] = to_col_edge[c]
            col[c] += east_step[c]
            to_col_edge[c] += col_spacing[c]
            entered_cell_at[r] = to_row_edge[r]
            row[r] += row_step[r]
            to_row_edge[r] += row_spacing[r]
            # Rounding can carry a ray past the grid's edge just before walk_to.
            off_grid = (col[moving] < 0) | (col[moving] >= grid.cols) | (row[moving] < 0) | (row[moving] >= grid.rows)
            walking[moving[off_grid]] = False
        return result

    def _span_over_grid(self, start: np.ndarray, rays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distances along each ray where, seen from above, it enters and leaves the grid.

        If it never crosses the grid, the first is not smaller than the second.
        """
        half = np.array([self.grid.width_m / 2, self.grid.height_m / 2])
        with np.errstate(divide="ignore", invalid="ignore"):
            to_low_edge = (-half - start[:2]) / rays[:, :2]
            to_high_edge = (half - start[:2]) / rays[:, :2]
        enters = np.minimum(to_low_edge, to_high_edge).max(axis=1)
        leaves = np.maximum(to_low_edge, to_high_edge).min(axis=1)
        return enters, leaves
