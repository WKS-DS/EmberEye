"""Tests for GeoRef on flat ground and on terrain (vigil/pipeline/georef.py).

GeoRef only sees what a drone reports, so these tests build Frames the way
an adapter would. The last test checks the result against the ground cells
the simulator actually drew.
"""

import asyncio
import math
from datetime import datetime, timezone

import numpy as np
import pytest

from vigil.geo import GroundGrid, LocalFrame
from vigil.models.types import Frame, GeoPoint, SensorType, Telemetry
from vigil.pipeline.georef import GeoRef, camera_from_frame
from vigil.prediction.terrain import Terrain
from vigil.sim.camera import CameraModel, OrbitPath, look_rotation

LOCAL = LocalFrame(origin_lat=47.6588, origin_lon=-117.4260, origin_alt_m=600.0)
CAMERA = CameraModel(width=320, height=240, fov_v_deg=50.0)


def block_terrain() -> tuple[Terrain, GroundGrid]:
    """3 x 3 cells of 10 m on flat ground at 0 m, with a 10 m block in the middle cell.

    The grid is centred on the local origin, so the block spans -5 to 5 m east and north.
    """
    elevation = np.zeros((3, 3))
    elevation[1, 1] = 10.0
    return Terrain(elevation, cell_size_m=10.0), GroundGrid(rows=3, cols=3, cell_size_m=10.0)


def make_frame(east_m, north_m, up_m, heading_deg, pitch_deg, **telemetry) -> Frame:
    """A Frame as an adapter would report it, for a camera at this local position."""
    lat, lon = LOCAL.to_gps(east_m, north_m)
    return Frame(
        ts=datetime(2026, 8, 1, tzinfo=timezone.utc),
        sensor=SensorType.RGB,
        image=np.zeros((CAMERA.height, CAMERA.width, 3), dtype=np.uint8),
        telemetry=Telemetry(
            position=GeoPoint(lat=float(lat), lon=float(lon), alt_m=LOCAL.origin_alt_m + up_m),
            heading_deg=heading_deg,
            gimbal_pitch=pitch_deg,
            **telemetry,
        ),
        fov_deg=CAMERA.fov_h_deg,
    )


def test_camera_from_frame_rebuilds_the_lens():
    camera = camera_from_frame(make_frame(0, 0, 100, 0, -45))
    assert (camera.width, camera.height) == (CAMERA.width, CAMERA.height)
    assert camera.fov_v_deg == pytest.approx(CAMERA.fov_v_deg)


def test_pose_comes_from_telemetry():
    pose = GeoRef(LOCAL, ground_up_m=5.0).pose(make_frame(40.0, -25.0, 120.0, 217.0, -60.0))
    assert np.allclose(pose.position, [40.0, -25.0, 120.0], atol=1e-6)
    assert np.allclose(pose.rotation, look_rotation(217.0, -60.0))
    assert pose.alt_agl_m == pytest.approx(115.0)


@pytest.mark.parametrize(
    "heading_deg, pitch_deg, ground_up_m",
    [(0.0, -90.0, 0.0), (90.0, -35.0, 0.0), (217.0, -60.0, 12.5), (330.0, -20.0, -8.0)],
)
def test_ground_points_round_trip(heading_deg, pitch_deg, ground_up_m):
    """Put points on the ground around where the camera aims, find the
    pixels that show them, and check GeoRef sends those pixels back."""
    georef = GeoRef(LOCAL, ground_up_m=ground_up_m)
    frame = make_frame(40.0, -25.0, 120.0, heading_deg, pitch_deg)
    pose = georef.pose(frame)

    forward = -pose.rotation[:, 2]
    aim = pose.position + (pose.up_m - ground_up_m) / -forward[2] * forward
    points = np.array([[aim[0] + de, aim[1] + dn, ground_up_m] for de in (-15, 0, 15) for dn in (-15, 0, 15)])

    pixels, in_front = CAMERA.project(pose, points)
    assert in_front.all()
    assert ((pixels >= 0) & (pixels <= [CAMERA.width - 1, CAMERA.height - 1])).all()
    assert np.allclose(georef.ground_points(frame, pixels[:, 0], pixels[:, 1]), points, atol=1e-6)


def test_pixels_that_never_reach_the_ground_get_nan():
    frame = make_frame(0, 0, 100, 0, 0)  # looking at the horizon
    top, bottom = GeoRef(LOCAL).ground_points(frame, np.array([160.0, 160.0]), np.array([0.0, 239.0]))
    assert np.isnan(top).all() and np.isfinite(bottom).all()
    assert np.isnan(GeoRef(LOCAL, ground_up_m=150.0).ground_points(frame, 160.0, 239.0)).all()  # camera underground

    terrain, grid = block_terrain()
    on_terrain = GeoRef(LOCAL, terrain=terrain, grid=grid)
    top, bottom = on_terrain.ground_points(frame, np.array([160.0, 160.0]), np.array([0.0, 239.0]))
    assert np.isnan(top).all() and np.isfinite(bottom).all()
    assert np.isnan(on_terrain.ground_points(make_frame(0, 0, 5, 0, -45), 160.0, 120.0)).all()  # inside the block


def test_flat_terrain_gives_the_same_points_as_flat_ground():
    terrain = Terrain.flat(rows=40, cols=40, cell_size_m=10.0, elevation_m=12.5)
    on_terrain = GeoRef(LOCAL, terrain=terrain, grid=GroundGrid(rows=40, cols=40, cell_size_m=10.0))
    frame = make_frame(40.0, -25.0, 120.0, 217.0, -20.0)  # top rows show sky, some rows land beyond the grid
    ys, xs = np.mgrid[0 : CAMERA.height : 7, 0 : CAMERA.width : 7]
    expected = GeoRef(LOCAL, ground_up_m=12.5).ground_points(frame, xs, ys)
    assert np.isnan(expected).any()
    assert np.allclose(on_terrain.ground_points(frame, xs, ys), expected, atol=1e-9, equal_nan=True)


@pytest.mark.parametrize(
    "target",
    [(-4.99, 0.0, 10.0), (5.0, 0.0, 5.0)],
    ids=["top-1cm-before-the-edge", "side"],
)
def test_terrain_finds_the_first_cell_a_ray_meets(target):
    """A camera 100 m east of the block looks west and slightly down at a
    point on it. The ray to the top point is below the block's top for only
    its last centimetre over the block; the ray to the side point reaches the
    block already below its top. Either way GeoRef must stop at the block.

    The side point is on the block's east edge, which a plain grid lookup
    counts as the cell east of the block, so this also checks that side hits
    land inside the cell they hit."""
    terrain, grid = block_terrain()
    georef = GeoRef(LOCAL, terrain=terrain, grid=grid)
    camera_east, camera_up = 100.0, 20.5
    pitch = -math.degrees(math.atan2(camera_up - target[2], camera_east - target[0]))
    frame = make_frame(camera_east, 0.0, camera_up, 270.0, pitch)

    pixels, _ = CAMERA.project(georef.pose(frame), np.array([target]))
    point = georef.ground_points(frame, pixels[:, 0], pixels[:, 1])[0]
    assert np.allclose(point, target, atol=1e-4)
    assert grid.cell_at(point[0], point[1]) == (1, 1)


def test_height_above_ground_uses_the_terrain_under_the_camera():
    terrain, grid = block_terrain()
    georef = GeoRef(LOCAL, terrain=terrain, grid=grid)
    assert georef.pose(make_frame(0, 0, 50, 0, -90)).alt_agl_m == pytest.approx(40.0)  # over the block
    assert georef.pose(make_frame(100, 0, 50, 0, -90)).alt_agl_m == pytest.approx(50.0)  # beyond the grid


def test_terrain_needs_a_matching_grid():
    terrain, grid = block_terrain()
    with pytest.raises(ValueError, match="together"):
        GeoRef(LOCAL, terrain=terrain)
    with pytest.raises(ValueError, match="same shape"):
        GeoRef(LOCAL, terrain=terrain, grid=GroundGrid(rows=4, cols=3, cell_size_m=10.0))
    with pytest.raises(ValueError, match="not both"):
        GeoRef(LOCAL, ground_up_m=5.0, terrain=terrain, grid=grid)


def test_gimbal_yaw_turns_the_camera():
    georef = GeoRef(LOCAL)
    turned = georef.pose(make_frame(0, 0, 100, 80.0, -45.0, gimbal_yaw=10.0))
    straight = georef.pose(make_frame(0, 0, 100, 90.0, -45.0))
    assert np.allclose(turned.rotation, straight.rotation)


def test_frames_without_what_georef_needs_are_refused():
    georef = GeoRef(LOCAL)
    frame = make_frame(0, 0, 100, 0, -45)
    frame.fov_deg = None
    with pytest.raises(ValueError, match="field of view"):
        georef.ground_points(frame, 0.0, 0.0)

    frame = make_frame(0, 0, 100, 0, -45)
    frame.telemetry.position.alt_m = None
    with pytest.raises(ValueError, match="altitude"):
        georef.pose(frame)

    with pytest.raises(ValueError, match="gimbal_pitch"):
        georef.pose(make_frame(0, 0, 100, 0, None))
    with pytest.raises(NotImplementedError, match="roll"):
        georef.pose(make_frame(0, 0, 100, 0, -45, gimbal_roll=2.0))


@pytest.mark.parametrize("kind", ["flat", "hill", "hard"])
def test_matches_the_cells_the_simulator_drew(kind):
    """Georeference every pixel of a few sim frames from their telemetry
    alone, and compare with the ground cell MuJoCo rendered there: on flat
    ground, on the demo's 40 m hill, and on the hard scenario's 90 m hill,
    which hides part of the ground behind it."""
    pytest.importorskip("mujoco")
    from vigil.adapters.sim import SimAdapter
    from vigil.sim.fire_sim import demo_scenario, hard_scenario, make_path

    if kind == "hard":
        scenario = hard_scenario(rows=60, cols=60)
        path = make_path("orbit", scenario.grid)
    else:
        scenario = demo_scenario(kind, rows=24, cols=24)
        path = OrbitPath(
            radius_m=150.0, period_s=120.0, altitude_min_m=60.0, altitude_max_m=110.0, altitude_period_s=70.0
        )

    async def collect():
        pairs = []
        async with SimAdapter(scenario, path, CAMERA, duration_s=60.0, frame_interval_s=20.0) as adapter:
            async for frame in adapter.stream_frames():
                pairs.append((frame, adapter.latest))
        return pairs

    georef = GeoRef(scenario.local_frame, terrain=scenario.terrain, grid=scenario.grid)
    grid = scenario.grid
    ys, xs = np.mgrid[0 : CAMERA.height, 0 : CAMERA.width]
    for frame, truth in asyncio.run(collect()):
        pose = georef.pose(frame)
        assert np.allclose(pose.position, truth.pose.position, atol=1e-6)
        assert np.allclose(pose.rotation, truth.pose.rotation, atol=1e-9)

        points = georef.ground_points(frame, xs, ys)
        found = np.isfinite(points[..., 0])
        rows, cols, on_grid = grid.cells_at(np.where(found, points[..., 0], 1e9), np.where(found, points[..., 1], 1e9))
        expected = np.where(found & on_grid, rows * grid.cols + cols, -1)
        assert (expected == truth.cell_ids).mean() > 0.995
