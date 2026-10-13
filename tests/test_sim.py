"""Tests for the fire simulator (embereye/sim/ and the sim adapter).

The most important ones check that the simulator's "answer key" is
trustworthy: the camera pose it reports really is the pose the picture was
taken from, and the fire mask really shows the cells that are burning.
"""
import asyncio

import cv2
import numpy as np
import pytest

# The simulator needs MuJoCo. Skip these tests (rather than fail) on a
# machine where it is not installed.
pytest.importorskip("mujoco")

from embereye.adapters.sim import SimAdapter
from embereye.geo import GroundGrid, LocalFrame
from embereye.models.types import SensorType
from embereye.prediction.spread import Weather
from embereye.prediction.terrain import Terrain
from embereye.sim.camera import CameraModel, OrbitPath, look_rotation
from embereye.sim.fire_sim import BURNED, BURNING, UNBURNED, FireSim, Scenario, mask_to_polygons

ROWS = COLS = 24
CELL_M = 10.0
CAMERA = CameraModel(width=320, height=240, fov_v_deg=50.0)
PATH = OrbitPath(radius_m=150.0, period_s=120.0, altitude_min_m=60.0, altitude_max_m=110.0, altitude_period_s=70.0)


def make_scenario(terrain=None) -> Scenario:
    if terrain is None:
        terrain = Terrain.flat(ROWS, COLS, CELL_M)
    return Scenario(
        grid=GroundGrid(ROWS, COLS, CELL_M),
        terrain=terrain,
        fuel_codes=np.full((ROWS, COLS), 1),
        weather=Weather(wind_speed_mph=5.0, wind_to_deg=45.0),
        ignition_cells=[(14, 9)],
        local_frame=LocalFrame(origin_lat=47.6588, origin_lon=-117.4260, origin_alt_m=600.0),
        fire_age_at_start_min=8.0,
        burn_duration_min=5.0,
    )


@pytest.fixture(scope="module")
def sim():
    """One simulator shared by the tests in this file (building it takes a moment)."""
    with FireSim(make_scenario(), PATH, CAMERA) as fire_sim:
        yield fire_sim


# ---------------------------------------------------------------------------
# Camera maths
# ---------------------------------------------------------------------------
def test_look_rotation_axes():
    # Facing north at the horizon: right is east, up is up, forward is north.
    right, up, back = look_rotation(0.0, 0.0).T
    assert np.allclose(right, [1, 0, 0])
    assert np.allclose(up, [0, 0, 1])
    assert np.allclose(-back, [0, 1, 0])

    # Facing east, looking straight down: forward is down, top of image is east.
    right, up, back = look_rotation(90.0, -90.0).T
    assert np.allclose(-back, [0, 0, -1], atol=1e-12)
    assert np.allclose(up, [1, 0, 0], atol=1e-12)


def test_orbit_stays_on_its_circle_and_aims_at_the_centre():
    for time_s in np.linspace(0, 300, 25):
        point = PATH.at(float(time_s))
        assert np.hypot(point.east_m, point.north_m) == pytest.approx(PATH.radius_m)
        assert PATH.altitude_min_m - 1e-9 <= point.up_m <= PATH.altitude_max_m + 1e-9

        # The direction the camera looks should pass through the target (0, 0, 0).
        forward = -look_rotation(point.heading_deg, point.pitch_deg)[:, 2]
        to_target = -np.array([point.east_m, point.north_m, point.up_m])
        assert np.allclose(forward, to_target / np.linalg.norm(to_target), atol=1e-9)


def test_orbit_altitude_actually_varies():
    heights = [PATH.at(float(t)).up_m for t in np.linspace(0, PATH.altitude_period_s, 50)]
    assert min(heights) == pytest.approx(PATH.altitude_min_m, abs=1.0)
    assert max(heights) == pytest.approx(PATH.altitude_max_m, abs=1.0)


def test_project_and_pixel_rays_are_inverses(sim):
    pose = sim.pose_at(33.0)
    points = np.array([[0.0, 0.0, 0.0], [40.0, -25.0, 0.0], [-60.0, 80.0, 0.0]])
    pixels, in_front = CAMERA.project(pose, points)
    assert in_front.all()

    rays = CAMERA.pixel_rays(pose, pixels[:, 0], pixels[:, 1])
    expected = points - pose.position
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(rays, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# The answer key: pose and segmentation
# ---------------------------------------------------------------------------
def test_observation_shapes_and_types(sim):
    obs = sim.observe(20.0)
    assert obs.rgb.shape == (240, 320, 3) and obs.rgb.dtype == np.uint8
    assert obs.cell_ids.shape == (240, 320) and obs.cell_ids.dtype == np.int32
    assert obs.fire_mask.shape == (240, 320) and obs.fire_mask.dtype == bool
    assert obs.truth_state.shape == (ROWS, COLS)
    assert obs.cell_ids.max() < ROWS * COLS and obs.cell_ids.min() >= -1
    assert obs.ts.tzinfo is not None


@pytest.mark.parametrize("time_s", [0.0, 17.0, 41.0, 95.0])
def test_reported_pose_matches_the_rendered_image(sim, time_s):
    """Shoot a ray through every pixel using the REPORTED pose, find where
    it meets the flat ground, and check that is the cell MuJoCo says the
    pixel shows. If the pose were wrong in any way, this would not agree."""
    obs = sim.observe(time_s)
    grid = sim.scenario.grid

    ys, xs = np.mgrid[0:CAMERA.height, 0:CAMERA.width]
    rays = CAMERA.pixel_rays(obs.pose, xs, ys)
    looks_down = rays[..., 2] < -1e-6

    # Distance along each ray to the ground plane (up = 0).
    distance = np.where(looks_down, -obs.pose.up_m / np.where(looks_down, rays[..., 2], -1.0), 0.0)
    east = obs.pose.east_m + distance * rays[..., 0]
    north = obs.pose.north_m + distance * rays[..., 1]

    rows, cols, on_grid = grid.cells_at(east, north)
    expected = np.where(looks_down & on_grid, rows * grid.cols + cols, -1)

    agree = expected == obs.cell_ids
    assert agree.mean() > 0.995

    # The few disagreements should be pixels sitting on the line between
    # two cells, i.e. never more than one cell apart.
    both = ~agree & (expected >= 0) & (obs.cell_ids >= 0)
    row_gap = np.abs(expected[both] // grid.cols - obs.cell_ids[both] // grid.cols)
    col_gap = np.abs(expected[both] % grid.cols - obs.cell_ids[both] % grid.cols)
    assert (row_gap <= 1).all() and (col_gap <= 1).all()


def test_pose_is_reported_consistently(sim):
    obs = sim.observe(52.0)
    pose = obs.pose
    frame = sim.scenario.local_frame

    east, north = frame.to_local(pose.lat, pose.lon)
    assert east == pytest.approx(pose.east_m, abs=1e-6)
    assert north == pytest.approx(pose.north_m, abs=1e-6)
    assert pose.alt_msl_m == pytest.approx(frame.origin_alt_m + pose.up_m)
    assert pose.alt_agl_m == pytest.approx(pose.up_m)       # flat ground at up = 0
    assert np.allclose(pose.rotation, look_rotation(pose.heading_deg, pose.pitch_deg), atol=1e-9)


def test_altitude_above_ground_accounts_for_terrain():
    hill = Terrain.hill(ROWS, COLS, CELL_M, height_m=30.0)
    # Hover straight over the hill top by making the orbit radius tiny.
    over_the_top = OrbitPath(radius_m=1.0, altitude_min_m=100.0, altitude_max_m=100.0)
    with FireSim(make_scenario(hill), over_the_top, CAMERA) as hill_sim:
        pose = hill_sim.pose_at(0.0)

    ground = hill.elevation[hill_sim.scenario.grid.cell_at(pose.east_m, pose.north_m)]
    assert ground > 25.0
    assert pose.alt_agl_m == pytest.approx(100.0 - ground)
    assert pose.alt_msl_m == pytest.approx(700.0)


# ---------------------------------------------------------------------------
# The fire
# ---------------------------------------------------------------------------
def test_fire_mask_shows_exactly_the_burning_cells(sim):
    obs = sim.observe(30.0)
    grid = sim.scenario.grid
    burning = obs.truth_state == BURNING
    assert burning.any() and obs.fire_mask.any()

    # Every fire pixel shows a cell that really is burning...
    seen_cells = obs.cell_ids[obs.fire_mask]
    assert burning.ravel()[seen_cells].all()

    # ...and every burning cell whose centre is in view is marked as fire.
    east, north = grid.cell_centers()
    centers = np.column_stack([east[burning], north[burning], np.zeros(burning.sum())])
    pixels, in_front = CAMERA.project(obs.pose, centers)
    x = np.rint(pixels[:, 0]).astype(int)
    y = np.rint(pixels[:, 1]).astype(int)
    in_view = in_front & (x >= 0) & (x < CAMERA.width) & (y >= 0) & (y < CAMERA.height)
    assert in_view.sum() > 0
    assert obs.fire_mask[y[in_view], x[in_view]].all()


def test_fire_polygons_outline_the_mask(sim):
    obs = sim.observe(30.0)
    assert len(obs.fire_polygons) >= 1
    for polygon in obs.fire_polygons:
        assert polygon.ndim == 2 and polygon.shape[1] == 2 and len(polygon) >= 3
        assert polygon[:, 0].min() >= 0 and polygon[:, 0].max() <= CAMERA.width - 1
        assert polygon[:, 1].min() >= 0 and polygon[:, 1].max() <= CAMERA.height - 1

    # Filling the polygons back in should cover the mask (and little else,
    # apart from any burned-out hole in the middle).
    filled = np.zeros(obs.fire_mask.shape, dtype=np.uint8)
    cv2.fillPoly(filled, [p.astype(np.int32) for p in obs.fire_polygons], 1)
    assert filled[obs.fire_mask].mean() > 0.99


def test_mask_to_polygons_drops_specks():
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:30, 10:30] = True       # a real region
    mask[40, 40] = True             # a single stray pixel
    polygons = mask_to_polygons(mask)
    assert len(polygons) == 1


def test_fire_grows_then_burns_out(sim):
    early, later = sim.fire_state(0.0), sim.fire_state(240.0)

    touched_early = early != UNBURNED
    touched_later = later != UNBURNED
    assert touched_later.sum() > touched_early.sum()
    assert touched_later[touched_early].all()       # fire never un-burns

    # A cell stays alight for burn_duration_min, then counts as burned out.
    arrival = sim.truth.arrival_time
    now = sim.fire_time_min(240.0)
    burn = sim.scenario.burn_duration_min
    assert ((later == BURNED) == (arrival + burn <= now)).all()
    assert ((later == BURNING) == ((arrival <= now) & (arrival + burn > now))).all()


def test_same_inputs_give_identical_frames():
    with FireSim(make_scenario(), PATH, CAMERA) as a, FireSim(make_scenario(), PATH, CAMERA) as b:
        obs_a, obs_b = a.observe(25.0), b.observe(25.0)
    assert np.array_equal(obs_a.rgb, obs_b.rgb)
    assert np.array_equal(obs_a.cell_ids, obs_b.cell_ids)
    assert obs_a.ts == obs_b.ts


def test_run_yields_frames_at_the_requested_interval(sim):
    observations = list(sim.run(duration_s=10.0, frame_interval_s=2.5))
    assert [obs.index for obs in observations] == [0, 1, 2, 3, 4]
    assert [obs.time_s for obs in observations] == [0.0, 2.5, 5.0, 7.5, 10.0]
    assert observations[-1].fire_time_min == pytest.approx(8.0 + 10.0 / 60.0)


def test_no_fire_is_an_error():
    scenario = make_scenario()
    scenario.ignition_cells = [(999, 999)]      # off the grid
    with pytest.raises(ValueError, match="no fire"):
        FireSim(scenario, PATH, CAMERA)


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------
def test_sim_adapter_emits_frames_with_telemetry():
    async def collect():
        adapter = SimAdapter(make_scenario(), PATH, CAMERA, duration_s=4.0, frame_interval_s=1.0)
        frames, answer_keys = [], []
        async with adapter:
            async for frame in adapter.stream_frames():
                frames.append(frame)
                answer_keys.append(adapter.latest)
        return frames, answer_keys

    frames, answer_keys = asyncio.run(collect())
    assert len(frames) == 5

    for frame, obs in zip(frames, answer_keys):
        assert frame.sensor == SensorType.RGB
        assert frame.image is obs.rgb
        assert frame.telemetry.position.lat == pytest.approx(obs.pose.lat)
        assert frame.telemetry.position.alt_m == pytest.approx(obs.pose.alt_msl_m)
        assert frame.telemetry.heading_deg == pytest.approx(obs.pose.heading_deg)
        assert frame.telemetry.gimbal_pitch == pytest.approx(obs.pose.pitch_deg)
        assert frame.fov_deg == pytest.approx(CAMERA.fov_h_deg)
    assert [frame.frame_id for frame in frames] == [f"{i:08d}" for i in range(5)]
