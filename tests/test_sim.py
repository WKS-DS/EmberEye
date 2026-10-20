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
from embereye.sim.camera import CameraModel, LawnmowerPath, OrbitPath, look_rotation
from embereye.sim.fire_sim import (
    BURNED, BURNING, UNBURNED, FireSim, Scenario, demo_scenario, hard_scenario, make_path, mask_to_polygons,
    visible_burning_fraction,
)
from embereye.sim.record import draw_preview

ROWS = COLS = 24
CELL_M = 10.0
CAMERA = CameraModel(width=320, height=240, fov_v_deg=50.0)
PATH = OrbitPath(radius_m=150.0, period_s=120.0, altitude_min_m=60.0, altitude_max_m=110.0, altitude_period_s=70.0)


def make_scenario(terrain=None, rows=ROWS, cols=COLS, ignition=(14, 9)) -> Scenario:
    if terrain is None:
        terrain = Terrain.flat(rows, cols, CELL_M)
    return Scenario(
        grid=GroundGrid(rows, cols, CELL_M),
        terrain=terrain,
        fuel_codes=np.full((rows, cols), 1),
        weather=Weather(wind_speed_mph=5.0, wind_to_deg=45.0),
        ignition_cells=[ignition],
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


MOWER = LawnmowerPath(width_m=600.0, spacing_m=200.0, n_passes=4, altitude_m=150.0, pitch_deg=-60.0, speed_mps=10.0)


def test_lawnmower_keeps_altitude_and_pitch_and_stays_on_its_rectangle():
    south = -(MOWER.n_passes - 1) * MOWER.spacing_m / 2
    for time_s in np.linspace(0, 3 * MOWER.pattern_length_m / MOWER.speed_mps, 400):
        point = MOWER.at(float(time_s))
        assert point.up_m == MOWER.altitude_m
        assert point.pitch_deg == MOWER.pitch_deg
        # Turns bulge out past the ends of the passes by the turn radius.
        assert abs(point.east_m) <= MOWER.width_m / 2 + MOWER.turn_radius_m + 1e-9
        assert south - 1e-9 <= point.north_m <= -south + 1e-9


def test_lawnmower_passes_alternate_direction():
    south = -(MOWER.n_passes - 1) * MOWER.spacing_m / 2
    leg = MOWER.width_m + np.pi * MOWER.turn_radius_m     # one pass plus the turn after it
    for i in range(MOWER.n_passes):
        point = MOWER.at((i * leg + MOWER.width_m / 2) / MOWER.speed_mps)    # middle of pass i
        assert point.north_m == pytest.approx(south + i * MOWER.spacing_m)
        assert point.east_m == pytest.approx(0.0, abs=1e-9)
        assert point.heading_deg == pytest.approx(90.0 if i % 2 == 0 else 270.0)


def test_lawnmower_moves_smoothly_in_the_direction_it_faces():
    dt = 0.5
    for time_s in np.arange(0, 2.2 * MOWER.pattern_length_m / MOWER.speed_mps, dt):
        a, b = MOWER.at(float(time_s)), MOWER.at(float(time_s + dt))
        step = np.hypot(b.east_m - a.east_m, b.north_m - a.north_m)
        assert step <= MOWER.speed_mps * dt + 1e-6          # never jumps
        if step < 0.5 * MOWER.speed_mps * dt:
            continue                                         # straddles the end of the pattern, where it turns back
        travel_deg = np.degrees(np.arctan2(b.east_m - a.east_m, b.north_m - a.north_m)) % 360
        facing_deg = MOWER.at(float(time_s + dt / 2)).heading_deg
        assert abs((travel_deg - facing_deg + 180) % 360 - 180) < 2.0


def test_lawnmower_flies_the_pattern_back_in_reverse():
    end_s = MOWER.pattern_length_m / MOWER.speed_mps
    for offset_s in [5.0, 47.0, 120.0]:
        out, back = MOWER.at(end_s - offset_s), MOWER.at(end_s + offset_s)
        assert (back.east_m, back.north_m) == pytest.approx((out.east_m, out.north_m))
        assert (back.heading_deg - out.heading_deg) % 360 == pytest.approx(180.0)


def test_make_path_keeps_the_demo_orbit_unchanged():
    assert make_path("orbit", demo_scenario().grid) == OrbitPath()
    with pytest.raises(ValueError, match="Unknown path"):
        make_path("zigzag", demo_scenario().grid)


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
# The hard scenario: fire that one frame can't take in
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def hard_lap():
    """The full-size hard scenario seen over one orbit lap, every 20 s."""
    scenario = hard_scenario()
    path = make_path("orbit", scenario.grid)
    with FireSim(scenario, path, CAMERA) as hard_sim:
        yield hard_sim, [hard_sim.observe(float(t)) for t in np.arange(0, path.period_s + 1, 20.0)]


def test_hard_scenario_has_two_separate_fires(hard_lap):
    hard_sim, observations = hard_lap
    for obs in (observations[0], hard_sim.observe(1200.0)):
        n_regions = cv2.connectedComponents((obs.truth_state != UNBURNED).astype(np.uint8))[0] - 1
        assert n_regions == 2


def test_hard_scenario_fire_is_often_partly_out_of_sight(hard_lap):
    _, observations = hard_lap
    fractions = np.array([visible_burning_fraction(obs) for obs in observations])
    assert (fractions < 0.9).mean() >= 0.5


def test_hill_hides_part_of_the_fire_from_part_of_the_orbit(hard_lap):
    """Count burning cells whose centre is in the frame but whose pixel
    shows ground at least 50 m nearer the camera: the hill is in the way.
    (On flat ground this is always 0; a far-off cell that is merely smaller
    than a pixel shows a neighbouring cell at about the same distance.)"""
    hard_sim, observations = hard_lap
    scenario = hard_sim.scenario
    east, north = scenario.grid.cell_centers()
    centers = np.column_stack([east.ravel(), north.ravel(), scenario.terrain.elevation.ravel()])

    hidden_counts = []
    for obs in observations:
        burning = np.flatnonzero(obs.truth_state.ravel() == BURNING)
        pixels, in_front = CAMERA.project(obs.pose, centers[burning])
        x, y = np.rint(pixels[:, 0]).astype(int), np.rint(pixels[:, 1]).astype(int)
        in_view = in_front & (x >= 0) & (x < CAMERA.width) & (y >= 0) & (y < CAMERA.height)

        shown = obs.cell_ids[y[in_view], x[in_view]]
        distance_burning = np.linalg.norm(centers[burning[in_view]] - obs.pose.position, axis=1)
        distance_shown = np.where(shown >= 0, np.linalg.norm(centers[shown] - obs.pose.position, axis=1), np.inf)
        hidden_counts.append(int((distance_shown < distance_burning - 50.0).sum()))

    assert max(hidden_counts) > 100      # from some of the orbit, much of the fire is behind the hill
    assert min(hidden_counts) == 0       # from elsewhere, none of it is


def test_preview_keeps_the_grid_shape_for_non_square_grids():
    wide = make_scenario(rows=20, cols=40, ignition=(10, 20))
    with FireSim(wide, PATH, CAMERA) as wide_sim:
        preview = draw_preview(wide_sim, wide_sim.observe(10.0))

    # Camera frame on the left, a square map panel as tall as it on the right.
    assert preview.shape == (CAMERA.height, CAMERA.width + CAMERA.height, 3)
    # The grid is 400 m wide and 200 m tall, so in the 240-pixel panel it is
    # drawn 240 x 120 with padding above and below. Sample a padding pixel
    # in the bottom-right corner, away from the labels and the drone marker.
    assert (preview[-3, -3] == 40).all()


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
