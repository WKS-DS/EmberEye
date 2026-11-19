"""Tests for the Fire Mapper (vigil/pipeline/fire_mapper.py).

The frames are built the way an adapter would report them. The last test
checks a whole sim flight against the cells the simulator says each pixel
shows.
"""

import asyncio
from datetime import datetime, timezone

import numpy as np
import pytest

from vigil.geo import GroundGrid, LocalFrame
from vigil.models.types import Frame, GeoPoint, SensorType, Telemetry
from vigil.pipeline.fire_mapper import FireMapper, frame_to_grids
from vigil.pipeline.georef import GeoRef
from vigil.sim.camera import CameraModel, OrbitPath

LOCAL = LocalFrame(origin_lat=47.6588, origin_lon=-117.4260, origin_alt_m=600.0)
CAMERA = CameraModel(width=320, height=240, fov_v_deg=50.0)


def straight_down_frame(up_m: float = 100.0) -> Frame:
    """A camera `up_m` above the local origin, facing north and looking straight down.

    From 100 m one pixel covers about 0.39 m of ground, so the frame shows
    about 124 m east-west and 93 m north-south. The left half of the image
    is west of the camera.
    """
    lat, lon = LOCAL.to_gps(0.0, 0.0)
    return Frame(
        ts=datetime(2026, 8, 1, tzinfo=timezone.utc),
        sensor=SensorType.RGB,
        image=np.zeros((CAMERA.height, CAMERA.width, 3), dtype=np.uint8),
        telemetry=Telemetry(
            position=GeoPoint(lat=float(lat), lon=float(lon), alt_m=LOCAL.origin_alt_m + up_m),
            heading_deg=0.0,
            gimbal_pitch=-90.0,
        ),
        fov_deg=CAMERA.fov_h_deg,
    )


def test_update_keeps_the_first_time_fire_was_seen():
    mapper = FireMapper(GroundGrid(rows=2, cols=2, cell_size_m=10.0))
    assert np.isnan(mapper.first_fire_min).all() and not mapper.covered.any()

    in_view = np.array([[True, True], [False, False]])
    mapper.update(in_view, np.array([[True, False], [False, False]]), time_min=0.0)
    assert mapper.first_fire_min[0, 0] == 0.0  # a fire seen at time 0 is kept

    in_view = np.array([[True, True], [True, False]])
    burning = np.array([[True, True], [False, True]])  # (1, 1) is burning but out of view
    mapper.update(in_view, burning, time_min=5.0)
    assert mapper.first_fire_min[0, 0] == 0.0 and mapper.first_fire_min[0, 1] == 5.0
    assert mapper.ever_burned().tolist() == [[True, True], [False, False]]
    assert mapper.covered.tolist() == [[True, True], [True, False]]

    with pytest.raises(ValueError, match="shape of the map"):
        mapper.update(np.zeros((3, 3), dtype=bool), np.zeros((3, 3), dtype=bool), time_min=6.0)


def test_cells_smaller_than_a_few_pixels_count_as_unseen():
    frame = straight_down_frame()
    no_fire = np.zeros((CAMERA.height, CAMERA.width), dtype=bool)

    big = GroundGrid(rows=8, cols=8, cell_size_m=10.0)  # 80 m square, about 26 pixels per cell side
    in_view, _ = frame_to_grids(GeoRef(LOCAL), big, frame, no_fire)
    assert in_view.all()

    tiny = GroundGrid(rows=40, cols=40, cell_size_m=0.25)  # under a pixel per cell side
    in_view, _ = frame_to_grids(GeoRef(LOCAL), tiny, frame, no_fire)
    assert not in_view.any()


def test_cells_burn_where_at_least_four_fire_pixels_land():
    frame = straight_down_frame()
    grid = GroundGrid(rows=8, cols=8, cell_size_m=10.0)  # columns 0-3 are west of the camera
    fire = np.zeros((CAMERA.height, CAMERA.width), dtype=bool)
    fire[:, : CAMERA.width // 2] = True  # the west half of the frame
    fire[120, 220] = True  # one stray pixel in an eastern cell
    fire[60:62, 260:262] = True  # 2 x 2 fire pixels in another eastern cell

    _, burning = frame_to_grids(GeoRef(LOCAL), grid, frame, fire)
    assert burning[:, :4].all()
    assert burning[:, 4:].sum() == 1  # only the 2 x 2 patch counts

    with pytest.raises(ValueError, match="fire mask"):
        frame_to_grids(GeoRef(LOCAL), grid, frame, fire[:100])


def test_map_of_a_sim_flight_matches_what_the_flight_saw():
    """Map a sim flight over the demo hill, with the sim's fire outlines
    filled in standing in for the detector. Then build the same map from
    the simulator's own record of which cell each pixel shows."""
    pytest.importorskip("mujoco")
    from vigil.adapters.sim import SimAdapter
    from vigil.masks import polygons_to_mask
    from vigil.sim.fire_sim import demo_scenario

    scenario = demo_scenario("hill", rows=24, cols=24)
    path = OrbitPath(radius_m=150.0, period_s=120.0, altitude_min_m=60.0, altitude_max_m=110.0, altitude_period_s=70.0)

    async def collect():
        pairs = []
        async with SimAdapter(scenario, path, CAMERA, duration_s=60.0, frame_interval_s=10.0) as adapter:
            async for frame in adapter.stream_frames():
                pairs.append((frame, adapter.latest))
        return pairs

    grid = scenario.grid
    georef = GeoRef(scenario.local_frame, terrain=scenario.terrain, grid=grid)
    mapper = FireMapper(grid)
    seen, seen_burning = np.zeros(grid.shape, dtype=bool), np.zeros(grid.shape, dtype=bool)
    for frame, truth in asyncio.run(collect()):
        fire = polygons_to_mask(truth.fire_polygons, truth.fire_mask.shape)
        in_view, burning = frame_to_grids(georef, grid, frame, fire)
        mapper.update(in_view, burning, truth.fire_time_min)

        shown = truth.cell_ids >= 0
        pixels = np.bincount(truth.cell_ids[shown], minlength=grid.rows * grid.cols).reshape(grid.shape)
        fire_pixels = np.bincount(truth.cell_ids[shown & fire], minlength=grid.rows * grid.cols).reshape(grid.shape)
        seen |= pixels >= 4
        seen_burning |= fire_pixels >= 4

    assert seen_burning.sum() > 20  # the flight saw a real fire
    assert (mapper.covered == seen).mean() > 0.99
    mapped = mapper.ever_burned()
    assert (mapped & seen_burning).sum() / (mapped | seen_burning).sum() > 0.95
