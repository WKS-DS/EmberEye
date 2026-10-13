"""Tests for the shared ground coordinates (embereye/geo.py)."""
import numpy as np
import pytest

from embereye.geo import GroundGrid, LocalFrame, utm_epsg_for

SPOKANE = (47.6588, -117.4260)
SEATTLE = (47.6062, -122.3321)


def test_utm_zone_follows_longitude():
    assert utm_epsg_for(*SPOKANE) == 32611      # zone 11 north
    assert utm_epsg_for(*SEATTLE) == 32610      # zone 10 north
    assert utm_epsg_for(-33.87, 151.21) == 32756    # Sydney: zone 56 south


def test_origin_is_zero_and_round_trip_is_exact():
    frame = LocalFrame(*SPOKANE)
    east, north = frame.to_local(*SPOKANE)
    assert abs(east) < 1e-6 and abs(north) < 1e-6

    lat, lon = frame.to_gps(123.4, -567.8)
    east, north = frame.to_local(lat, lon)
    assert east == pytest.approx(123.4, abs=1e-6)
    assert north == pytest.approx(-567.8, abs=1e-6)


def test_north_and_east_point_the_right_way():
    frame = LocalFrame(*SPOKANE)
    lat_n, lon_n = frame.to_gps(0.0, 1000.0)
    lat_e, lon_e = frame.to_gps(1000.0, 0.0)
    assert lat_n > SPOKANE[0]       # going north raises latitude
    assert lon_e > SPOKANE[1]       # going east raises longitude
    # One degree of latitude is about 111 km, so 1 km is about 0.009 degrees.
    assert lat_n - SPOKANE[0] == pytest.approx(0.009, abs=0.0005)


def test_local_frame_accepts_arrays():
    frame = LocalFrame(*SPOKANE)
    east = np.array([0.0, 100.0, -250.0])
    north = np.array([0.0, -50.0, 400.0])
    lat, lon = frame.to_gps(east, north)
    back_east, back_north = frame.to_local(lat, lon)
    assert np.allclose(back_east, east, atol=1e-6)
    assert np.allclose(back_north, north, atol=1e-6)


def test_grid_row_zero_is_north_and_col_zero_is_west():
    grid = GroundGrid(rows=4, cols=6, cell_size_m=10.0)
    assert grid.width_m == 60.0 and grid.height_m == 40.0

    east, north = grid.cell_center(0, 0)
    assert (east, north) == (-25.0, 15.0)       # top-left cell: west and north of centre

    east, north = grid.cell_center(3, 5)
    assert (east, north) == (25.0, -15.0)       # bottom-right cell: east and south


def test_cell_at_reverses_cell_center():
    grid = GroundGrid(rows=7, cols=9, cell_size_m=12.5)
    for row in range(grid.rows):
        for col in range(grid.cols):
            assert grid.cell_at(*grid.cell_center(row, col)) == (row, col)


def test_points_off_the_grid():
    grid = GroundGrid(rows=4, cols=4, cell_size_m=10.0)
    assert grid.cell_at(0.0, 0.0) is not None
    assert grid.cell_at(20.1, 0.0) is None      # just past the east edge
    assert grid.cell_at(0.0, -20.1) is None     # just past the south edge


def test_cells_at_matches_cell_at():
    grid = GroundGrid(rows=5, cols=8, cell_size_m=10.0)
    rng = np.random.default_rng(0)
    east = rng.uniform(-60, 60, size=200)
    north = rng.uniform(-40, 40, size=200)

    rows, cols, on_grid = grid.cells_at(east, north)
    for i in range(len(east)):
        expected = grid.cell_at(east[i], north[i])
        if expected is None:
            assert not on_grid[i]
        else:
            assert on_grid[i] and (rows[i], cols[i]) == expected
