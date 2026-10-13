"""Tests for the arrival-time fire spread model (embereye/prediction/spread.py)."""
import math

import numpy as np
import pytest

from embereye.prediction.spread import FUEL_MODELS, NON_BURNABLE, SpreadPredictor, Weather
from embereye.prediction.terrain import Terrain

CALM = Weather(wind_speed_mph=0.0, wind_to_deg=0.0)
GRASS = 1
GRASS_ROS = FUEL_MODELS[GRASS]["ros_ref"]      # m/min


def make_predictor(rows=21, cols=21, cell_size_m=10.0, fuel=GRASS, weather=CALM, terrain=None):
    """A predictor on uniform fuel, ignited at the centre cell."""
    if terrain is None:
        terrain = Terrain.flat(rows, cols, cell_size_m)
    predictor = SpreadPredictor(terrain, np.full((rows, cols), fuel), weather)
    center = (rows // 2, cols // 2)
    predictor.set_ignition([center])
    predictor.propagate()
    return predictor, center


def test_calm_flat_fire_spreads_at_the_base_rate():
    predictor, (r, c) = make_predictor()
    arrival = predictor.arrival_time

    assert arrival[r, c] == 0.0
    # 5 cells of 10 m at 2.5 m/min = 20 minutes, the same in every direction.
    expected = 5 * 10.0 / GRASS_ROS
    for cell in [(r, c + 5), (r, c - 5), (r - 5, c), (r + 5, c)]:
        assert arrival[cell] == pytest.approx(expected)
    # Diagonal cells are sqrt(2) times further away.
    assert arrival[r - 5, c + 5] == pytest.approx(expected * math.sqrt(2))


def test_slow_fire_still_spreads():
    """A step-based model loses any fire slower than one cell per step.
    The arrival-time model must not: slow fuel just takes longer."""
    predictor, (r, c) = make_predictor(cell_size_m=30.0, fuel=8)     # timber litter, 0.3 m/min
    assert predictor.arrival_time[r, c + 1] == pytest.approx(30.0 / 0.3)    # 100 minutes
    assert np.isfinite(predictor.arrival_time).all()


def test_wind_pushes_the_fire_downwind():
    east_wind = Weather(wind_speed_mph=10.0, wind_to_deg=90.0)     # blowing towards the east
    predictor, (r, c) = make_predictor(weather=east_wind)
    arrival = predictor.arrival_time

    downwind, upwind, crosswind = arrival[r, c + 5], arrival[r, c - 5], arrival[r - 5, c]
    assert downwind < crosswind < upwind


def test_fire_runs_uphill_faster_than_downhill():
    rows = cols = 21
    rising_north = Terrain.ramp(rows, cols, 10.0, slope_deg=20.0, uphill_deg=0.0)
    predictor, (r, c) = make_predictor(rows, cols, terrain=rising_north)
    arrival = predictor.arrival_time

    uphill, downhill = arrival[r - 5, c], arrival[r + 5, c]     # row - 5 is five cells north
    assert uphill < downhill


def test_wetter_fuel_burns_slower():
    dry, (r, c) = make_predictor(weather=Weather(0.0, 0.0, fuel_moisture_pct=6.0))
    wet, _      = make_predictor(weather=Weather(0.0, 0.0, fuel_moisture_pct=20.0))
    assert dry.arrival_time[r, c + 5] < wet.arrival_time[r, c + 5]


def test_non_burnable_cells_block_the_fire():
    rows = cols = 11
    fuel = np.full((rows, cols), GRASS)
    fuel[:, 6] = NON_BURNABLE       # a road from the north edge to the south edge
    predictor = SpreadPredictor(Terrain.flat(rows, cols, 10.0), fuel, CALM)
    predictor.set_ignition([(5, 2)])
    predictor.propagate()

    assert np.isfinite(predictor.arrival_time[:, :6]).all()     # west of the road burns
    assert np.isinf(predictor.arrival_time[:, 6:]).all()        # the road and everything east do not


def test_ignition_skips_cells_that_cannot_burn():
    fuel = np.full((5, 5), GRASS)
    fuel[2, 2] = NON_BURNABLE
    predictor = SpreadPredictor(Terrain.flat(5, 5, 10.0), fuel, CALM)

    ignited = predictor.set_ignition([(2, 2), (99, 99), (0, 0)])    # road, off-grid, grass
    assert ignited == 1


def test_predict_horizons_are_measured_from_time_zero():
    predictor, _ = make_predictor(rows=61, cols=61, cell_size_m=30.0, fuel=8)
    predictions = predictor.predict([1, 3])

    assert set(predictions) == {"1hr", "3hr"}
    assert np.array_equal(predictions["3hr"], predictor.burned_at(180))
    # Everything burned after 1 hour is still burned after 3, and more besides.
    assert predictions["3hr"][predictions["1hr"]].all()
    assert predictions["3hr"].sum() > predictions["1hr"].sum()


def test_bad_inputs_are_rejected():
    terrain = Terrain.flat(5, 5, 10.0)
    with pytest.raises(ValueError, match="shape"):
        SpreadPredictor(terrain, np.full((4, 5), GRASS), CALM)
    with pytest.raises(ValueError, match="Unknown fuel codes"):
        SpreadPredictor(terrain, np.full((5, 5), 42), CALM)


def test_wind_from_is_opposite_wind_to():
    assert Weather(5.0, wind_to_deg=60.0).wind_from_deg == 240.0
    assert Weather(5.0, wind_to_deg=270.0).wind_from_deg == 90.0


def test_ramp_terrain_has_the_requested_slope_and_aspect():
    terrain = Terrain.ramp(10, 10, 10.0, slope_deg=15.0, uphill_deg=90.0)   # rising towards the east
    assert np.allclose(terrain.slope, 15.0)
    assert np.allclose(terrain.aspect, 270.0)       # so it faces downhill to the west
