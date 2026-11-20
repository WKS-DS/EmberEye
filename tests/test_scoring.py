"""Tests for the scores against the simulator's answer key (vigil/sim/scoring.py).

scripts/score_m2.py and scripts/spread_sensitivity.py run these on whole
flights and fires; here each one gets a small input whose answer can be
worked out by hand.
"""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("mujoco")  # vigil.sim imports the MuJoCo scene

from vigil.geo import GroundGrid  # noqa: E402
from vigil.prediction.spread import SpreadPredictor  # noqa: E402
from vigil.prediction.terrain import Terrain  # noqa: E402
from vigil.sim.fire_sim import BURNED, BURNING, UNBURNED, hard_scenario  # noqa: E402
from vigil.sim.scoring import (  # noqa: E402
    forecast_growth_scores,
    growth_scores,
    map_scores,
    observed_grids,
    true_ground_distance,
)


def test_observed_grids_use_the_fire_mappers_rules():
    # A 3 x 4 pixel frame over a 2 x 2 grid (cells 0-3); -1 = no cell.
    cell_ids = np.array(
        [
            [0, 0, 0, 2],
            [0, 0, 1, 2],
            [1, 1, 2, 2],
        ]
    )
    fire = np.array(
        [
            [True, True, True, True],
            [True, False, True, True],
            [False, False, False, False],
        ]
    )
    in_view, burning = observed_grids(cell_ids, fire, (2, 2), min_pixels=4)
    # Cell 0: 5 pixels, 4 on fire. Cell 1: 3 pixels, too few. Cell 2: 4 pixels, 3 on fire. Cell 3: none.
    assert in_view.tolist() == [[True, False], [True, False]]
    assert burning.tolist() == [[True, False], [False, False]]


def test_map_scores():
    truth = np.array([[BURNED, BURNING, UNBURNED], [BURNING, UNBURNED, UNBURNED]])
    covered = np.array([[True, True, True], [False, True, True]])  # (1, 0) was never in view
    ever_burned = np.array([[True, False, True], [False, False, False]])
    seen_burning = np.array([[True, False, False], [False, False, False]])

    scores = map_scores(ever_burned, covered, truth, seen_burning)
    # Over covered cells: burned = (0,0), (0,1); mapped = (0,0), (0,2). 1 shared of 3.
    assert scores["iou_all_burned"] == pytest.approx(1 / 3)
    assert scores["iou_seen"] == pytest.approx(1 / 2)
    assert scores["coverage"] == pytest.approx(5 / 6)
    assert scores["burned_seen"] == pytest.approx(2 / 3)


def test_true_ground_distance_finds_the_top_and_the_side_of_a_tile():
    # 3 x 3 cells of 10 m, flat at 0 m except a 10 m block in the middle
    # cell (index 4), which spans -5 to 5 m east and north.
    elevation = np.zeros((3, 3))
    elevation[1, 1] = 10.0
    terrain, grid = Terrain(elevation, cell_size_m=10.0), GroundGrid(rows=3, cols=3, cell_size_m=10.0)

    down = np.array([[0.0, 0.0, -1.0]])
    assert true_ground_distance(np.array([0.0, 0.0, 50.0]), down, np.array([4]), grid, terrain) == pytest.approx(40.0)
    assert true_ground_distance(np.array([10.0, 0.0, 50.0]), down, np.array([5]), grid, terrain) == pytest.approx(50.0)

    west = np.array([[-1.0, 0.0, 0.0]])  # level, 5 m up, 20 m east of the middle: meets the block's east side
    assert true_ground_distance(np.array([20.0, 0.0, 5.0]), west, np.array([4]), grid, terrain) == pytest.approx(15.0)


def test_growth_scores_leave_out_what_had_already_burned():
    burned_now = np.array([[True, True, False, False, False]])
    truth = np.array([[True, True, True, True, False]])  # new: cells 2 and 3
    predicted = np.array([[True, False, True, False, True]])  # new: cells 2 and 4; missing cell 1 doesn't count

    scores = growth_scores(predicted, truth, burned_now)
    assert scores["iou_growth"] == pytest.approx(1 / 3)  # cell 2 shared, of cells 2, 3, 4
    assert scores["growth_ratio"] == pytest.approx(2 / 2)

    nothing_new = growth_scores(burned_now, burned_now, burned_now)
    assert nothing_new["iou_growth"] == 1.0
    assert np.isnan(nothing_new["growth_ratio"])


def test_forecast_with_the_right_inputs_matches_and_reversed_wind_does_not():
    """The setup of scripts/spread_sensitivity.py. The sim's fire is the same
    model as the forecast, so the right inputs must score close to 1."""
    scenario = hard_scenario(rows=330, cols=330, cell_size_m=30.0)
    truth = SpreadPredictor(scenario.terrain, scenario.fuel_codes, scenario.weather)
    truth.set_ignition(scenario.ignition_cells)
    truth.propagate()

    right = SpreadPredictor(scenario.terrain, scenario.fuel_codes, scenario.weather)
    reversed_wind = replace(scenario.weather, wind_to_deg=(scenario.weather.wind_to_deg + 180) % 360)
    wrong = SpreadPredictor(scenario.terrain, scenario.fuel_codes, reversed_wind)

    assert forecast_growth_scores(truth, right, 60.0, [3])[3]["iou_growth"] > 0.95
    assert forecast_growth_scores(truth, wrong, 60.0, [3])[3]["iou_growth"] < 0.3
