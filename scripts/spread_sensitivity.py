"""
spread_sensitivity.py — How wrong can the spread forecast's inputs be? (milestone M3)

Usage:
    python scripts/spread_sensitivity.py

The simulator's "real" fire comes from the same SpreadPredictor that makes
the forecast, so with the right inputs the forecast is close to perfect by
construction. This script measures how fast it falls apart as one input at
a time is made wrong: wind direction (turned clockwise), wind speed, fuel
moisture, or the fuel map.

Setup: the hard scenario at 330 x 330 cells of 30 m (the cell size of
LANDFIRE fuel maps), so the true fire stays inside the grid for 7.5 h.
Each forecast starts from the true fire at 60 min and predicts 1, 3 and
6 h ahead. Nothing is rendered or written to disk; it takes a few seconds.

Each table cell is "IoU / ratio" over the new growth only (vigil/sim/scoring.py, growth_scores):
  IoU    overlap of the predicted and the true new burn (1 = perfect)
  ratio  predicted new burn area / true new burn area (above 1 = the forecast runs too big)
  *      the forecast reached the grid edge, so its area is cut short and the ratio is too low
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np

from vigil.prediction.spread import SpreadPredictor, Weather
from vigil.sim.fire_sim import hard_scenario
from vigil.sim.scoring import forecast_growth_scores

NOW_MIN = 60.0
HOURS = (1, 3, 6)
SHORT_GRASS = 1  # fuel code, see prediction/spread.py


def wrong_inputs(weather: Weather, fuel_codes: np.ndarray) -> list[tuple[str, Weather, np.ndarray]]:
    """The right inputs, then each input changed on its own, as (label, weather, fuel codes)."""
    cases = [("right inputs", weather, fuel_codes)]
    for offset in (15, 30, 45, 90, 180):
        turned = replace(weather, wind_to_deg=(weather.wind_to_deg + offset) % 360)
        cases.append((f"wind direction +{offset}°", turned, fuel_codes))
    for factor in (0, 0.5, 1.5, 2):
        scaled = replace(weather, wind_speed_mph=weather.wind_speed_mph * factor)
        cases.append((f"wind speed x{factor:g}", scaled, fuel_codes))
    for moisture in (4, 12, 16):
        cases.append((f"fuel moisture {moisture}%", replace(weather, fuel_moisture_pct=moisture), fuel_codes))
    cases.append(("fuel map all grass", weather, np.full_like(fuel_codes, SHORT_GRASS)))
    return cases


def reaches_edge(burned: np.ndarray) -> bool:
    """True if a burn grid touches the edge of the grid, where the fire is cut short."""
    return bool(burned[0].any() or burned[-1].any() or burned[:, 0].any() or burned[:, -1].any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    scenario = hard_scenario(rows=330, cols=330, cell_size_m=30.0)
    truth = SpreadPredictor(scenario.terrain, scenario.fuel_codes, scenario.weather)
    truth.set_ignition(scenario.ignition_cells)
    truth.propagate()

    w = scenario.weather
    print(
        f"Hard scenario, 330 x 330 cells of 30 m. True inputs: wind {w.wind_speed_mph:g} mph towards "
        f"{w.wind_to_deg:g}°, fuel moisture {w.fuel_moisture_pct:g}%, the scenario's fuel map."
    )
    print(f"Forecasts start from the true fire at {NOW_MIN:g} min. Each cell: IoU / ratio of the new growth.\n")
    print(f"{'input changed':24s} " + "   ".join(f"{f'{h:g} h':13s}" for h in HOURS))
    for label, weather, fuel_codes in wrong_inputs(scenario.weather, scenario.fuel_codes):
        forecast = SpreadPredictor(scenario.terrain, fuel_codes, weather)
        scores = forecast_growth_scores(truth, forecast, NOW_MIN, HOURS)
        cells = []
        for h in HOURS:
            edge = "*" if reaches_edge(forecast.burned_at(NOW_MIN + 60 * h)) else " "
            cells.append(f"{scores[h]['iou_growth']:.3f} / {scores[h]['growth_ratio']:.2f}{edge}")
        print(f"{label:24s} " + "   ".join(cells))
    print("\n* the forecast reached the grid edge, so its area is cut short and the ratio is too low")


if __name__ == "__main__":
    main()
