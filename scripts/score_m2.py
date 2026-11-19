"""
score_m2.py — Score GeoRef and the Fire Mapper on simulated flights (milestone M2).

Usage:
    python scripts/score_m2.py [--scenario demo hard] [--path orbit lawnmower] [--duration 900] [--interval 15]

For each scenario and path: fly the path, run GeoRef (with the scenario's
terrain) and the Fire Mapper on every frame, and print one row of scores.
The sim's fire outlines, filled in, stand in for the detector, so this
scores the geometry, not fire detection. Nothing is written to disk.

Columns (definitions in vigil/sim/scoring.py and docs/ARCHITECTURE.md, Scoring):
  coverage     share of all cells in view at least once
  burned seen  share of the burned cells in view at least once
  IoU all      "ever burned" vs every burning or burned-out cell at the end, over cells in view at least once
  IoU seen     "ever burned" vs the cells the flight saw burning (by the simulator's record of each pixel)
  GeoRef error distance from each pixel's georeferenced point to the point the simulator drew (mean, 95th
               percentile), over pixels that show a grid cell. With the sim's exact telemetry it is close
               to 0, because GeoRef uses the same terrain the sim draws; it is here to catch regressions
               and for when telemetry noise is added.
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from vigil.adapters.sim import SimAdapter
from vigil.masks import polygons_to_mask
from vigil.pipeline.fire_mapper import MIN_PIXELS_PER_CELL, FireMapper, frame_to_grids
from vigil.pipeline.georef import GeoRef
from vigil.sim.camera import CameraModel
from vigil.sim.fire_sim import PATHS, SCENARIOS, make_path
from vigil.sim.scoring import map_scores, observed_grids, true_ground_distance


async def score_flight(scenario_name: str, path_name: str, duration_s: float, interval_s: float) -> dict[str, float]:
    """Fly one scenario and path, and return its scores."""
    scenario = SCENARIOS[scenario_name]()
    grid, terrain, camera = scenario.grid, scenario.terrain, CameraModel()
    georef = GeoRef(scenario.local_frame, terrain=terrain, grid=grid)
    mapper = FireMapper(grid)
    seen_burning = np.zeros(grid.shape, dtype=bool)
    errors = []
    ys, xs = np.mgrid[0 : camera.height, 0 : camera.width]

    started = time.perf_counter()
    path = make_path(path_name, grid)
    async with SimAdapter(scenario, path, camera, duration_s=duration_s, frame_interval_s=interval_s) as adapter:
        async for frame in adapter.stream_frames():
            truth = adapter.latest
            fire = polygons_to_mask(truth.fire_polygons, truth.fire_mask.shape)

            in_view, burning = frame_to_grids(georef, grid, frame, fire)
            mapper.update(in_view, burning, truth.fire_time_min)
            seen_burning |= observed_grids(truth.cell_ids, fire, grid.shape, MIN_PIXELS_PER_CELL)[1]

            shows_cell = truth.cell_ids >= 0
            rays = camera.pixel_rays(truth.pose, xs, ys)[shows_cell]
            distance = true_ground_distance(truth.pose.position, rays, truth.cell_ids[shows_cell], grid, terrain)
            true_points = truth.pose.position + distance[:, None] * rays
            errors.append(np.linalg.norm(georef.ground_points(frame, xs, ys)[shows_cell] - true_points, axis=1))

    errors = np.concatenate(errors)
    scores = map_scores(mapper.ever_burned(), mapper.covered, truth.truth_state, seen_burning)
    scores.update(
        frames=truth.index + 1,
        georef_mean_m=float(np.nanmean(errors)),
        georef_p95_m=float(np.nanpercentile(errors, 95)),
        georef_missing=float(np.isnan(errors).mean()),  # pixels that show a cell but GeoRef placed nowhere
        seconds=time.perf_counter() - started,
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", nargs="+", choices=list(SCENARIOS), default=list(SCENARIOS))
    parser.add_argument("--path", nargs="+", choices=PATHS, default=list(PATHS))
    parser.add_argument("--duration", type=float, default=900.0, help="seconds of flight")
    parser.add_argument("--interval", type=float, default=15.0, help="seconds between frames")
    args = parser.parse_args()

    print(f"{args.duration:.0f} s flights, a frame every {args.interval:.0f} s, 640 x 480 camera\n")
    print("scenario  path       frames  coverage  burned seen  IoU all  IoU seen  GeoRef mean / p95 m  run s")
    for scenario_name in args.scenario:
        for path_name in args.path:
            s = asyncio.run(score_flight(scenario_name, path_name, args.duration, args.interval))
            print(
                f"{scenario_name:8s}  {path_name:9s}  {s['frames']:6d}  {s['coverage']:8.0%}  {s['burned_seen']:11.0%}  "
                f"{s['iou_all_burned']:7.3f}  {s['iou_seen']:8.3f}  {s['georef_mean_m']:9.4f} / {s['georef_p95_m']:.4f}  "
                f"{s['seconds']:5.0f}"
            )
            if s["georef_missing"] > 0:
                print(f"          {s['georef_missing']:.2%} of pixels that show a cell got no point from GeoRef")


if __name__ == "__main__":
    main()
