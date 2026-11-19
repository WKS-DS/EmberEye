"""
scoring.py — Score the M2 pipeline (GeoRef + Fire Mapper) against the simulator's answer key.

The scores are defined in docs/ARCHITECTURE.md (System-Level Validation,
Scoring); scripts/score_m2.py runs them on whole flights.

The fire map is scored against two truths:
- Everything burned: every cell burning or burned out at the end of the
  flight. This is the documented score. It also counts cells nobody could
  have seen burning: cells that burned out before the drone first looked,
  or caught fire after its last look.
- What the flight saw: the cells that were inside a fire outline while in
  view, judged with the Fire Mapper's own rules but from the simulator's
  record of which cell each pixel shows (SimObservation.cell_ids). This
  scores the mapping itself.

Everything here takes plain arrays, so this module doesn't import the
pipeline it scores.
"""

from __future__ import annotations

import numpy as np

from vigil.geo import GroundGrid
from vigil.masks import iou
from vigil.prediction.terrain import Terrain
from vigil.sim.fire_sim import UNBURNED
from vigil.sim.world import TILE_DEPTH_M


def observed_grids(
    cell_ids: np.ndarray, fire_mask: np.ndarray, shape: tuple[int, int], min_pixels: int
) -> tuple[np.ndarray, np.ndarray]:
    """One frame's `in_view` and `burning` grids, from the cell each pixel truly shows.

    The same rules as pipeline.fire_mapper.frame_to_grids: in view if at
    least `min_pixels` pixels show the cell, burning if at least that many
    of them show fire. cell_ids is (H, W), -1 where no cell is shown.
    """
    shown = cell_ids >= 0
    n_cells = shape[0] * shape[1]
    pixels = np.bincount(cell_ids[shown], minlength=n_cells).reshape(shape)
    fire_pixels = np.bincount(cell_ids[shown & fire_mask], minlength=n_cells).reshape(shape)
    return pixels >= min_pixels, fire_pixels >= min_pixels


def map_scores(
    ever_burned: np.ndarray, covered: np.ndarray, truth_state: np.ndarray, seen_burning: np.ndarray
) -> dict[str, float]:
    """Scores for a finished fire map; all inputs are (rows, cols) grids.

    iou_all_burned: IoU with every burning or burned-out cell, over the
                    cells that were in view at least once (`covered`).
    iou_seen:       IoU with `seen_burning`, the cells the flight saw burning.
    coverage:       share of all cells that were in view at least once.
    burned_seen:    share of the burned cells that were in view at least once.
    """
    burned = truth_state != UNBURNED
    return {
        "iou_all_burned": iou(ever_burned & covered, burned & covered),
        "iou_seen": iou(ever_burned, seen_burning),
        "coverage": float(covered.mean()),
        "burned_seen": float((burned & covered).sum() / max(int(burned.sum()), 1)),
    }


def true_ground_distance(
    start: np.ndarray, rays: np.ndarray, cell_ids: np.ndarray, grid: GroundGrid, terrain: Terrain
) -> np.ndarray:
    """Distance along each ray (N, 3) to the ground the simulator drew for that pixel.

    That is where the ray enters the tile of the cell the pixel shows
    (cell_ids, all >= 0): a box from below the lowest ground up to the
    cell's elevation (see sim/world.py). Found by the usual ray/box test:
    the ray is inside the box once it is between the box's two faces on
    every axis, so it enters at the latest of the three near faces.
    """
    rows, cols = np.divmod(cell_ids, grid.cols)
    east, north = grid.cell_center(rows, cols)
    half = grid.cell_size_m / 2
    bottom = np.full(east.shape, terrain.elevation.min() - TILE_DEPTH_M)
    low = np.stack([east - half, north - half, bottom], axis=1)
    high = np.stack([east + half, north + half, terrain.elevation[rows, cols]], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        to_low, to_high = (low - start) / rays, (high - start) / rays
    return np.nanmax(np.minimum(to_low, to_high), axis=1)
