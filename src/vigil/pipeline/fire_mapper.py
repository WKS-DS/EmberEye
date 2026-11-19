"""
fire_mapper.py — A ground map of where fire has been seen, built up frame by frame.

Two steps, in pipeline order (Detector -> GeoRef -> Temporal Filter -> Fire Mapper):

1. frame_to_grids: place every pixel of one frame on the ground with
   GeoRef and count, for each grid cell, how many pixels show it and how
   many of those show fire. That gives the frame's `in_view` and
   `burning` grids.
2. FireMapper.update: add one frame's grids to the map. The temporal
   filter will sit between the two steps; until it exists, each frame's
   `burning` grid goes straight in.

A cell counts as in view only if at least MIN_PIXELS_PER_CELL pixels show
it, and as burning only if at least that many of them show fire. A
far-away cell that shrinks to about a pixel can't be judged, so it counts
as unseen rather than as seen without fire. And one stray fire pixel at
the edge of a fire, placed in the neighbouring cell, doesn't set that
cell on fire: on sim flights, "any fire pixel" marked 8-9 such cells per
flight, "at least 4" marked 0-2, and both missed the same cells.

Every pixel is georeferenced, not a sample: GeoRef takes 6-33 ms for a
whole 320 x 240 frame (docs/ARCHITECTURE.md, GeoRef). Doing it per pixel
also works the same on flat ground and on terrain, where a straight edge
in the image lands on a curve on the ground.
"""

from __future__ import annotations

import numpy as np

from vigil.geo import GroundGrid
from vigil.models.types import Frame
from vigil.pipeline.georef import GeoRef

# A cell smaller than about 2 x 2 pixels in the frame counts as unseen, and
# a cell needs that many fire pixels to count as burning.
MIN_PIXELS_PER_CELL = 4


def frame_to_grids(
    georef: GeoRef,
    grid: GroundGrid,
    frame: Frame,
    fire_mask: np.ndarray,
    min_pixels: int = MIN_PIXELS_PER_CELL,
) -> tuple[np.ndarray, np.ndarray]:
    """The (rows, cols) True/False grids `in_view` and `burning` for one frame.

    fire_mask: (H, W) True where the pixel shows fire, for example the
    detector's fire outlines filled in with masks.polygons_to_mask.

    in_view: at least `min_pixels` pixels show the cell.
    burning: at least `min_pixels` of those pixels show fire.
    """
    height, width = frame.image.shape[:2]
    if fire_mask.shape != (height, width):
        raise ValueError(f"The fire mask is {fire_mask.shape} but the frame is {(height, width)}.")

    ys, xs = np.mgrid[0:height, 0:width]
    points = georef.ground_points(frame, xs, ys)
    found = np.isfinite(points[..., 0])  # pixels that show the ground at all
    rows, cols, on_grid = grid.cells_at(points[..., 0][found], points[..., 1][found])
    cell = rows[on_grid] * grid.cols + cols[on_grid]
    shows_fire = fire_mask[found][on_grid]

    n_cells = grid.rows * grid.cols
    pixels = np.bincount(cell, minlength=n_cells).reshape(grid.shape)
    fire_pixels = np.bincount(cell[shows_fire], minlength=n_cells).reshape(grid.shape)
    return pixels >= min_pixels, fire_pixels >= min_pixels


class FireMapper:
    """Everywhere fire has been seen, and everywhere that has been looked at.

    mapper = FireMapper(scenario.grid)
    for each frame:
        in_view, burning = frame_to_grids(georef, mapper.grid, frame, fire_mask)
        mapper.update(in_view, burning, time_min)
    mapper.ever_burned()  # where Stage 2 starts from
    mapper.covered        # where the map can be trusted
    """

    def __init__(self, grid: GroundGrid) -> None:
        self.grid = grid
        # When fire was first seen in each cell, in minutes on the caller's
        # clock. NaN means never: 0 can't mean never, or a fire seen at
        # time 0 would vanish.
        self.first_fire_min = np.full(grid.shape, np.nan)
        self.covered = np.zeros(grid.shape, dtype=bool)  # in view at least once

    def update(self, in_view: np.ndarray, burning: np.ndarray, time_min: float) -> None:
        """Add one frame's (rows, cols) True/False grids, seen at `time_min`."""
        if in_view.shape != self.grid.shape or burning.shape != self.grid.shape:
            raise ValueError(f"Both grids must be {self.grid.shape}, the shape of the map.")
        self.covered |= in_view
        newly_burning = burning & in_view & np.isnan(self.first_fire_min)
        self.first_fire_min[newly_burning] = time_min

    def ever_burned(self) -> np.ndarray:
        """True for every cell where fire has been seen at least once."""
        return ~np.isnan(self.first_fire_min)
