"""
terrain.py — Ground elevation, slope and aspect on a grid.

This is the "grid of numbers" half of the Terrain Analyzer in
docs/ARCHITECTURE.md. It works on any elevation array: a made-up one for
the simulator today, a real USGS elevation file later (loading those needs
`rasterio` and will be added alongside this class, not instead of it).

Two derived quantities matter for fire:
- slope:  how steep the ground is, in degrees (0 = flat).
- aspect: the compass direction the ground faces DOWNHILL. A slope that
          drops away to the east has aspect 90.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Terrain:
    """Elevation for every grid cell, plus the slope and aspect worked out from it."""

    elevation:   np.ndarray     # (rows, cols) metres above the local datum; row 0 = north edge
    cell_size_m: float

    # Filled in by __post_init__ — `init=False` means "not a constructor argument".
    slope:  np.ndarray = field(init=False)   # degrees
    aspect: np.ndarray = field(init=False)   # degrees clockwise from north, pointing downhill

    def __post_init__(self) -> None:
        self.elevation = np.asarray(self.elevation, dtype=float)

        # np.gradient gives the change in elevation per metre along each
        # array axis: first down the rows, then across the columns.
        d_row, d_col = np.gradient(self.elevation, self.cell_size_m)

        # Columns count eastward, so d_col is already "per metre east".
        # Rows count SOUTHward, so "per metre north" is the negative.
        d_east  = d_col
        d_north = -d_row

        self.slope = np.degrees(np.arctan(np.hypot(d_east, d_north)))
        # The gradient points uphill; downhill is the opposite direction.
        # arctan2(east, north) turns a direction into a compass bearing.
        self.aspect = np.degrees(np.arctan2(-d_east, -d_north)) % 360.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    # ------------------------------------------------------------ made-up terrain
    @classmethod
    def flat(cls, rows: int, cols: int, cell_size_m: float, elevation_m: float = 0.0) -> "Terrain":
        """Perfectly level ground."""
        return cls(np.full((rows, cols), elevation_m), cell_size_m)

    @classmethod
    def ramp(cls, rows: int, cols: int, cell_size_m: float, slope_deg: float, uphill_deg: float) -> "Terrain":
        """One even slope across the whole grid, rising towards `uphill_deg`.

        Useful for tests: every cell has the same slope and aspect.
        """
        row_idx, col_idx = np.indices((rows, cols))
        east  = col_idx * cell_size_m
        north = -row_idx * cell_size_m
        # Distance travelled in the uphill direction, times rise per metre.
        along = east * np.sin(np.radians(uphill_deg)) + north * np.cos(np.radians(uphill_deg))
        elevation = along * np.tan(np.radians(slope_deg))
        return cls(elevation - elevation.min(), cell_size_m)

    @classmethod
    def hill(
        cls,
        rows: int,
        cols: int,
        cell_size_m: float,
        height_m: float = 40.0,
        radius_m: Optional[float] = None,
        center_row: Optional[float] = None,
        center_col: Optional[float] = None,
    ) -> "Terrain":
        """A single smooth, rounded hill.

        `radius_m` is roughly where the hill has dropped to 60% of its
        height. Defaults put the hill in the middle of the grid.
        """
        if radius_m is None:
            radius_m = 0.2 * min(rows, cols) * cell_size_m
        if center_row is None:
            center_row = (rows - 1) / 2
        if center_col is None:
            center_col = (cols - 1) / 2

        row_idx, col_idx = np.indices((rows, cols))
        dist_m = np.hypot(row_idx - center_row, col_idx - center_col) * cell_size_m
        elevation = height_m * np.exp(-0.5 * (dist_m / radius_m) ** 2)
        return cls(elevation, cell_size_m)
