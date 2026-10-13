"""
geo.py — Shared ground coordinates.

Two small tools that every geometry-aware part of EmberEye agrees on:

- LocalFrame: converts between GPS (lat/lon) and "local metres" — how far
  east and north a point is from a chosen origin. All the geometry is done
  in metres; GPS is just the format positions are stored and reported in.

- GroundGrid: a rectangle of square cells laid over the ground, centred on
  that origin. The simulator, the spread predictor and (later) the Fire
  Mapper all index the ground with the same grid, so "cell (12, 40)" means
  the same patch of ground everywhere.

Conventions used throughout the project:
- Local coordinates are (east, north, up) in metres.
- Compass angles are degrees clockwise from north: 0 = N, 90 = E.
- Grid row 0 is the NORTH edge and column 0 is the WEST edge, the same
  way a map or a satellite image is stored. Rows count southward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pyproj import Transformer


def utm_epsg_for(lat: float, lon: float) -> int:
    """Return the EPSG code of the UTM zone that contains this lat/lon.

    UTM splits the Earth into 60 north-south strips ("zones"), each 6
    degrees of longitude wide. Inside a zone, positions are plain metres,
    which is what we want. Picking the zone from the longitude (instead of
    hard-coding one) keeps the maths accurate wherever the drone flies:
    Seattle is zone 10, Spokane is zone 11.
    """
    zone = int((lon + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    north_hemisphere = lat >= 0
    return (32600 if north_hemisphere else 32700) + zone


class LocalFrame:
    """Converts between GPS and metres east/north of an origin point.

    Note: "north" here is UTM grid north. It differs from true north by a
    fraction of a degree near the middle of a zone, which is far below
    anything else we measure.
    """

    def __init__(self, origin_lat: float, origin_lon: float, origin_alt_m: float = 0.0) -> None:
        self.origin_lat   = origin_lat
        self.origin_lon   = origin_lon
        self.origin_alt_m = origin_alt_m    # metres above sea level at local up = 0

        self.epsg = utm_epsg_for(origin_lat, origin_lon)
        # always_xy=True means pyproj takes and returns (lon, lat) / (x, y)
        # in that order, regardless of what the coordinate system prefers.
        self._to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self._origin_x, self._origin_y = self._to_utm.transform(origin_lon, origin_lat)

    def to_local(self, lat, lon):
        """GPS -> (east_m, north_m). Accepts single numbers or numpy arrays."""
        x, y = self._to_utm.transform(lon, lat)
        return x - self._origin_x, y - self._origin_y

    def to_gps(self, east_m, north_m):
        """(east_m, north_m) -> (lat, lon). Accepts single numbers or numpy arrays."""
        lon, lat = self._to_utm.transform(
            east_m + self._origin_x,
            north_m + self._origin_y,
            direction="INVERSE",
        )
        return lat, lon


@dataclass(frozen=True, slots=True)
class GroundGrid:
    """A rows x cols grid of square cells, centred on the local origin."""

    rows:        int
    cols:        int
    cell_size_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    @property
    def width_m(self) -> float:
        """East-west size of the whole grid."""
        return self.cols * self.cell_size_m

    @property
    def height_m(self) -> float:
        """North-south size of the whole grid."""
        return self.rows * self.cell_size_m

    def cell_center(self, row, col):
        """(row, col) -> (east_m, north_m) of the middle of that cell.

        Works on single numbers or numpy arrays.
        """
        east  = (col + 0.5) * self.cell_size_m - self.width_m / 2
        north = self.height_m / 2 - (row + 0.5) * self.cell_size_m
        return east, north

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """East and north of every cell centre, each as a (rows, cols) array."""
        rows, cols = np.indices(self.shape)
        return self.cell_center(rows, cols)

    def cell_at(self, east_m: float, north_m: float) -> Optional[tuple[int, int]]:
        """Which cell contains this point? None if it is off the grid."""
        col = math.floor((east_m + self.width_m / 2) / self.cell_size_m)
        row = math.floor((self.height_m / 2 - north_m) / self.cell_size_m)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return row, col
        return None

    def cells_at(self, east_m: np.ndarray, north_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Array version of cell_at.

        Returns (rows, cols, on_grid). `on_grid` is a True/False array; the
        row and column values are only meaningful where it is True.
        """
        cols = np.floor((np.asarray(east_m) + self.width_m / 2) / self.cell_size_m).astype(int)
        rows = np.floor((self.height_m / 2 - np.asarray(north_m)) / self.cell_size_m).astype(int)
        on_grid = (rows >= 0) & (rows < self.rows) & (cols >= 0) & (cols < self.cols)
        return rows, cols, on_grid
