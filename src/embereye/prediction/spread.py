"""
spread.py — Fire spread as "when does the fire arrive at each cell?"

This is the Spread Predictor from docs/ARCHITECTURE.md (Stage 2). It is
used twice in the project:

- as the PREDICTOR: given where the fire is now, where will it be later?
- as the simulator's GROUND TRUTH: the "real" fire that the simulated
  drone flies over.

How it works
------------
Instead of stepping the fire forward a few minutes at a time, we work out
an ARRIVAL TIME for every cell: the number of minutes after ignition at
which the fire front reaches it. The fire's footprint at any moment is
then simply "all cells whose arrival time has passed".

Arrival times come from a shortest-path search (Dijkstra's algorithm).
Think of the grid as a road map where driving from one cell to its
neighbour takes `distance / spread rate` minutes. The fire reaches every
cell by its quickest route, which is exactly what a shortest-path search
finds. This has no minimum or maximum speed built in, unlike a
step-by-step simulation.

The physics is deliberately simple (it is NOT the full Rothermel model):
a base spread rate per fuel type, multiplied by factors for wind, slope
and fuel moisture.

Known limitation: fire can only move to the 8 neighbouring cells, so a
fire on flat ground with no wind grows as an octagon, not a circle.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from embereye.prediction.terrain import Terrain

# ---------------------------------------------------------------------------
# Fuel
# ---------------------------------------------------------------------------
NON_BURNABLE = 99   # water, roads, bare rock — fire cannot enter these cells

# Fuel types (numbered like the Anderson 13 fuel models) and their base rate
# of spread in metres per minute on flat ground with no wind.
#
# NOTE: these rates are the placeholder values from docs/ARCHITECTURE.md.
# They have not been calibrated against published fuel-model data, so treat
# real-world predictions made with them as illustrative only.
FUEL_MODELS = {
    1:  {"name": "short_grass",       "ros_ref": 2.5, "heat_kj_m2": 700},
    2:  {"name": "timber_grass",      "ros_ref": 1.8, "heat_kj_m2": 900},
    3:  {"name": "tall_grass",        "ros_ref": 4.5, "heat_kj_m2": 500},
    4:  {"name": "chaparral",         "ros_ref": 4.0, "heat_kj_m2": 2500},
    5:  {"name": "brush",             "ros_ref": 1.0, "heat_kj_m2": 700},
    6:  {"name": "dormant_brush",     "ros_ref": 1.5, "heat_kj_m2": 600},
    7:  {"name": "southern_rough",    "ros_ref": 1.2, "heat_kj_m2": 800},
    8:  {"name": "timber_litter",     "ros_ref": 0.3, "heat_kj_m2": 400},
    9:  {"name": "hardwood_litter",   "ros_ref": 0.5, "heat_kj_m2": 500},
    10: {"name": "timber_understory", "ros_ref": 2.0, "heat_kj_m2": 2800},
    11: {"name": "light_slash",       "ros_ref": 1.5, "heat_kj_m2": 1200},
    12: {"name": "medium_slash",      "ros_ref": 2.5, "heat_kj_m2": 1800},
    13: {"name": "heavy_slash",       "ros_ref": 3.5, "heat_kj_m2": 2500},
    NON_BURNABLE: {"name": "non_burnable", "ros_ref": 0.0, "heat_kj_m2": 0},
}

MPH_TO_MS = 0.447       # miles per hour -> metres per second
MIN_ROS_M_PER_MIN = 0.01  # burnable fuel never drops below this spread rate

# The 8 neighbours of a cell, as (row step, column step). Rows count
# southward, so a row step of -1 is a move NORTH.
NEIGHBORS = [
    (-1, 0), (1, 0), (0, -1), (0, 1),       # N, S, W, E
    (-1, -1), (-1, 1), (1, -1), (1, 1),     # NW, NE, SW, SE
]


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Weather:
    """The weather inputs the spread model uses."""

    wind_speed_mph:    float
    wind_to_deg:       float            # compass direction the wind blows TOWARDS (it pushes fire this way)
    fuel_moisture_pct: float = 8.0      # how damp the dead vegetation is; 8% is the reference "dry" value

    @property
    def wind_from_deg(self) -> float:
        """Where the wind comes FROM — the way weather forecasts report it."""
        return (self.wind_to_deg + 180.0) % 360.0


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class SpreadPredictor:
    """Arrival-time fire spread on a grid.

    Typical use, starting from what the Fire Mapper has seen burning
    `now_min` minutes into the incident:

        predictor = SpreadPredictor(terrain, fuel_codes, weather)
        predictor.set_ignition(burning_cells, start_time_min=now_min)
        forecast = predictor.predict([1, 3, 6])   # 1, 3 and 6 hours after now_min

    All times are minutes on one clock. `arrival_time` and `burned_at()`
    use that clock directly; `predict()` counts its horizons from "now".
    """

    def __init__(self, terrain: Terrain, fuel_codes: np.ndarray, weather: Weather) -> None:
        self.terrain     = terrain
        self.fuel_codes  = np.asarray(fuel_codes, dtype=int)
        self.weather     = weather
        self.cell_size_m = terrain.cell_size_m

        if self.fuel_codes.shape != terrain.shape:
            raise ValueError(
                f"fuel_codes shape {self.fuel_codes.shape} does not match terrain shape {terrain.shape}"
            )
        self.rows, self.cols = terrain.shape

        # Base spread rate for every cell, looked up from its fuel code.
        unknown = sorted(set(np.unique(self.fuel_codes).tolist()) - set(FUEL_MODELS))
        if unknown:
            raise ValueError(f"Unknown fuel codes: {unknown}")
        self.base_ros = np.zeros(terrain.shape)
        for code, fuel in FUEL_MODELS.items():
            self.base_ros[self.fuel_codes == code] = fuel["ros_ref"]
        self.burnable = self.base_ros > 0

        # Minutes after ignition at which fire reaches each cell.
        # Infinity means "not reached".
        self.arrival_time = np.full(terrain.shape, np.inf)

        # The latest start_time_min passed to set_ignition(), i.e. the
        # moment the fire was last observed. predict() measures from here.
        # None until something has been ignited.
        self.latest_ignition_min: Optional[float] = None

        # Wind, slope and fuel do not change during a run, so the time to
        # cross into every cell from each of the 8 directions can be worked
        # out once, up front. _travel_time[k][row, col] is the minutes it
        # takes fire to ENTER (row, col) when moving in direction NEIGHBORS[k].
        self._travel_time = []
        for d_row, d_col in NEIGHBORS:
            is_diagonal = d_row != 0 and d_col != 0
            distance_m = self.cell_size_m * (math.sqrt(2) if is_diagonal else 1.0)
            ros = self.spread_rate(d_row, d_col)
            travel = np.full(terrain.shape, np.inf)
            travel[self.burnable] = distance_m / ros[self.burnable]
            self._travel_time.append(travel)

    # ------------------------------------------------------------- spread rate
    def spread_rate(self, d_row: int, d_col: int) -> np.ndarray:
        """Spread rate (m/min) for fire entering each cell while moving in
        the grid direction (d_row, d_col). Non-burnable cells get 0.
        """
        # Compass bearing of this move. -d_row because rows count southward.
        spread_dir = math.degrees(math.atan2(d_col, -d_row)) % 360.0

        # --- Wind: one number, the same for every cell.
        # alignment is +1 when spreading straight downwind, -1 straight
        # upwind, 0 when spreading across the wind.
        wind_alignment = math.cos(math.radians(self.weather.wind_to_deg - spread_dir))
        wind_ms = self.weather.wind_speed_mph * MPH_TO_MS
        if wind_alignment > 0:
            wind_factor = 1 + 0.5 * (wind_ms ** 1.2) * wind_alignment
        else:
            wind_factor = 1 / (1 + 0.3 * wind_ms * abs(wind_alignment))

        # --- Slope: different for every cell.
        # Aspect points downhill, so spreading opposite to it is uphill.
        tan_slope = np.tan(np.radians(self.terrain.slope))
        uphill_alignment = np.cos(np.radians(self.terrain.aspect - spread_dir + 180.0))
        uphill_factor   = 1 + 5.275 * tan_slope ** 2 * uphill_alignment
        downhill_factor = 1 / (1 + 2.0 * tan_slope * np.abs(uphill_alignment))
        slope_factor = np.where(uphill_alignment > 0, uphill_factor, downhill_factor)

        # --- Moisture: damper fuel burns slower. 1.0 at the 8% reference.
        moisture_factor = max(0.1, 1 - (self.weather.fuel_moisture_pct - 8) / 25)

        ros = self.base_ros * wind_factor * slope_factor * moisture_factor
        return np.where(self.burnable, np.maximum(MIN_ROS_M_PER_MIN, ros), 0.0)

    # --------------------------------------------------------------- ignition
    def reset(self) -> None:
        """Forget all ignitions and arrival times."""
        self.arrival_time[:] = np.inf
        self.latest_ignition_min = None

    def set_ignition(self, burning_cells: Iterable[Sequence[int]], start_time_min: float = 0.0) -> int:
        """Mark cells as already on fire at `start_time_min`.

        Cells that are off the grid or non-burnable are skipped. Returns
        how many cells were actually ignited, so the caller can check.
        """
        ignited = 0
        for row, col in burning_cells:
            row, col = int(row), int(col)
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                continue
            if not self.burnable[row, col]:
                continue
            self.arrival_time[row, col] = min(self.arrival_time[row, col], start_time_min)
            ignited += 1

        if ignited > 0 and math.isfinite(start_time_min):
            if self.latest_ignition_min is None or start_time_min > self.latest_ignition_min:
                self.latest_ignition_min = start_time_min
        return ignited

    # ------------------------------------------------------------ propagation
    def propagate(self) -> np.ndarray:
        """Fill in the arrival time of every cell the fire can reach.

        Dijkstra's algorithm: keep a to-do list ordered by arrival time,
        always process the earliest cell next, and offer each neighbour a
        new arrival time via this cell. If that beats the neighbour's
        current best, record it and add the neighbour to the list.
        """
        arrival = self.arrival_time

        # heapq keeps a list ordered so that the smallest item pops first.
        # Each item is (arrival_time, row, col).
        queue = [(float(arrival[row, col]), int(row), int(col))
                 for row, col in np.argwhere(np.isfinite(arrival))]
        heapq.heapify(queue)

        while queue:
            time_here, row, col = heapq.heappop(queue)

            # A cell can be queued several times; skip stale entries where a
            # quicker route was found after this one was added.
            if time_here > arrival[row, col]:
                continue

            for k, (d_row, d_col) in enumerate(NEIGHBORS):
                n_row, n_col = row + d_row, col + d_col
                if not (0 <= n_row < self.rows and 0 <= n_col < self.cols):
                    continue

                new_arrival = time_here + self._travel_time[k][n_row, n_col]
                if new_arrival < arrival[n_row, n_col]:
                    arrival[n_row, n_col] = new_arrival
                    heapq.heappush(queue, (new_arrival, n_row, n_col))

        return arrival

    # ----------------------------------------------------------------- output
    def burned_at(self, time_min: float) -> np.ndarray:
        """True/False grid: has the fire reached each cell by `time_min`?"""
        return self.arrival_time <= time_min

    def predict(
        self,
        hours_list: Sequence[float] = (1, 3, 6),
        now_min: Optional[float] = None,
    ) -> dict[str, np.ndarray]:
        """Burned-area grids at several times ahead, e.g. {"1hr": ..., "3hr": ...}.

        Every horizon is measured from `now_min`, so "1hr" means one hour
        after now. If `now_min` is not given, "now" is the latest ignition
        time passed to set_ignition() (0 for a fire ignited at time zero).
        """
        if now_min is None:
            now_min = self.latest_ignition_min if self.latest_ignition_min is not None else 0.0
        self.propagate()
        return {f"{hours:g}hr": self.burned_at(now_min + hours * 60) for hours in hours_list}
