"""
fire_sim.py — A simulated fire with a simulated drone flying over it.

FireSim ties the pieces together:

    SpreadPredictor (the "real" fire)  ->  which cells are burning at time t
    FireWorld       (the MuJoCo scene) ->  what the camera sees
    CameraPath      (the scripted flight) -> where the camera is

For each frame it hands back a SimObservation holding the picture, a mask
of the fire pixels, the exact camera pose, and the true state of the
ground. Because the simulator knows the truth, anything built on top
(georeferencing, fire mapping, prediction) can be scored against it.

There is no machine learning here. "Detection" is read straight off
MuJoCo's segmentation render: a pixel is fire if the tile it shows is
burning. That keeps the detector out of the picture while the geometry
and mapping are being tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

import numpy as np

from embereye.geo import GroundGrid, LocalFrame
from embereye.masks import mask_to_polygons  # also importable from here, as before
from embereye.prediction.spread import FUEL_MODELS, NON_BURNABLE, SpreadPredictor, Weather
from embereye.prediction.terrain import Terrain
from embereye.sim.camera import CameraModel, CameraPath, CameraPose, LawnmowerPath, OrbitPath
from embereye.sim.world import FireWorld

# Values used in the truth grid (SimObservation.truth_state).
UNBURNED = 0
BURNING  = 1
BURNED   = 2    # the fire has passed and the cell has burned out

# Tile colours, as (red, green, blue) from 0 to 1.
FLAME_START_RGB = np.array([1.00, 0.80, 0.15])   # just ignited: bright yellow-orange
FLAME_END_RGB   = np.array([0.75, 0.10, 0.02])   # about to burn out: deep red
ASH_RGB         = np.array([0.10, 0.09, 0.08])
ROAD_RGB        = np.array([0.50, 0.50, 0.50])
FUEL_RGB = {                                     # by fuel name, see prediction/spread.py
    "short_grass":       (0.62, 0.66, 0.30),
    "timber_grass":      (0.50, 0.60, 0.28),
    "tall_grass":        (0.70, 0.68, 0.32),
    "chaparral":         (0.42, 0.45, 0.22),
    "brush":             (0.45, 0.50, 0.25),
    "dormant_brush":     (0.55, 0.50, 0.30),
    "southern_rough":    (0.38, 0.48, 0.24),
    "timber_litter":     (0.20, 0.35, 0.18),
    "hardwood_litter":   (0.30, 0.40, 0.20),
    "timber_understory": (0.14, 0.30, 0.15),
    "light_slash":       (0.50, 0.42, 0.28),
    "medium_slash":      (0.45, 0.38, 0.25),
    "heavy_slash":       (0.40, 0.33, 0.22),
}

# A fixed start time so that the same scenario always produces the same
# timestamps. Pass start_time= to FireSim to change it.
DEFAULT_START_TIME = datetime(2026, 8, 1, 21, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Scenario — everything that defines the simulated world
# ---------------------------------------------------------------------------
@dataclass
class Scenario:
    """The world and the fire in it."""

    grid:           GroundGrid
    terrain:        Terrain
    fuel_codes:     np.ndarray              # (rows, cols) fuel type per cell
    weather:        Weather
    ignition_cells: list[tuple[int, int]]   # (row, col) cells where the fire starts
    local_frame:    LocalFrame              # where on Earth the grid's centre sits

    fire_age_at_start_min: float = 20.0     # how long the fire has been burning when the flight begins
    burn_duration_min:     float = 12.0     # how long a cell stays alight before burning out


# ---------------------------------------------------------------------------
# Observation — what the sim hands back for each frame
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SimObservation:
    """One frame from the simulated flight, with the answer key attached."""

    index:          int                 # frame number, starting at 0
    time_s:         float               # seconds since the flight started
    fire_time_min:  float               # minutes since the fire ignited
    ts:             datetime            # UTC timestamp of this frame

    rgb:            np.ndarray          # (H, W, 3) uint8 — what the camera sees
    cell_ids:       np.ndarray          # (H, W) int32 — ground cell shown by each pixel (row * cols + col), -1 = none
    fire_mask:      np.ndarray          # (H, W) bool — True where the pixel shows a burning cell
    fire_polygons:  list[np.ndarray]    # outline of each fire region, (N, 2) arrays of pixel (x, y)

    pose:           CameraPose          # exact camera position and orientation
    truth_state:    np.ndarray          # (rows, cols) uint8 — UNBURNED / BURNING / BURNED for every cell


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------
class FireSim:
    """Simulated fire + simulated camera flight.

    Typical use:

        with FireSim(demo_scenario(), OrbitPath()) as sim:
            for obs in sim.run(duration_s=600, frame_interval_s=5):
                ...   # obs.rgb, obs.fire_polygons, obs.pose, obs.truth_state
    """

    def __init__(
        self,
        scenario: Scenario,
        path: CameraPath,
        camera: Optional[CameraModel] = None,
        start_time: datetime = DEFAULT_START_TIME,
        seed: int = 0,
    ) -> None:
        self.scenario   = scenario
        self.path       = path
        self.camera     = camera if camera is not None else CameraModel()
        self.start_time = start_time

        # The "real" fire: work out once when it reaches every cell.
        self.truth = SpreadPredictor(scenario.terrain, scenario.fuel_codes, scenario.weather)
        ignited = self.truth.set_ignition(scenario.ignition_cells)
        if ignited == 0:
            raise ValueError("No ignition cell is on the grid and burnable, so there is no fire to simulate.")
        self.truth.propagate()

        self.world = FireWorld(scenario.grid, scenario.terrain, self.camera)
        self._ground_colors = self._make_ground_colors(seed)

    # Support `with FireSim(...) as sim:` so the renderer is always released.
    def __enter__(self) -> "FireSim":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        self.world.close()

    # ------------------------------------------------------------------- fire
    def fire_time_min(self, time_s: float) -> float:
        """Convert flight time (seconds) to fire time (minutes since ignition)."""
        return self.scenario.fire_age_at_start_min + time_s / 60.0

    def fire_state(self, time_s: float) -> np.ndarray:
        """The true state of every cell at this moment of the flight."""
        now = self.fire_time_min(time_s)
        arrival = self.truth.arrival_time

        state = np.full(arrival.shape, UNBURNED, dtype=np.uint8)
        state[arrival <= now] = BURNING
        state[arrival + self.scenario.burn_duration_min <= now] = BURNED
        return state

    # ----------------------------------------------------------------- colours
    def _make_ground_colors(self, seed: int) -> np.ndarray:
        """Colour of every tile before any fire, as (rows, cols, 3)."""
        fuel_codes = self.scenario.fuel_codes
        colors = np.zeros(fuel_codes.shape + (3,))
        for code, fuel in FUEL_MODELS.items():
            rgb = ROAD_RGB if code == NON_BURNABLE else FUEL_RGB[fuel["name"]]
            colors[fuel_codes == code] = rgb

        # Slight random brightness per tile so the ground is not one flat
        # colour. A fixed seed makes it the same on every run.
        rng = np.random.default_rng(seed)
        brightness = rng.uniform(0.92, 1.08, size=fuel_codes.shape)
        return np.clip(colors * brightness[..., None], 0.0, 1.0)

    def tile_colors(self, state: np.ndarray, time_s: float) -> np.ndarray:
        """Colour of every tile given the fire state."""
        colors = self._ground_colors.copy()

        # Burning tiles fade from bright yellow to deep red as they burn down.
        burning = state == BURNING
        burning_for = self.fire_time_min(time_s) - self.truth.arrival_time[burning]
        progress = np.clip(burning_for / self.scenario.burn_duration_min, 0.0, 1.0)[:, None]
        colors[burning] = FLAME_START_RGB * (1 - progress) + FLAME_END_RGB * progress

        colors[state == BURNED] = ASH_RGB
        return colors

    # -------------------------------------------------------------------- pose
    def _ground_up_at(self, east_m: float, north_m: float) -> float:
        """Height of the ground under a point (lowest ground level beyond the grid)."""
        cell = self.scenario.grid.cell_at(east_m, north_m)
        if cell is None:
            return float(self.scenario.terrain.elevation.min())
        return float(self.scenario.terrain.elevation[cell])

    def pose_at(self, time_s: float) -> CameraPose:
        """Place the camera where the path says and return its exact pose."""
        point = self.path.at(time_s)
        position, rotation = self.world.move_camera(
            point.east_m, point.north_m, point.up_m, point.heading_deg, point.pitch_deg
        )
        east, north, up = (float(v) for v in position)
        lat, lon = self.scenario.local_frame.to_gps(east, north)

        return CameraPose(
            east_m=east,
            north_m=north,
            up_m=up,
            heading_deg=point.heading_deg,
            pitch_deg=point.pitch_deg,
            rotation=rotation,
            lat=float(lat),
            lon=float(lon),
            alt_msl_m=self.scenario.local_frame.origin_alt_m + up,
            alt_agl_m=up - self._ground_up_at(east, north),
        )

    # ----------------------------------------------------------------- frames
    def observe(self, time_s: float, index: int = 0) -> SimObservation:
        """Produce one frame: what the camera sees `time_s` into the flight."""
        state = self.fire_state(time_s)
        self.world.set_tile_colors(self.tile_colors(state, time_s))

        pose = self.pose_at(time_s)
        rgb, cell_ids = self.world.render()

        # A pixel is fire if the cell it shows is burning. `ravel()` flattens
        # the grid so it can be indexed by the flat cell index.
        burning_flat = (state == BURNING).ravel()
        on_grid = cell_ids >= 0
        fire_mask = np.zeros(cell_ids.shape, dtype=bool)
        fire_mask[on_grid] = burning_flat[cell_ids[on_grid]]

        return SimObservation(
            index=index,
            time_s=time_s,
            fire_time_min=self.fire_time_min(time_s),
            ts=self.start_time + timedelta(seconds=time_s),
            rgb=rgb,
            cell_ids=cell_ids,
            fire_mask=fire_mask,
            fire_polygons=mask_to_polygons(fire_mask),
            pose=pose,
            truth_state=state,
        )

    def run(self, duration_s: float, frame_interval_s: float = 1.0) -> Iterator[SimObservation]:
        """Fly the path, yielding one observation every `frame_interval_s`
        seconds of simulated time, from 0 up to and including `duration_s`.

        This is a generator: frames are produced one at a time as the
        caller's loop asks for them, so a long flight never has to fit in
        memory all at once.
        """
        n_frames = int(duration_s // frame_interval_s) + 1
        for index in range(n_frames):
            yield self.observe(index * frame_interval_s, index)


# ---------------------------------------------------------------------------
# Ready-made scenarios and flight paths
# ---------------------------------------------------------------------------
def demo_scenario(terrain_kind: str = "flat", rows: int = 80, cols: int = 80, cell_size_m: float = 10.0) -> Scenario:
    """A small grass fire pushed north-east by the wind.

    The ground is mostly short grass, with a block of timber to the north
    and a road (non-burnable) across the fire's path. `terrain_kind` is
    "flat" or "hill".
    """
    grid = GroundGrid(rows, cols, cell_size_m)

    if terrain_kind == "flat":
        terrain = Terrain.flat(rows, cols, cell_size_m)
    elif terrain_kind == "hill":
        terrain = Terrain.hill(rows, cols, cell_size_m, height_m=40.0)
    else:
        raise ValueError(f"Unknown terrain_kind {terrain_kind!r}; use 'flat' or 'hill'.")

    # Positions are given as fractions of the grid so the layout scales
    # with the grid size.
    def at(fraction: float, size: int) -> int:
        return int(round(fraction * (size - 1)))

    fuel_codes = np.full((rows, cols), 1)                                           # short grass everywhere
    fuel_codes[at(0.08, rows):at(0.35, rows), at(0.30, cols):at(0.75, cols)] = 10   # timber block to the north
    fuel_codes[at(0.25, rows):at(0.85, rows), at(0.65, cols)] = NON_BURNABLE        # a north-south road

    return Scenario(
        grid=grid,
        terrain=terrain,
        fuel_codes=fuel_codes,
        weather=Weather(wind_speed_mph=8.0, wind_to_deg=60.0, fuel_moisture_pct=8.0),
        ignition_cells=[(at(0.65, rows), at(0.33, cols))],
        local_frame=LocalFrame(origin_lat=47.6588, origin_lon=-117.4260, origin_alt_m=600.0),
    )


def hard_scenario(rows: int = 160, cols: int = 160, cell_size_m: float = 10.0, hill_height_m: float = 90.0) -> Scenario:
    """A harder test than the demo: more than one frame can see at once.

    - The grid is large (1.6 km square by default), so the fire never fits
      in a single frame and the map has to be built up over time.
    - Two separate ignitions, so there are several fire regions.
    - A tall hill between them hides part of the fire whenever the camera
      is on the far side of it, so some burning cells go unseen.

    Positions are fractions of the grid, as in demo_scenario, so a smaller
    grid (e.g. 60 x 60 in tests) has the same layout.
    """
    grid = GroundGrid(rows, cols, cell_size_m)

    def at(fraction: float, size: int) -> int:
        return int(round(fraction * (size - 1)))

    terrain = Terrain.hill(
        rows, cols, cell_size_m,
        height_m=hill_height_m,
        radius_m=0.08 * min(rows, cols) * cell_size_m,
        center_row=at(0.50, rows),
        center_col=at(0.48, cols),
    )

    fuel_codes = np.full((rows, cols), 1)                                           # short grass everywhere
    fuel_codes[at(0.10, rows):at(0.30, rows), at(0.55, cols):at(0.85, cols)] = 10   # timber block to the north-east
    fuel_codes[at(0.60, rows):at(0.90, rows), at(0.62, cols)] = NON_BURNABLE        # a north-south road in the south-east

    return Scenario(
        grid=grid,
        terrain=terrain,
        fuel_codes=fuel_codes,
        weather=Weather(wind_speed_mph=8.0, wind_to_deg=60.0, fuel_moisture_pct=8.0),
        ignition_cells=[(at(0.68, rows), at(0.30, cols)),      # south-west of the hill
                        (at(0.42, rows), at(0.66, cols))],     # north-east of the hill
        local_frame=LocalFrame(origin_lat=47.6588, origin_lon=-117.4260, origin_alt_m=600.0),
        fire_age_at_start_min=25.0,
    )


SCENARIOS = {"demo": demo_scenario, "hard": hard_scenario}
PATHS = ("orbit", "lawnmower")


def make_path(kind: str, grid: GroundGrid) -> CameraPath:
    """A flight path sized to the grid.

    - "orbit": circles the grid centre at a radius of 350 m per 800 m of
      grid, so on the demo's 800 m grid it is exactly OrbitPath()'s default.
    - "lawnmower": passes 300 m apart over the middle 80% of the grid.
    """
    extent = max(grid.width_m, grid.height_m)
    if kind == "orbit":
        return OrbitPath(radius_m=350.0 * extent / 800.0)
    if kind == "lawnmower":
        width = 0.8 * extent
        spacing = 300.0
        return LawnmowerPath(width_m=width, spacing_m=spacing, n_passes=int(width // spacing) + 1)
    raise ValueError(f"Unknown path {kind!r}; use one of {PATHS}.")


def visible_burning_fraction(obs: SimObservation) -> float:
    """Share of the burning cells that at least one pixel of the frame shows.

    Cells can be missed because they are outside the frame or hidden behind
    terrain. 1.0 when nothing is burning.
    """
    burning = obs.truth_state == BURNING
    n_burning = int(burning.sum())
    if n_burning == 0:
        return 1.0
    shown = np.unique(obs.cell_ids[obs.fire_mask])
    return len(shown) / n_burning
