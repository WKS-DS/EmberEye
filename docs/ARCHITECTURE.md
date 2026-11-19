# Vigil System Architecture

## Overview

Vigil is a drone-based wildfire detection and prediction system with three stages:

- **Stage 1: Detection** — Real-time fire identification and alerting
- **Stage 2: Prediction** — Fire spread forecasting for proactive crew positioning
- **Stage 3: Suppression Support** *(Future Work)* — Predictive barrier placement to slow fire spread and buy time for responders

The **Dashboard** and **Drone** span all stages, providing the interface and sensor platform respectively.

---

## Stage 1: Detection Phase

**Pipeline order:** Detector → GeoRef → Temporal Filter → Fire Mapper → Alerter. The temporal filter works on the ground grid, so it runs after GeoRef.

### 1. Detector (YOLO11 Segmentation)

**Purpose:** Runs inference on each video frame to detect and segment fire regions. Outputs pixel masks with class labels and confidence scores. Segmentation (not detection boxes) provides the outlines needed for perimeter mapping.

**How to build:**
- Use Ultralytics YOLO11 segmentation: `pip install ultralytics`
- Fine-tune on fire/smoke dataset with segmentation masks
- Input: RGB frames (thermal-derived labels for training, RGB for inference to avoid circular evaluation)
- Output: pixel masks with class and confidence

```python
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")  # segmentation model
results = model.predict(frame, conf=0.5)

for result in results:
    if result.masks is not None:
        for mask, box in zip(result.masks.xy, result.boxes):
            polygon = mask  # list of (x, y) points
            cls = int(box.cls)  # 0=fire, 1=smoke
            conf = float(box.conf)
```

**Training data strategy:**
- Generate masks from the thermal frames with the palette-aware method in `src/vigil/data/flame2.py`. FLAME 2's thermal frames are false-colour JPEGs, not temperatures: the colour palette changes partway through the dataset and appears to be rescaled frame by frame, so no fixed pixel threshold finds fire across the whole set.
- Train the model on RGB frames using the thermal-derived masks, **after** checking that RGB and thermal pixels line up. That is unverified: on frame 13701, thermal outlines drawn on the RGB frame matched nothing visible. Many fire frames are also mostly smoke in RGB, so some fire the thermal camera sees is invisible in RGB.
- Hand-label a separate test set (~300 frames) to avoid circular evaluation
- Split FLAME 2 into contiguous blocks of frames, not at random, so near-identical neighbouring frames never land in different splits. All no-fire frames are frames 1–13700 (before the burn), so validation and test each need a block from that range. The blocks are in `src/vigil/data/flame2_split.py`: about 72% train, 10% val and 10% test, with val and test each holding every stratum, and 300 unused frames between blocks of different splits. `scripts/export_flame2_yolo.py` writes the split and the YOLO-seg labels.

**Labelling decisions (Phase 4, 2026-10-19):**
- `labels.txt` is the frame-level truth. No-fire frames always get empty label files, whatever the mask says. Fire frames get `hot_mask` with the frame gate on (`max_frame_median=0.25`). A fire frame whose mask comes out empty, gated or not, is excluded from training and listed in the split report, never written as an empty label.
- The gate's 4.4% rate on no-fire frames is **not** a false-positive figure: it was tuned and measured on inferno-palette frames of open grassland only.
- Hand-check before any export: 48 thermal frames, 8 per stratum (palette variant inferno / rainbow-red / rainbow-magenta × smoke flag YY / YN), picked by `scripts/export_handcheck.py` and scored by `scripts/score_handcheck.py`. Go: mean IoU ≥ 0.5 in every stratum. A stratum below that gets parameter tuning or has its frames dropped from training labels. Result (2026-10-21): every stratum passed, mean 0.63; held-out tuning for rainbow frames gave no clear gain, so the defaults stand (`docs/findings/flame2-label-validation.md`).
- Smoky frames (YY: smoke fills ≥50% of the RGB frame): the RGB baseline trains on YN fire frames only (14,316) plus the no-fire frames; adding YY is an ablation run. Validation and test report YY and YN as separate groups. The thermal-only experiment uses all fire frames. The split JSON records each frame's YY/YN flag and palette variant.
- Scene confound: every no-fire frame is open grassland and sky (before the burn), while fire frames are forest. Image-level false-positive rates on FLAME 2's no-fire frames are therefore not evidence that the detector avoids false alarms in forest. TODO(Will): add forest no-fire negatives from another dataset (e.g. FLAME 1's No_Fire set).
- Leakage (measured in the 4.4 export): before the burn the drone hovers over the same grassland, so 60% of sampled no-fire test frames have a near-copy in train (RGB and thermal thumbnails both correlate above 0.9). Sampled fire frames in val and test have none. Fire-frame metrics are the headline; no-fire results are optimistic.

**Hardware:**
- On-drone (edge): NVIDIA Jetson Orin Nano (~$500) — runs YOLO11 at 15-30 FPS. TODO(Will): reconcile with `docs/PARTS_LIST.md`, which budgets a Raspberry Pi 5.
- Ground station: Any GPU (RTX 3060+) for real-time, or CPU for slower processing
- Optional: Thermal camera (FLIR Lepton ~$200, or FLIR Vue for higher-end)

**Evaluation experiments:**
1. RGB-only detection (thermal-derived labels)
2. Thermal-only detection
3. RGB + thermal fusion
4. Report mAP, precision, recall by class; note smoke labels require separate annotation

---

### 2. GeoRef (Georeferencing Module)

**Purpose:** Find the point on the ground that a pixel shows, in local metres (and from there, GPS). Inputs: pixel (x, y), the camera pose from telemetry, the camera's field of view, and terrain elevation.

**Built** (`pipeline/georef.py`, `GeoRef`): flat ground, or terrain from a `Terrain` elevation grid on a `GroundGrid`. It rebuilds the camera from a frame's telemetry and field of view only. Against the cells MuJoCo rendered, it picks the right cell for 99.99% of pixels on the flat demo and the demo hill, and 99.96% on the hard scenario's 90 m hill (worst frame 99.82%, on the full-size demo hill); the rest are pixels on a cell border. Flat ground at the datum gets between 1% and 90% right on the same hill flights. It refuses frames that report gimbal roll, and takes gimbal yaw as relative to the drone's heading.

**How it works:**
1. **A ray per pixel.** Build the camera pose from telemetry (position, heading, gimbal pitch and yaw) and get the pixel's viewing direction with `CameraModel.pixel_rays(pose, x, y)` from `sim/camera.py`. This is exact pinhole-camera geometry, tested as the inverse of `CameraModel.project()`.
2. **Intersect the ray with the terrain.** Each cell's ground is flat at the cell's elevation, like the simulator's tiles. Walk the ray across the grid one cell at a time, from the edge where it enters a cell to the edge where it leaves, and stop in the first cell where it has dropped below the cell's top: it hits that top, or the cell's side if it came in below the top. Only the stretch between the highest and lowest ground level, over the grid, needs walking; a ray that meets no cell lands on the lowest level beyond the grid.
3. **Camera height from MSL altitude + DEM**, not "AGL under the drone". The ground under the drone is not the ground the camera is looking at, so a single height-above-ground number can't place an oblique view.

**Coordinates:** use `geo.LocalFrame` for GPS ↔ metres. It picks the UTM zone from the longitude (Spokane is zone 11, EPSG:32611). Never hard-code a zone.

**Why terrain matters:** on the simulator's 40 m demo hill, intersecting rays with a flat plane at the ground height under the drone misplaces points by a mean of 26–43 m per frame (95th percentile 64–124 m), measured on 21 frames of the demo orbit. On flat ground the same method is exact.

**Why not fixed steps:** stepping along the ray (say every half cell) until it is underground, then narrowing down by bisection, misses a cell whenever the ray dips below that cell's corner for less than one step. On the demo hill that put about 5% of pixels a cell too far, and it was 20–30 times slower than walking cell by cell.

Notes:
- Flat-topped cells match the simulator exactly. A real DEM will need interpolation between cells (bilinear); not built yet.
- Pixels that look level or upwards get no ground point, even if they show a hillside higher than the camera. That only happens when the drone flies lower than the terrain around it.
- In the sim, the true ground point for a pixel comes from `SimObservation.cell_ids` (which cell it shows) plus the same ray/terrain intersection. See Scoring under System-Level Validation.

**Hardware:**
- Drone with GPS module and IMU (standard)
- Gimbal with pitch/roll telemetry
- Camera with known FOV specs

---

### 3. Temporal Filter (replaces Tracker)

**Purpose:** Reduce false positives by confirming fire over several observations before it is mapped or alerted on. Standard object trackers (ByteTrack, BoT-SORT) assume separate, solid objects; fire merges and splits, and from a moving drone most apparent motion is the camera's.

**Where it runs:** on the **ground grid, after GeoRef**, not in image space. Comparing masks between frames in pixel coordinates mixes real fire change with camera motion; comparing the same ground cell across frames does not.

**Rule:** a cell is confirmed once it has been seen burning in at least N of the last M observations in which it was in view. Frames where the cell was out of view count neither way.

```python
import numpy as np


class TemporalFilter:
    """Confirm fire cell by cell on the ground grid. Input comes from GeoRef."""

    def __init__(self, grid_shape, n=3, m=5):
        self.n = n
        # The last m observations of every cell: 1 = burning, 0 = clear, -1 = none yet.
        self.recent = np.full((m,) + tuple(grid_shape), -1, dtype=np.int8)
        self.confirmed = np.zeros(grid_shape, dtype=bool)

    def update(self, in_view, burning):
        """in_view, burning: (rows, cols) True/False grids for one frame."""
        # Only cells in view this frame get a new entry; the rest keep their history.
        self.recent[:, in_view] = np.roll(self.recent[:, in_view], 1, axis=0)
        self.recent[0, in_view] = burning[in_view]
        self.confirmed |= (self.recent == 1).sum(axis=0) >= self.n
        return self.confirmed
```

The `in_view` and `burning` grids for a frame come from georeferencing that frame's pixels onto the `GroundGrid` (see Fire Mapper for how).

**Hardware:** CPU only — minimal overhead

---

### 4. Fire Mapper

**Purpose:** Accumulate confirmed fire into a ground map of everywhere fire has been seen. Bridges Stage 1 (per-frame masks) to Stage 2 (the fire area the predictor starts from).

**Built** (`pipeline/fire_mapper.py`). `frame_to_grids` places every pixel of a frame on the ground with GeoRef and gives the frame's `in_view` and `burning` grids: a cell is in view if at least 4 pixels show it (so far-away cells of about a pixel count as unseen), and burning if at least 4 of them show fire (so a stray fire pixel at a fire's edge doesn't set the neighbouring cell on fire). `FireMapper.update` adds a frame's grids to `first_fire_min` (NaN = never) and `covered`; until the temporal filter exists, each frame's `burning` grid goes straight in. On sim flights, with the sim's fire outlines filled in standing in for the detector, each frame's grids match the cells the simulator says each pixel shows (IoU 0.997 or better for both grids).

**How to build:**
- Use `geo.GroundGrid` for the map. Row 0 is the **north** edge and rows count southward, like an image or a DEM, so a cell's north coordinate goes *down* as the row number goes up. A grid where north increases with the row index would flip the map.
- Store the first-confirmed time in a float grid filled with **NaN** for "never". Don't use 0 to mean "unburned" in a time grid: a detection at t = 0 would vanish.
- Keep a separate True/False **coverage** grid of cells that have been in view at least once. Scoring needs it.
- Getting a frame's fire onto the grid: fill the detector's fire outlines into an image mask (`masks.polygons_to_mask`) and georeference every pixel. Georeferencing only the outline's corners and filling the ground polygon would be exact on flat ground only: over a hill a straight edge in the image lands on a curve. Every pixel is affordable because GeoRef handles a whole 320 x 240 frame in 6–33 ms.

How the Mapper is scored against the simulator is defined under **Scoring** in System-Level Validation.

**Hardware:** CPU only

---

### 5. Alerter

**Purpose:** Sends notifications when fire is confirmed, once per fire region.

**Where `fire_id` comes from:** with no object tracker, a "fire" is a connected region of the Fire Mapper's confirmed grid (`cv2.connectedComponents`). A region keeps its ID from one update to the next when it overlaps a region from the previous update: it takes the ID of the old region it overlaps most, and when two old regions merge, the older ID wins. A region that overlaps nothing gets a new ID.

**How to build:**
- Twilio for SMS (`twilio` Python SDK)
- Firebase Cloud Messaging or Pushover for push notifications
- Rate limiting: one alert per fire region per N minutes
- Load credentials from environment variables, never from code: the repo is public.

```python
from twilio.rest import Client
import time


class Alerter:
    def __init__(self, account_sid, auth_token, from_number, to_numbers, cooldown_sec=300):
        self.client = Client(account_sid, auth_token)
        self.from_number = from_number
        self.to_numbers = to_numbers
        self.cooldown = cooldown_sec
        self.last_alert = {}

    def send_alert(self, fire_id, lat, lon, confidence):
        now = time.time()
        if fire_id in self.last_alert and (now - self.last_alert[fire_id]) < self.cooldown:
            return False  # rate limited

        for to_number in self.to_numbers:
            self.client.messages.create(
                body=f"🔥 Fire detected at {lat:.5f}, {lon:.5f} (conf: {confidence:.0%})",
                from_=self.from_number,
                to=to_number,
            )

        self.last_alert[fire_id] = now
        return True
```

**Hardware:**
- Server or Raspberry Pi running the pipeline
- Cellular modem if operating remotely

---

## Stage 2: Prediction Phase

### 1. Terrain Analyzer

**Purpose:** Elevation, slope and aspect for the area. GeoRef needs elevation (terrain intersection); the Spread Predictor needs slope and aspect.

**Already built:** `Terrain(elevation, cell_size_m)` in `prediction/terrain.py` computes slope and aspect on a grid in **metres**, with row 0 at the north edge. What's left is loading a real DEM into it.

**Reproject first.** USGS 3DEP 1/3 arc-second DEMs are stored in geographic coordinates (degrees, NAD83). Their pixel size, `transform[0]`, is about 9.3e-5 **degrees**, not metres, so using it as the cell size in `np.gradient` gives wrong slopes everywhere. Reproject to the local UTM zone (the one `geo.LocalFrame` picks) on a metre grid first:

```python
import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject
from vigil.prediction.terrain import Terrain


def load_dem(path, utm_epsg, cell_size_m=10.0):
    """Read a DEM, reproject it onto a UTM grid in metres, and wrap it as a Terrain."""
    dst_crs = f"EPSG:{utm_epsg}"
    with rasterio.open(path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=cell_size_m
        )
        elevation = np.empty((height, width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=elevation,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
        )
    return Terrain(elevation, cell_size_m), transform
```

TODO: handle nodata cells, and crop/align the result to the project's `GroundGrid`.

**Data sources:**
- USGS 3DEP: https://elevation.nationalmap.gov/
- `elevation` Python package for quick DEM fetching

**Hardware:** Storage for DEM tiles (~50MB per 100km²), CPU only

---

### 2. Weather Integrator

**Purpose:** Fetches current weather and turns it into the `Weather` inputs the spread model takes (`prediction/spread.py`).

**How to build:**
- NOAA/NWS API for US coverage (free). It **requires a `User-Agent` header** that identifies the app and a contact.
- Parse string values ("10 mph", "NW") into numbers.
- NWS reports where the wind comes FROM; the model wants where it blows TO.
- Spread depends on **midflame** wind (at flame height), which is much slower than the forecast wind because vegetation shelters it. Convert with a wind adjustment factor (WAF), typically 0.1 under closed canopy to 0.6 in open grass.
- TODO: confirm the measurement height of the NWS forecast wind before applying the WAF. NWS gridded forecasts document it as the 10 m (33 ft) wind; US fire-behaviour tables use the 20 ft wind, roughly the 10 m wind divided by 1.15.

```python
import re
import requests
from vigil.prediction.spread import Weather

HEADERS = {"User-Agent": "Vigil capstone (TODO(Will): contact email)"}

COMPASS_TO_DEG = {
    "N": 0,
    "NNE": 22.5,
    "NE": 45,
    "ENE": 67.5,
    "E": 90,
    "ESE": 112.5,
    "SE": 135,
    "SSE": 157.5,
    "S": 180,
    "SSW": 202.5,
    "SW": 225,
    "WSW": 247.5,
    "W": 270,
    "WNW": 292.5,
    "NW": 315,
    "NNW": 337.5,
}


def get_weather(lat, lon, wind_adjustment_factor=0.4) -> Weather:
    point = requests.get(f"https://api.weather.gov/points/{lat},{lon}", headers=HEADERS, timeout=10).json()
    forecast = requests.get(point["properties"]["forecastHourly"], headers=HEADERS, timeout=10).json()
    current = forecast["properties"]["periods"][0]

    # "10 mph" or "5 to 10 mph". Taking the top of a range is a deliberate,
    # conservative choice: it errs towards faster spread.
    match = re.search(r"(\d+)\s*(?:to\s*(\d+))?\s*mph", current["windSpeed"], re.I)
    forecast_wind_mph = int(match.group(2) or match.group(1)) if match else 0

    # TODO: choose the adjustment factor per fuel/canopy instead of one value.
    midflame_wind_mph = forecast_wind_mph * wind_adjustment_factor

    wind_from_deg = COMPASS_TO_DEG.get(current["windDirection"], 0.0)

    temp_f = current["temperature"]
    humidity = (current.get("relativeHumidity") or {}).get("value", 30)
    # Placeholder rule, not a published fuel-moisture model: drier when hot,
    # damper when humid. TODO: replace with a real dead-fuel-moisture model.
    fuel_moisture = max(2, min(30, 2.0 + 0.3 * humidity - 0.1 * (temp_f - 70)))

    return Weather(
        wind_speed_mph=midflame_wind_mph,
        wind_to_deg=(wind_from_deg + 180) % 360,
        fuel_moisture_pct=fuel_moisture,
    )
```

**Data sources:**
- NOAA/NWS API: https://api.weather.gov (free)
- OpenWeatherMap as backup

**Hardware:** Internet connection, CPU only

---

### 3. Fuel Mapper

**Purpose:** The fuel type of every grid cell, as the integer codes `SpreadPredictor` expects.

**How to build:**
- Download LANDFIRE's **FBFM13** product (the 13 Anderson fuel models). Not FBFM40: its codes run from 91 to 204, and `SpreadPredictor` raises `Unknown fuel codes` on them.
- FBFM13 marks non-burnable ground with codes 91, 92, 93, 98 and 99 (urban, snow/ice, agriculture, water, barren). The model only knows 99, so map all five to `NON_BURNABLE` (99) when loading.
- Reproject and align to the `GroundGrid` like the DEM, but with nearest-neighbour resampling: codes are categories, so they must never be averaged.

The spread rates live in `FUEL_MODELS` in `prediction/spread.py`: each fuel's base rate of spread in m/min **on flat ground with no wind**. They are uncalibrated placeholders, not published fuel-model values, so real-world predictions made with them are illustrative only.

```python
import numpy as np
from vigil.prediction.spread import FUEL_MODELS, NON_BURNABLE

LANDFIRE_NON_BURNABLE = [91, 92, 93, 98, 99]


def to_model_codes(landfire_codes: np.ndarray) -> np.ndarray:
    """LANDFIRE FBFM13 codes -> the codes SpreadPredictor accepts."""
    codes = landfire_codes.astype(int).copy()
    codes[np.isin(codes, LANDFIRE_NON_BURNABLE)] = NON_BURNABLE
    unknown = set(np.unique(codes).tolist()) - set(FUEL_MODELS)
    if unknown:
        raise ValueError(f"Unexpected fuel codes (nodata?): {sorted(unknown)}")
    return codes
```

**Data sources:**
- LANDFIRE: https://landfire.gov/ (FBFM13)

**Hardware:** Storage for fuel rasters (~100MB per 100km²), CPU only

---

### 4. Spread Predictor

**Built:** `SpreadPredictor` in `prediction/spread.py`. It computes an **arrival time** for every cell (when the fire front reaches it) with Dijkstra's shortest-path algorithm: moving into a neighbouring cell takes distance ÷ spread rate minutes, and the fire reaches each cell by its quickest route. The spread rate is the fuel's base rate times factors for wind, slope and moisture. A 500×500 grid takes about 0.6 s on a laptop.

**Using it:**
- `set_ignition(cells, start_time_min=now)` seeds the fire (from the Fire Mapper); `predict([1, 3, 6])` then returns burned-area grids for **1, 3 and 6 hours after now**. "Now" defaults to the latest ignition time; pass `now_min=` to override it.
- `arrival_time` and `burned_at(t)` use the same absolute clock, in minutes.
- The simulator uses this same class as its ground-truth fire.

**Known limitations** (for the thesis):
- **Grid directions.** Fire can only move to the 8 neighbouring cells, so paths in between zig-zag. Arrival times come out up to 8.2% too late, worst at about 22.5° off the grid axes, and a calm fire grows as an octagon. A 16-neighbour stencil would reduce this.
- **Slope** uses the destination cell's slope and aspect, not the actual rise between the two cells.
- **Uphill speed-up is understated.** The `5.275·tan²(slope)` term is Rothermel's slope factor without its packing-ratio term (β^-0.3, which is greater than 1 for real fuel beds).
- **No moisture of extinction.** The moisture factor bottoms out at 0.1 rather than stopping the fire, so very wet fuel still burns, slowly.
- **Wind multipliers** for a head fire (spreading straight downwind) are about 2.3× at 5 mph, 4.0× at 10, 7.9× at 20 and 12.3× at 30 mph. They come from the formula, not from measurements.
- **Fuel rates** are uncalibrated placeholders (see Fuel Mapper).

**Validation:**
- Benchmark against Next Day Wildfire Spread dataset (1km/day resolution)
- Report IoU (intersection over union) between predicted and actual burned area
- For sim testing: run predictor with deliberately wrong wind/fuel values to test against ground truth

---

## Stage 3: Suppression Support (Future Work)

*Condensed. The full Stage 3 write-up (why autonomous suppression fails, retardant chemistry, use cases, hardware costs, research questions) is in git history: `git show 746f8db:docs/ARCHITECTURE.md`.*

### Core Concept

Stage 3 does not attempt autonomous fire suppression. Instead, it uses Stage 2's predictions to **place protective barriers ahead of the fire**, slowing spread and buying time for human responders.

### Fire Retardant Technology

| Type | Products | Use Case |
|------|----------|----------|
| Long-term (LTR) | Phos-Chek LC95A, MVP-Fx | Pre-applied hours ahead; stays effective after water evaporates |
| Gel | Barricade, ThermGel | Structure protection; holds moisture for hours |
| Class A Foam | Phos-Chek WD881, Silv-Ex | Lightweight; 20L + 0.5% concentrate ≈ 60-80L water |

### Dashboard Integration: Barrier Planning Tool

1. Prediction overlay — perimeters at 1hr, 3hr, 6hr
2. Risk layer — structures, roads, infrastructure
3. Barrier drawing — swipe to mark protection line
4. Resource calculator — volume, sorties, drop sequence
5. Execution — autonomous dispatch or tanker recommendations

---

## Evaluation Strategy

### Detection Accuracy (Stage 1)

**Dataset:** FLAME 2 fire/smoke imagery

**Protocol:**
1. Generate training masks from the thermal frames with the palette-aware method (see Detector)
2. Train YOLO11-seg on RGB frames with thermal-derived masks
3. Hand-label separate test set (~300 frames) to avoid circular evaluation
4. Split into contiguous frame blocks (not random) to prevent data leakage

**Experiments:**
| Model | Input | Expected finding |
|-------|-------|------------------|
| YOLO11-seg | RGB only | Baseline; validates thermal labels transfer to RGB |
| YOLO11-seg | Thermal only | Upper bound; may overfit to threshold |
| YOLO11-seg | RGB + thermal fusion | Potential improvement |

**Metrics:** mAP@50, mAP@50:95, precision, recall per class

**Note:** Smoke class requires separate hand-labeling (thermal masks only find heat). The `smoke` flag in FLAME 2's `labels.txt` means "smoke fills at least 50% of the RGB frame", not "smoke present", so it can't serve as a smoke label either.

---

### System-Level Validation (Stages 1-2)

**Platform:** MuJoCo simulation with scripted camera path (no SITL initially). Built: `src/vigil/sim/`.

**Sim setup:**
- Run Spread Predictor as ground truth fire
- Render burning cells as colored ground tiles
- Detect with MuJoCo segmentation mask (not YOLO) for system testing
- Exact camera pose available for georeferencing ground truth

**Scale:** one MuJoCo box per grid cell renders a 640×480 frame in about 27 ms at 80×80 cells and 80 ms at 160×160 (measured on the development laptop). A 500×500 grid would be 250,000 boxes; it hasn't been tried and is unlikely to be practical. Bigger grids should switch to a MuJoCo heightfield with a texture for the RGB view, and get cell IDs by ray casting instead of segmentation.

**What to measure:**
| Metric | Method |
|--------|--------|
| Fire Mapper accuracy | IoU between mapped area and ground truth grid (see Scoring) |
| GeoRef error | Distance from projected point to known ground truth (see Scoring) |
| Alert latency | Time from first visible fire to alert |
| Prediction accuracy | Run predictor with **wrong** wind/fuel; compare to truth |

**Why scripted camera:** "Flight path execution accuracy" measures autopilot, not Vigil. Save SITL for later integration.

#### Scoring

- **Truth.** The sim's truth grid (`SimObservation.truth_state`) has three states: UNBURNED, BURNING and BURNED (burned out).
- **Fire Mapper, primary score: "ever burned".** Fire polygons are outer outlines (OpenCV `RETR_EXTERNAL`), so burned-out centres get filled in, and the Mapper builds up its map over time. Its main output is therefore "ever seen burning", scored by IoU against `truth_state != UNBURNED` at the same timestamp.
- **Coverage.** Score IoU only over cells that have been in view at least once (the Mapper's coverage grid), and report coverage separately, so a cell nobody looked at doesn't count as a miss.
- **Active front (optional, secondary).** IoU against `truth_state == BURNING`. This needs its own Mapper output, e.g. cells seen burning in the last K minutes.
- **GeoRef error.** For sampled pixels, the distance in metres between the georeferenced point and the true ground point. The truth comes from `SimObservation.cell_ids` (which cell the pixel shows) located exactly by the same ray/terrain intersection. Report the mean and the 95th percentile.

---

### Spread Prediction Validation

**Dataset:** Next Day Wildfire Spread (Kaggle) — 1km/day resolution

**Metric:** IoU between predicted and actual burned area

**Note:** NIFC historical perimeters require assembling weather/fuel data and accounting for suppression effects (fires being fought). The Kaggle dataset is pre-assembled.

---

## Component Summary

### Stage 1: Detection

| Component | Purpose | Key Libraries |
|-----------|---------|---------------|
| Detector | Segment fire/smoke in frames | ultralytics (yolo11-seg) |
| GeoRef | Pixel → ground point (camera ray + terrain) | numpy, pyproj (via `geo.LocalFrame`) |
| Temporal Filter | Confirm fire per ground cell (N of last M views) | numpy |
| Fire Mapper | Accumulate confirmed fire into the ground grid | numpy (`geo.GroundGrid`) |
| Alerter | Notify once per fire region | twilio, requests, opencv (regions) |

### Stage 2: Prediction

| Component | Purpose | Key Libraries |
|-----------|---------|---------------|
| Terrain Analyzer | Elevation, slope, aspect (`prediction/terrain.py` built; DEM loading to do) | rasterio (reprojection), numpy |
| Weather Integrator | Wind, temp, humidity, fuel moisture | requests |
| Fuel Mapper | LANDFIRE FBFM13 codes → model fuel codes | rasterio |
| Spread Predictor | Arrival-time fire propagation (`prediction/spread.py`, built) | numpy, heapq |

### Shared

| Component | Purpose |
|-----------|---------|
| Dashboard | Web UI for all stages |
| Drone | Sensor platform |

---

## Suggested Build Order

1. ✅ **Sim adapter** — Spread Predictor fire on ground grid, scripted MuJoCo camera, exact pose (done: `src/vigil/sim/`, `adapters/sim.py`)
2. 🔄 **GeoRef + Fire Mapper** — Score against sim ground truth (GeoRef done, flat ground and terrain: `pipeline/georef.py`; Fire Mapper done: `pipeline/fire_mapper.py`; scoring script next)
3. **Arrival-time predictor** — Test with wrong wind/fuel vs ground truth
4. **FLAME 2 detector** — Parallel track; independent of sim

---

## Data Sources

| Data | Source | Resolution | Cost |
|------|--------|------------|------|
| Elevation/DEM | USGS 3DEP | 10m | Free |
| Fuel models | LANDFIRE | 30m | Free |
| Weather | NOAA/NWS API | Point-based | Free |
| Fire imagery | FLAME 2 | Video | Free |
| Spread benchmark | Next Day Wildfire Spread | 1km/day | Free |
