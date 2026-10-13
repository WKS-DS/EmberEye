# EmberEye System Architecture

## Overview

EmberEye is a drone-based wildfire detection and prediction system with three stages:

- **Stage 1: Detection** — Real-time fire identification and alerting
- **Stage 2: Prediction** — Fire spread forecasting for proactive crew positioning
- **Stage 3: Suppression Support** *(Future Work)* — Predictive barrier placement to slow fire spread and buy time for responders

The **Dashboard** and **Drone** span all stages, providing the interface and sensor platform respectively.

---

## Stage 1: Detection Phase

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
- Generate labels from thermal imagery (threshold at ~180°C for fire)
- Train model on RGB frames using thermal-derived masks
- Hand-label a separate test set (~300 frames) to avoid circular evaluation
- Split FLAME 2 by sequence (not random) to prevent data leakage from consecutive frames

**Hardware:**
- On-drone (edge): NVIDIA Jetson Orin Nano (~$500) — runs YOLO11 at 15-30 FPS
- Ground station: Any GPU (RTX 3060+) for real-time, or CPU for slower processing
- Optional: Thermal camera (FLIR Lepton ~$200, or FLIR Vue for higher-end)

**Evaluation experiments:**
1. RGB-only detection (thermal-derived labels)
2. Thermal-only detection
3. RGB + thermal fusion
4. Report mAP, precision, recall by class; note smoke labels require separate annotation

---

### 2. Temporal Filter (replaces Tracker)

**Purpose:** Confirms detections over multiple frames before alerting. Standard object trackers (ByteTrack, BoT-SORT) assume separate objects; fire merges, splits, and most apparent motion is camera movement. Use temporal filtering only to reduce false positives.

**How to build:**
- Require N consecutive frames with overlapping detections before confirming
- Use IoU between frame masks, not track IDs
- Pass confirmed detections to Fire Mapper

```python
class TemporalFilter:
    def __init__(self, confirm_frames=3, iou_threshold=0.3):
        self.confirm_frames = confirm_frames
        self.iou_threshold = iou_threshold
        self.history = []  # list of recent masks
    
    def update(self, masks):
        self.history.append(masks)
        if len(self.history) > self.confirm_frames:
            self.history.pop(0)
        
        if len(self.history) < self.confirm_frames:
            return []
        
        # Find masks with consistent overlap across history
        confirmed = []
        for mask in self.history[-1]:
            if self._has_consistent_overlap(mask):
                confirmed.append(mask)
        return confirmed
    
    def _has_consistent_overlap(self, mask):
        # Check IoU with at least one mask in each prior frame
        for frame_masks in self.history[:-1]:
            if not any(self._iou(mask, m) > self.iou_threshold for m in frame_masks):
                return False
        return True
```

**Hardware:** CPU only — minimal overhead

---

### 3. GeoRef (Georeferencing Module)

**Purpose:** Converts pixel coordinates to real-world GPS coordinates. Fuses drone telemetry (position, altitude AGL, heading, gimbal pitch/roll) with camera intrinsics and terrain elevation.

**How to build:**
- Input: pixel (x, y), drone GPS, altitude AGL (not ASL), heading, gimbal angles, camera FOV
- Require terrain elevation from Stage 2's Terrain Analyzer for non-flat ground
- Ray-cast from camera through image plane to terrain intersection

```python
import numpy as np
from pyproj import Transformer

class GeoRef:
    def __init__(self, camera_fov_h, camera_fov_v, image_width, image_height):
        self.fov_h = np.radians(camera_fov_h)
        self.fov_v = np.radians(camera_fov_v)
        self.img_w = image_width
        self.img_h = image_height
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)  # UTM zone 10N
    
    def pixel_to_gps(self, pixel_x, pixel_y, drone_lat, drone_lon, altitude_agl, 
                     heading_deg, gimbal_pitch_deg, terrain_elevation_func=None):
        # Pixel offset from center
        cx, cy = self.img_w / 2, self.img_h / 2
        dx = (pixel_x - cx) / cx  # -1 to 1
        dy = (pixel_y - cy) / cy  # -1 to 1
        
        # Angular offset
        az_offset = dx * (self.fov_h / 2)
        el_offset = dy * (self.fov_v / 2)
        
        # Total angles (gimbal pitch: 0 = forward, -90 = nadir)
        azimuth = np.radians(heading_deg) + az_offset
        elevation = np.radians(gimbal_pitch_deg) + el_offset
        
        # Ray direction (NED frame)
        ray_n = np.cos(elevation) * np.cos(azimuth)
        ray_e = np.cos(elevation) * np.sin(azimuth)
        ray_d = -np.sin(elevation)  # positive down
        
        if ray_d <= 0:
            return None  # ray points up, no ground intersection
        
        # Ground intersection (flat terrain approximation)
        t = altitude_agl / ray_d
        offset_n = ray_n * t
        offset_e = ray_e * t
        
        # Convert to GPS
        drone_x, drone_y = self.transformer.transform(drone_lon, drone_lat)
        target_x = drone_x + offset_e
        target_y = drone_y + offset_n
        
        target_lon, target_lat = self.transformer.transform(target_x, target_y, direction='INVERSE')
        
        # Optional: iterate with terrain elevation for accuracy
        if terrain_elevation_func is not None:
            # Refine intersection with actual terrain height
            pass
        
        return target_lat, target_lon
```

**Hardware:**
- Drone with GPS module and IMU (standard)
- Gimbal with pitch/roll telemetry
- Camera with known FOV specs

---

### 4. Fire Mapper

**Purpose:** Accumulates georeferenced detections into a ground grid of burning cells. Bridges Stage 1 (per-frame masks) to Stage 2 (fire perimeter grid).

**How to build:**
- Maintain a spatial grid (e.g., 10m cells)
- Project each confirmed mask polygon to ground coordinates
- Mark cells as burning; track first-seen and last-seen times
- Output current fire perimeter as grid or polygon

```python
import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

class FireMapper:
    def __init__(self, cell_size=10, grid_extent=5000):
        self.cell_size = cell_size  # meters
        self.grid_extent = grid_extent  # meters from origin
        self.grid_size = grid_extent // cell_size
        
        # Grid state: 0=unburned, timestamp=first detection time
        self.fire_grid = np.zeros((self.grid_size, self.grid_size))
        self.origin_lat = None
        self.origin_lon = None
        self.transformer = None
    
    def set_origin(self, lat, lon):
        from pyproj import Transformer
        self.origin_lat = lat
        self.origin_lon = lon
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
        self.origin_x, self.origin_y = self.transformer.transform(lon, lat)
    
    def update(self, mask_polygons_gps, timestamp):
        """Update grid with new georeferenced mask polygons."""
        for poly_gps in mask_polygons_gps:
            # Convert GPS polygon to grid coordinates
            poly_xy = []
            for lat, lon in poly_gps:
                x, y = self.transformer.transform(lon, lat)
                gx = int((x - self.origin_x) / self.cell_size + self.grid_size // 2)
                gy = int((y - self.origin_y) / self.cell_size + self.grid_size // 2)
                poly_xy.append((gx, gy))
            
            # Rasterize polygon to grid cells
            if len(poly_xy) >= 3:
                poly = Polygon(poly_xy)
                minx, miny, maxx, maxy = [int(v) for v in poly.bounds]
                for gx in range(max(0, minx), min(self.grid_size, maxx + 1)):
                    for gy in range(max(0, miny), min(self.grid_size, maxy + 1)):
                        if poly.contains(box(gx, gy, gx + 1, gy + 1).centroid):
                            if self.fire_grid[gy, gx] == 0:
                                self.fire_grid[gy, gx] = timestamp
    
    def get_perimeter(self):
        """Return current fire perimeter as list of (lat, lon) points."""
        burning = self.fire_grid > 0
        # Extract contour and convert back to GPS
        # (simplified: return bounding cells)
        coords = np.argwhere(burning)
        if len(coords) == 0:
            return []
        
        perimeter_gps = []
        for gy, gx in coords:
            x = self.origin_x + (gx - self.grid_size // 2) * self.cell_size
            y = self.origin_y + (gy - self.grid_size // 2) * self.cell_size
            lon, lat = self.transformer.transform(x, y, direction='INVERSE')
            perimeter_gps.append((lat, lon))
        
        return perimeter_gps
    
    def get_burning_cells(self):
        """Return grid coordinates of all burning cells for Stage 2."""
        return np.argwhere(self.fire_grid > 0)
```

**Hardware:** CPU only

---

### 5. Alerter

**Purpose:** Sends notifications when fire is confirmed by the Temporal Filter.

**How to build:**
- Twilio for SMS (`twilio` Python SDK)
- Firebase Cloud Messaging or Pushover for push notifications
- Rate limiting: one alert per fire region per N minutes

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
                to=to_number
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

**Purpose:** Pulls elevation data and calculates slope angle and aspect. Required by both GeoRef (terrain intersection) and Spread Predictor (fire behavior).

**How to build:**
- Fetch DEM tiles from USGS 3DEP (10m resolution)
- Calculate slope and aspect
- Provide elevation lookup for GeoRef

```python
import rasterio
import numpy as np

class TerrainAnalyzer:
    def __init__(self, dem_path):
        self.dem = rasterio.open(dem_path)
        self.elevation = self.dem.read(1)
        self.transform = self.dem.transform
        self.cell_size = self.transform[0]  # meters per pixel
        
        # Calculate slope and aspect
        dy, dx = np.gradient(self.elevation, self.cell_size)
        self.slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        self.aspect = np.degrees(np.arctan2(-dx, dy)) % 360  # 0=N, 90=E, 180=S, 270=W
    
    def get_elevation(self, lat, lon):
        row, col = self.dem.index(lon, lat)
        if 0 <= row < self.elevation.shape[0] and 0 <= col < self.elevation.shape[1]:
            return float(self.elevation[row, col])
        return None
    
    def get_slope(self, lat, lon):
        row, col = self.dem.index(lon, lat)
        if 0 <= row < self.slope.shape[0] and 0 <= col < self.slope.shape[1]:
            return float(self.slope[row, col])
        return 0.0
    
    def get_aspect(self, lat, lon):
        row, col = self.dem.index(lon, lat)
        if 0 <= row < self.aspect.shape[0] and 0 <= col < self.aspect.shape[1]:
            return float(self.aspect[row, col])
        return 0.0
    
    def get_slope_grid(self, bounds):
        """Return slope array for a bounding box."""
        # bounds: (min_lon, min_lat, max_lon, max_lat)
        pass

```

**Data sources:**
- USGS 3DEP: https://elevation.nationalmap.gov/
- `elevation` Python package for quick DEM fetching

**Hardware:** Storage for DEM tiles (~50MB per 100km²), CPU only

---

### 2. Weather Integrator

**Purpose:** Fetches real-time weather. Returns structured data with correct keys and parsed values.

**How to build:**
- NOAA/NWS API for US coverage (free)
- Parse string values ("10 mph", "NW") into numbers
- Wind direction: convert compass to degrees; note NWS reports where wind comes FROM

```python
import requests
import re

class WeatherIntegrator:
    COMPASS_TO_DEG = {
        'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
        'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
        'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
        'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
    }
    
    def get_weather(self, lat, lon):
        point = requests.get(f"https://api.weather.gov/points/{lat},{lon}").json()
        forecast_url = point["properties"]["forecastHourly"]
        
        forecast = requests.get(forecast_url).json()
        current = forecast["properties"]["periods"][0]
        
        # Parse wind speed (e.g., "10 mph" or "5 to 10 mph")
        wind_str = current["windSpeed"]
        wind_match = re.search(r'(\d+)\s*(?:to\s*(\d+))?\s*mph', wind_str, re.I)
        if wind_match:
            wind_speed = int(wind_match.group(2) or wind_match.group(1))
        else:
            wind_speed = 0
        
        # Parse wind direction (where wind comes FROM)
        wind_dir_str = current["windDirection"]
        wind_from_deg = self.COMPASS_TO_DEG.get(wind_dir_str, 0)
        # Convert to where wind is GOING (for spread calculation)
        wind_to_deg = (wind_from_deg + 180) % 360
        
        # Temperature and humidity
        temp_f = current["temperature"]
        humidity = current.get("relativeHumidity", {}).get("value", 30)
        
        # Dead fuel moisture (simplified Nelson model)
        # Moisture DECREASES with temperature, INCREASES with humidity
        fuel_moisture = 2.0 + 0.3 * humidity - 0.1 * (temp_f - 70)
        fuel_moisture = max(2, min(30, fuel_moisture))
        
        return {
            "wind_speed": wind_speed,  # mph
            "wind_from_deg": wind_from_deg,  # where wind comes from
            "wind_to_deg": wind_to_deg,  # where wind pushes fire
            "temperature": temp_f,  # Fahrenheit
            "humidity": humidity,  # percent
            "fuel_moisture": fuel_moisture  # percent
        }
```

**Data sources:**
- NOAA/NWS API: https://api.weather.gov (free)
- OpenWeatherMap as backup

**Hardware:** Internet connection, CPU only

---

### 3. Fuel Mapper

**Purpose:** Identifies vegetation type and assigns burn characteristics.

**How to build:**
- Load LANDFIRE fuel model rasters
- Map fuel codes to spread parameters (with units)

```python
import rasterio

# Anderson 13 fuel models with rate of spread in m/min under reference conditions
# (mid-flame wind 2 m/s, slope 0, fuel moisture 8%)
FUEL_MODELS = {
    1: {"name": "short_grass", "ros_ref": 2.5, "heat_kj_m2": 700},
    2: {"name": "timber_grass", "ros_ref": 1.8, "heat_kj_m2": 900},
    3: {"name": "tall_grass", "ros_ref": 4.5, "heat_kj_m2": 500},
    4: {"name": "chaparral", "ros_ref": 4.0, "heat_kj_m2": 2500},
    5: {"name": "brush", "ros_ref": 1.0, "heat_kj_m2": 700},
    6: {"name": "dormant_brush", "ros_ref": 1.5, "heat_kj_m2": 600},
    7: {"name": "southern_rough", "ros_ref": 1.2, "heat_kj_m2": 800},
    8: {"name": "timber_litter", "ros_ref": 0.3, "heat_kj_m2": 400},
    9: {"name": "hardwood_litter", "ros_ref": 0.5, "heat_kj_m2": 500},
    10: {"name": "timber_understory", "ros_ref": 2.0, "heat_kj_m2": 2800},
    11: {"name": "light_slash", "ros_ref": 1.5, "heat_kj_m2": 1200},
    12: {"name": "medium_slash", "ros_ref": 2.5, "heat_kj_m2": 1800},
    13: {"name": "heavy_slash", "ros_ref": 3.5, "heat_kj_m2": 2500},
}

class FuelMapper:
    def __init__(self, landfire_path):
        self.raster = rasterio.open(landfire_path)
        self.data = self.raster.read(1)
    
    def get_fuel(self, lat, lon):
        row, col = self.raster.index(lon, lat)
        if 0 <= row < self.data.shape[0] and 0 <= col < self.data.shape[1]:
            code = int(self.data[row, col])
            return FUEL_MODELS.get(code, FUEL_MODELS[1])
        return FUEL_MODELS[1]
    
    def get_fuel_grid(self, grid_coords, transform_func):
        """Return fuel parameters for each grid cell."""
        pass
```

**Data sources:**
- LANDFIRE: https://landfire.gov/

**Hardware:** Storage for fuel rasters (~100MB per 100km²), CPU only

---

### 4. Spread Predictor

**Purpose:** Simulates fire spread using arrival-time algorithm (not step-based cellular automata). Outputs predicted perimeters at specified times.

**Approach:** Dijkstra-style shortest-path where edge weight = travel time = distance / spread rate. Fixes speed-cap and cumulative-step bugs in naive CA approach.

**Key physics (simplified, not full Rothermel):**
- Wind effect: exponential with alignment to spread direction
- Slope effect: uphill accelerates (preheating), downhill decelerates; use aspect to determine direction
- Moisture effect: dampens spread rate

```python
import numpy as np
import heapq

class SpreadPredictor:
    def __init__(self, terrain, fuel_grid, weather, cell_size=30):
        """
        terrain: TerrainAnalyzer instance or (elevation, slope, aspect) arrays
        fuel_grid: 2D array of fuel model dicts
        weather: dict with wind_speed, wind_to_deg, fuel_moisture
        cell_size: meters per grid cell
        """
        self.slope = terrain.slope if hasattr(terrain, 'slope') else terrain[1]
        self.aspect = terrain.aspect if hasattr(terrain, 'aspect') else terrain[2]
        self.fuel = fuel_grid
        self.weather = weather
        self.cell_size = cell_size
        
        self.rows, self.cols = self.slope.shape
        # Arrival time: inf = not yet reached
        self.arrival_time = np.full((self.rows, self.cols), np.inf)
    
    def set_ignition(self, burning_cells, start_time=0):
        """Set initial fire cells from Fire Mapper output."""
        for row, col in burning_cells:
            if 0 <= row < self.rows and 0 <= col < self.cols:
                self.arrival_time[row, col] = start_time
    
    def calc_spread_rate(self, from_row, from_col, to_row, to_col):
        """Calculate spread rate in m/min from one cell to neighbor."""
        # Direction of spread
        dr = to_row - from_row
        dc = to_col - from_col
        spread_dir = np.degrees(np.arctan2(dc, -dr)) % 360  # 0=N
        
        # Fuel at destination
        fuel = self.fuel[to_row, to_col]
        base_ros = fuel.get("ros_ref", 1.0)  # m/min
        
        # Wind factor
        # wind_to_deg is where wind pushes fire
        wind_alignment = np.cos(np.radians(self.weather["wind_to_deg"] - spread_dir))
        wind_speed = self.weather["wind_speed"] * 0.447  # mph to m/s
        if wind_alignment > 0:
            wind_factor = 1 + 0.5 * (wind_speed ** 1.2) * wind_alignment
        else:
            wind_factor = 1 / (1 + 0.3 * wind_speed * abs(wind_alignment))
        
        # Slope factor
        slope_deg = self.slope[to_row, to_col]
        aspect_deg = self.aspect[to_row, to_col]
        
        # Check if spreading uphill or downhill
        # aspect = direction of DOWNslope; spreading opposite to aspect = uphill
        uphill_alignment = np.cos(np.radians(aspect_deg - spread_dir + 180))
        
        if uphill_alignment > 0:
            # Uphill: accelerate (preheating effect)
            slope_factor = 1 + 5.275 * (np.tan(np.radians(slope_deg)) ** 2) * uphill_alignment
        else:
            # Downhill: decelerate
            slope_factor = 1 / (1 + 2.0 * np.tan(np.radians(slope_deg)) * abs(uphill_alignment))
        
        # Moisture factor (fuel_moisture in percent)
        moisture = self.weather.get("fuel_moisture", 8)
        moisture_factor = max(0.1, 1 - (moisture - 8) / 25)
        
        ros = base_ros * wind_factor * slope_factor * moisture_factor
        return max(0.01, ros)  # minimum spread rate
    
    def propagate(self):
        """Run Dijkstra to compute arrival times at all cells."""
        # Priority queue: (arrival_time, row, col)
        pq = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.arrival_time[row, col] < np.inf:
                    heapq.heappush(pq, (self.arrival_time[row, col], row, col))
        
        # 8 neighbors
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
        
        while pq:
            t, row, col = heapq.heappop(pq)
            
            if t > self.arrival_time[row, col]:
                continue  # already processed with shorter time
            
            for dr, dc in neighbors:
                nr, nc = row + dr, col + dc
                
                # Bounds check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                
                # Travel time to neighbor
                dist = self.cell_size * (1.414 if (dr and dc) else 1.0)
                ros = self.calc_spread_rate(row, col, nr, nc)
                travel_time = dist / ros  # minutes
                
                new_arrival = t + travel_time
                
                if new_arrival < self.arrival_time[nr, nc]:
                    self.arrival_time[nr, nc] = new_arrival
                    heapq.heappush(pq, (new_arrival, nr, nc))
    
    def get_perimeter_at(self, time_minutes):
        """Return cells that have burned by given time."""
        burned = self.arrival_time <= time_minutes
        return np.argwhere(burned)
    
    def predict(self, hours_list=[1, 3, 6]):
        """Return predicted perimeters at specified hours."""
        self.propagate()
        
        predictions = {}
        for h in hours_list:
            minutes = h * 60
            cells = self.get_perimeter_at(minutes)
            predictions[f"{h}hr"] = cells.tolist()
        
        return predictions
```

**Hardware:** CPU — Dijkstra on a 500×500 grid runs in seconds

**Validation:**
- Benchmark against Next Day Wildfire Spread dataset (1km/day resolution)
- Report IoU (intersection over union) between predicted and actual burned area
- For sim testing: run predictor with deliberately wrong wind/fuel values to test against ground truth

---

## Stage 3: Suppression Support (Future Work)

*[Stage 3 content unchanged — see previous version]*

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
1. Generate training labels from thermal thresholding
2. Train YOLO11-seg on RGB frames with thermal-derived masks
3. Hand-label separate test set (~300 frames) to avoid circular evaluation
4. Split by sequence (not random) to prevent data leakage

**Experiments:**
| Model | Input | Expected finding |
|-------|-------|------------------|
| YOLO11-seg | RGB only | Baseline; validates thermal labels transfer to RGB |
| YOLO11-seg | Thermal only | Upper bound; may overfit to threshold |
| YOLO11-seg | RGB + thermal fusion | Potential improvement |

**Metrics:** mAP@50, mAP@50:95, precision, recall per class

**Note:** Smoke class requires separate hand-labeling (thermal thresholding only finds heat).

---

### System-Level Validation (Stages 1-2)

**Platform:** MuJoCo simulation with scripted camera path (no SITL initially)

**Sim setup:**
- Run Spread Predictor as ground truth fire
- Render burning cells as colored ground tiles
- Detect with MuJoCo segmentation mask (not YOLO) for system testing
- Exact camera pose available for georeferencing ground truth

**What to measure:**
| Metric | Method |
|--------|--------|
| Fire Mapper accuracy | IoU between mapped perimeter and ground truth grid |
| GeoRef error | Distance from projected point to known ground truth |
| Alert latency | Time from first visible fire to alert |
| Prediction accuracy | Run predictor with **wrong** wind/fuel; compare to truth |

**Why scripted camera:** "Flight path execution accuracy" measures autopilot, not EmberEye. Save SITL for later integration.

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
| Temporal Filter | Confirm detections over N frames | numpy, shapely |
| GeoRef | Pixel → GPS with gimbal + terrain | numpy, pyproj, rasterio |
| Fire Mapper | Accumulate detections into ground grid | numpy, shapely |
| Alerter | Send notifications | twilio, requests |

### Stage 2: Prediction

| Component | Purpose | Key Libraries |
|-----------|---------|---------------|
| Terrain Analyzer | Elevation, slope, aspect | rasterio, numpy |
| Weather Integrator | Wind, temp, humidity, fuel moisture | requests |
| Fuel Mapper | Vegetation type → spread params | rasterio |
| Spread Predictor | Arrival-time fire propagation | numpy, heapq |

### Shared

| Component | Purpose |
|-----------|---------|
| Dashboard | Web UI for all stages |
| Drone | Sensor platform |

---

## Suggested Build Order

1. **Sim adapter** — Spread Predictor fire on ground grid, scripted MuJoCo camera, exact pose
2. **GeoRef + Fire Mapper** — Score against sim ground truth
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
