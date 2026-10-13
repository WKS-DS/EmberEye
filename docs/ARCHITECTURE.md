# EmberEye System Architecture

## Overview

EmberEye is a drone-based wildfire detection and prediction system with three stages:

- **Stage 1: Detection** — Real-time fire identification and alerting
- **Stage 2: Prediction** — Fire spread forecasting for proactive crew positioning
- **Stage 3: Suppression Support** *(Future Work)* — Predictive barrier placement to slow fire spread and buy time for responders

The **Dashboard** and **Drone** span all stages, providing the interface and sensor platform respectively.

---

## Stage 1: Detection Phase

### 1. Detector (YOLOv11)

**Purpose:** Runs inference on each video frame to detect fire and smoke. Outputs bounding boxes with class labels and confidence scores.

**How to build:**
- Use Ultralytics YOLOv11: `pip install ultralytics`
- Fine-tune on a fire/smoke dataset (FLAME, D-Fire, or custom)
- Input: RGB or thermal frames
- Output: list of detections `[class, x, y, w, h, confidence]`

```python
from ultralytics import YOLO

model = YOLO("yolov11n.pt")  # or your fine-tuned weights
results = model.predict(frame, conf=0.5)

for box in results[0].boxes:
    cls = box.cls  # 0=fire, 1=smoke
    conf = box.conf
    x1, y1, x2, y2 = box.xyxy[0]
```

**Hardware:**
- On-drone (edge): NVIDIA Jetson Orin Nano (~$500) — runs YOLOv11 at 15-30 FPS
- Ground station: Any GPU (RTX 3060+) for real-time, or CPU for slower processing
- Optional: Thermal camera (FLIR Lepton ~$200, or FLIR Vue for higher-end)

---

### 2. Tracker

**Purpose:** Maintains identity of detected fires across frames. Without it, you'd get a "new" fire every frame. The tracker says "fire #1 moved from here to here" and can track fire spread over time.

**How to build:**
- Use ByteTrack or BoT-SORT (both work with YOLO)
- Ultralytics has built-in tracking: `model.track()`

```python
# Built-in tracking with YOLOv11
results = model.track(frame, persist=True)

for box in results[0].boxes:
    track_id = box.id  # consistent ID across frames
    # Fire #3 has been burning for 45 seconds, moved 20m northeast
```

**Hardware:** Same as Detector — tracking adds minimal overhead (runs on CPU alongside GPU inference)

---

### 3. GeoRef (Georeferencing Module)

**Purpose:** Converts pixel coordinates of a detected fire (e.g., "fire at pixel 320, 240") into real-world GPS coordinates. Fuses drone telemetry (position, altitude, heading, gimbal angle) with camera properties to project where that pixel points in the world.

**How to build:**
- Input: bounding box center (x, y), drone GPS, altitude, heading, camera pitch/roll
- Math: ray-casting from camera through image plane to ground intersection
- Libraries: `numpy` for matrix math, `pyproj` for coordinate transformations

```python
def pixel_to_gps(pixel_x, pixel_y, drone_lat, drone_lon, altitude, heading, camera_fov):
    # Calculate angular offset from image center
    # Project ray to ground plane
    # Apply rotation for drone heading
    # Return lat, lon
    pass
```

**Hardware:**
- Drone with GPS module (standard on most drones)
- IMU (inertial measurement unit) for orientation — standard on DJI, Pixhawk
- Camera with known focal length/FOV specs

---

### 4. Alerter

**Purpose:** Sends notifications when a fire is confirmed — SMS to fire departments, push notifications to mobile apps, webhook calls to dispatch systems, or email alerts.

**How to build:**
- Twilio for SMS (`twilio` Python SDK)
- Firebase Cloud Messaging or Pushover for push notifications
- Simple HTTP webhooks for integrations
- Rate limiting to prevent alert fatigue

```python
from twilio.rest import Client

def send_alert(lat, lon, confidence):
    client = Client(account_sid, auth_token)
    client.messages.create(
        body=f"🔥 Fire detected at {lat}, {lon} (conf: {confidence:.0%})",
        from_="+1234567890",
        to="+0987654321"
    )
```

**Hardware:**
- Server or Raspberry Pi running the pipeline
- Cellular modem if drone operates remotely without WiFi

---

## Stage 2: Prediction Phase

### 1. Terrain Analyzer

**Purpose:** Pulls elevation data for the fire area and calculates slope angle and aspect (compass direction a slope faces). Fires spread faster uphill and in the direction of prevailing winds channeled by terrain.

**How to build:**
- Fetch DEM tiles from USGS 3DEP (10m resolution) or NASA SRTM (30m)
- Calculate slope and aspect using gradient algorithms
- Cache tiles locally for repeated queries

```python
import rasterio
import numpy as np

def load_terrain(lat, lon, radius_km=5):
    dem = rasterio.open(f"elevation_{lat}_{lon}.tif")
    elevation = dem.read(1)
    
    # Calculate slope (degrees)
    dy, dx = np.gradient(elevation, 10)  # 10m cell size
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Calculate aspect (compass direction of downslope)
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360
    
    return elevation, slope, aspect
```

**Data sources:**
- USGS 3DEP: https://elevation.nationalmap.gov/
- `elevation` Python package for quick DEM fetching

**Hardware:** Storage for DEM tiles (~50MB per 100km²), CPU only

---

### 2. Weather Integrator

**Purpose:** Fetches real-time and forecast weather conditions — wind speed, wind direction, temperature, relative humidity. Updates continuously as conditions change.

**How to build:**
- NOAA/NWS API for US coverage (free, no API key needed)
- Calculate derived metrics like dead fuel moisture content
- Poll every 10-15 minutes for updates

```python
import requests

def get_weather(lat, lon):
    point = requests.get(f"https://api.weather.gov/points/{lat},{lon}").json()
    forecast_url = point["properties"]["forecastHourly"]
    
    forecast = requests.get(forecast_url).json()
    current = forecast["properties"]["periods"][0]
    
    return {
        "wind_speed_mph": current["windSpeed"],
        "wind_direction": current["windDirection"],
        "temperature_f": current["temperature"],
        "humidity_pct": current.get("relativeHumidity", {}).get("value", 30)
    }

def calc_fuel_moisture(temp_f, humidity_pct):
    # Simplified dead fuel moisture equilibrium
    return 0.942 * (humidity_pct ** 0.679) + 0.4 * (temp_f / 100)
```

**Data sources:**
- NOAA/NWS API: https://api.weather.gov (free)
- OpenWeatherMap as backup (free tier: 1000 calls/day)
- Optional: On-site weather station (Kestrel 5500, ~$350)

**Hardware:** Internet connection, CPU only

---

### 3. Fuel Mapper

**Purpose:** Identifies vegetation type and density at each location. Grass burns fast but low intensity; timber burns slower but with more energy. Maps fuel moisture based on vegetation health (NDVI) and weather.

**How to build:**
- Load LANDFIRE fuel model rasters (13 Anderson or 40 Scott/Burgan fuel models)
- Optionally integrate satellite NDVI for live vegetation health
- Assign burn characteristics per fuel type

```python
FUEL_MODELS = {
    1: {"name": "short_grass", "base_ros": 2.5, "heat": 700},
    4: {"name": "chaparral", "base_ros": 4.0, "heat": 2500},
    8: {"name": "timber_litter", "base_ros": 0.5, "heat": 1200},
    10: {"name": "timber_understory", "base_ros": 2.0, "heat": 2800},
    # ... 40 total models
}

def get_fuel_at(lat, lon, fuel_raster):
    row, col = fuel_raster.index(lon, lat)
    fuel_code = fuel_raster.read(1)[row, col]
    return FUEL_MODELS.get(fuel_code, FUEL_MODELS[1])
```

**Data sources:**
- LANDFIRE: https://landfire.gov/ (fuel models, canopy cover, vegetation type)
- Sentinel-2 NDVI for live moisture estimation

**Hardware:** Storage for fuel rasters (~100MB per 100km²), CPU only

---

### 4. Spread Predictor

**Purpose:** The core prediction engine. Takes current fire perimeter + terrain + weather + fuel and simulates where the fire will be in 1hr, 6hr, 24hr. Outputs predicted perimeters and spread direction vectors.

**Approach:** Grid-based cellular automata with Rothermel spread physics

**Key variables affecting fire spread:**
| Category | Variables | Effect |
|----------|-----------|--------|
| Topography | Slope angle, aspect | Fire spreads ~2x faster per 10° uphill |
| Wind | Speed, direction | Exponential effect on spread rate |
| Fuel | Type, moisture, density | Determines base burn rate and intensity |
| Humidity | Relative humidity | Dampens spread through fuel moisture |

```python
import numpy as np

class SpreadPredictor:
    def __init__(self, terrain, fuel, weather, cell_size=30):
        self.elevation, self.slope, self.aspect = terrain
        self.fuel = fuel
        self.weather = weather
        self.cell_size = cell_size  # meters
        self.fire_state = np.zeros_like(self.elevation)  # 0=unburned, 1=burning, 2=burned
        
    def set_ignition(self, fire_perimeter_coords):
        for lat, lon in fire_perimeter_coords:
            row, col = self.coords_to_grid(lat, lon)
            self.fire_state[row, col] = 1
    
    def step(self, dt_minutes=10):
        new_state = self.fire_state.copy()
        
        for row, col in self.get_burning_cells():
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = row + dr, col + dc
                if self.fire_state[nr, nc] == 0:
                    ros = self.calc_spread_rate(row, col, nr, nc)
                    distance = self.cell_size * (1.414 if dr and dc else 1)
                    time_to_spread = distance / ros
                    
                    if time_to_spread <= dt_minutes:
                        new_state[nr, nc] = 1
            
            new_state[row, col] = 2
        
        self.fire_state = new_state
    
    def calc_spread_rate(self, from_row, from_col, to_row, to_col):
        fuel = self.fuel[to_row, to_col]
        slope = self.slope[to_row, to_col]
        
        # Wind alignment with spread direction
        spread_dir = np.degrees(np.arctan2(to_col - from_col, from_row - to_row))
        wind_alignment = np.cos(np.radians(self.weather["wind_dir"] - spread_dir))
        wind_factor = 1 + 0.05 * (self.weather["wind_speed"] ** 1.5) * max(0, wind_alignment)
        
        # Slope effect (uphill = faster)
        slope_factor = 1 + 5.275 * (np.tan(np.radians(slope)) ** 2)
        
        # Moisture dampening
        moisture_factor = max(0.1, 1 - self.weather["fuel_moisture"] / 30)
        
        return fuel["base_ros"] * wind_factor * slope_factor * moisture_factor
    
    def predict(self, hours=6):
        predictions = {}
        steps_per_hour = 6  # 10-min steps
        
        for h in [1, 3, 6]:
            if h <= hours:
                for _ in range(h * steps_per_hour):
                    self.step()
                predictions[f"{h}hr"] = self.get_perimeter_coords()
        
        return predictions
```

**Hardware:** CPU for physics-based simulation, GPU optional for ML-enhanced prediction

---

## Stage 3: Suppression Support (Future Work)

### Core Concept

Stage 3 does not attempt autonomous fire suppression — no drone-scale payload can extinguish an established wildfire. Instead, Stage 3 uses Stage 2's predictions to **place protective barriers ahead of the fire**, slowing spread and buying time for human responders.

**Key insight:** A strategic retardant line *where the fire will be* uses far less material than encircling the current fire or direct suppression.

### Why Autonomous Suppression Fails

| Challenge | Reality |
|-----------|---------|
| Payload mismatch | Heavy-lift drone carries 20-40L; a small brush fire needs hundreds of liters minimum |
| Fire growth rate | Fire doubles in size every few minutes in dry conditions — suppression capacity is fixed |
| Narrow intervention window | By detect → fly → drop, fire has outgrown what payload can handle |
| Coordination complexity | Autonomous aircraft in active fire zones with manned helicopters and ground crews |

### The Alternative: Predictive Barrier Placement

Use Stage 2 predictions to drop retardant **where fire will spread**, not where it currently burns:

1. Stage 2 outputs predicted perimeters at 1hr, 3hr, 6hr
2. System identifies high-value areas in predicted path (structures, roads, critical infrastructure)
3. Calculates optimal barrier line to protect those areas
4. Recommends or executes retardant drops ahead of the fire front

**Value proposition:** Firefighters arriving on scene find fire already slowed or redirected away from structures.

---

### Fire Retardant Technology

#### Long-Term Retardants (LTR) — The Red Stuff

Stays effective after water evaporates. Can be pre-applied hours before fire arrives.

| Product | Chemistry | Mechanism |
|---------|-----------|-----------|
| Phos-Chek LC95A | Ammonium phosphate | Phosphate inhibits combustion; coats fuel so it chars instead of burning |
| Phos-Chek MVP-Fx | Ammonium polyphosphate | Same mechanism, improved viscosity for aerial drops |
| Fortress FR | Magnesium chloride + phosphate | Less corrosive, works at higher temps |

**Constraints:**
- Toxic to aquatic life — cannot drop within 300ft of waterways
- Corrosive to metal/equipment
- Red dye for visibility

---

#### Gel Retardants — Structure Protection

Water-absorbing polymer gels that stick to surfaces and hold moisture for hours.

| Product | Use Case |
|---------|----------|
| Barricade | Applied to homes/structures before fire arrives |
| ThermGel | Adheres to vegetation, slow-release moisture |
| FireIce | Similar polymer gel for home defense |

**For drone use:** Could pre-coat vegetation in predicted fire path. Gel holds water in place, creating moisture barrier fire must push through.

---

#### Class A Foam — Lightweight Option

Surfactant added to water. Breaks surface tension so water penetrates fuel better; creates smothering blanket.

| Product | Concentration |
|---------|---------------|
| Phos-Chek WD881 | 0.1-1% mixed with water |
| Silv-Ex | 0.5-1% |

**Advantage for drones:** Mostly water + small concentrate amount. 20L tank with 0.5% foam concentrate ≈ effectiveness of 60-80L plain water.

---

#### Emerging Technology

| Technology | Status | Notes |
|------------|--------|-------|
| Iron-based retardants | Research | Less toxic, potentially more effective |
| Cellulose-based gels | Commercial | Biodegradable, food-safe |
| Encapsulated retardants | Experimental | Time-release capsules dropped ahead of fire |
| Drone-deployed sprinklers | Prototype | Drones drop small sprinkler units that activate when fire approaches |

---

### Dashboard Integration: Barrier Planning Tool

**Concept:** Firefighter tablet interface for interactive barrier placement

#### Features

1. **Prediction overlay** — Predicted perimeters at 1hr, 3hr, 6hr on map
2. **Risk layer** — Structures, roads, critical infrastructure highlighted
3. **Barrier drawing tool** — Swipe to mark a protection line
4. **Resource calculator** — System computes:
   - Length of barrier line
   - Retardant volume required
   - Number of drone sorties or tanker drops
   - Optimal drop sequence (uphill first? wind-facing edge?)
5. **Execution mode** — Either:
   - Autonomous dispatch (drone swarm)
   - Recommendations for manned tanker drops
   - "Approve and execute" hybrid

#### Use Cases

| Scenario | Action |
|----------|--------|
| Fire heading toward subdivision | Draw barrier between fire and homes; system dispatches drones with gel |
| Fire approaching highway | Recommend tanker drop coordinates to protect evacuation route |
| Crew en route, 20 min out | Pre-position barrier to slow fire before they arrive |

---

### Stage 3 Components (Future)

| Component | Purpose | Technology |
|-----------|---------|------------|
| Barrier Planner | Calculate optimal protection lines from Stage 2 predictions | Optimization algorithm, constraint solver |
| Resource Allocator | Determine retardant type, volume, delivery method | Decision logic based on fire size, assets at risk |
| Drop Coordinator | Sequence and dispatch drone/tanker drops | Mission planning, airspace deconfliction |
| Payload Drone | Deliver retardant to target coordinates | Heavy-lift drone (DJI Agras, custom build) |

---

### Hardware for Stage 3

| Item | Purpose | Est. Cost |
|------|---------|-----------|
| Heavy-lift drone (DJI Agras T40) | 40L payload capacity | ~$20,000 |
| Spray system | Retardant dispersal | ~$2,000 |
| Retardant concentrate | Class A foam or gel | ~$50-200/gallon |
| Ground control station | Mission planning, swarm coordination | ~$5,000 |

---

### Research Questions for Stage 3

1. What is the minimum effective barrier width for different fuel types?
2. How far ahead of fire front should retardant be placed to maximize effectiveness?
3. Can swarm coordination enable meaningful coverage with small payloads?
4. What regulatory pathway exists for autonomous retardant drops in active fire zones?
5. How does cost-per-protected-acre compare between drone-delivered barriers vs. manned tanker drops?

---

## Evaluation Strategy

### Detection Accuracy (Stage 1)
- **Dataset:** FLAME 2 (real fire/smoke imagery)
- **Metrics:** mAP, precision, recall, F1 at various confidence thresholds
- **Note:** YOLO model trained on real imagery; synthetic MuJoCo fires won't transfer

### System-Level Validation (Stages 1-2)
- **Platform:** MuJoCo simulation with SITL drone
- **What to measure:**
  - Flight path execution accuracy
  - Tracking consistency (ID maintenance across frames)
  - Georeferencing error (sim knows ground truth camera pose)
  - Alert latency (detection → notification)
  - Spread prediction accuracy (compare predicted vs actual perimeters)
- **Note:** Sim provides RGB/depth/segmentation only — no thermal, fire, or smoke. Use synthetic markers or inject detections.

### Recommended Split
| Claim | Validation Method |
|-------|-------------------|
| "Detects fire/smoke with X% accuracy" | FLAME 2 dataset evaluation |
| "Tracks detections across frames" | MuJoCo simulation |
| "Georeferencing error < Y meters" | MuJoCo (ground truth pose available) |
| "Alerts within Z seconds" | MuJoCo end-to-end timing |
| "Predicts spread with W% accuracy" | Historical NIFC perimeters |

---

## Component Summary

### Stage 1: Detection

| Component | Key Libraries | Compute | Est. Cost |
|-----------|---------------|---------|-----------|
| Detector | ultralytics, torch | Jetson Orin Nano | ~$500 |
| Tracker | Built into ultralytics | Same as detector | — |
| GeoRef | numpy, pyproj | CPU (lightweight) | — |
| Alerter | twilio, requests | CPU | ~$0.01/SMS |

### Stage 2: Prediction

| Component | Key Libraries | Data Source | Compute |
|-----------|---------------|-------------|---------|
| Terrain Analyzer | rasterio, elevation, numpy | USGS 3DEP | CPU |
| Weather Integrator | requests | NOAA/NWS API | CPU |
| Fuel Mapper | rasterio | LANDFIRE | CPU |
| Spread Predictor | numpy, custom CA | Stage 1 + above | CPU |

### Stage 3: Suppression Support (Future)

| Component | Purpose | Status |
|-----------|---------|--------|
| Barrier Planner | Calculate optimal protection lines | Concept |
| Resource Allocator | Retardant type/volume decisions | Concept |
| Drop Coordinator | Mission planning and dispatch | Concept |
| Payload Drone | Retardant delivery | Hardware exists |

### Shared Components

| Component | Purpose |
|-----------|---------|
| **Dashboard** | Web UI for live detection, predicted spread overlays, barrier planning, crew positioning |
| **Drone** | Sensor platform (Stages 1-2), retardant delivery platform (Stage 3) |

---

## Data Sources Reference

| Data | Source | Resolution | Cost |
|------|--------|------------|------|
| Elevation/DEM | USGS 3DEP | 10m | Free |
| Fuel models | LANDFIRE | 30m | Free |
| Weather | NOAA/NWS API | Point-based | Free |
| Vegetation health | Sentinel-2 NDVI | 10m | Free |
| Historical fires | NIFC | Daily perimeters | Free |
