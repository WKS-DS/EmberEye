# EmberEye

Aerial thermal situational awareness for wildland fire response.

A drone-agnostic pipeline that ingests thermal and RGB imagery plus
telemetry from any compatible drone platform, detects and tracks fire,
personnel, and vehicles, and delivers real-time situational awareness
to an incident commander through a live web dashboard.

**Status:** Capstone project, in development.

---

## Core design principle

EmberEye's software stack does not care what drone it is talking to.
Every drone-specific concern lives inside a **drone adapter** — one class
that knows how to pull frames and telemetry from a specific platform and
convert them to EmberEye's internal format. Everything else in the
pipeline treats a DJI Mavic 3 Thermal, an Autel EVO Dual, a custom PX4
build, or a recorded video file identically.

This means departments can plug EmberEye into whatever drone they already
own. It also means we can develop the entire ML and dashboard stack against
recorded footage, without needing a drone in the air every day.

---

## Repository layout

```
embereye/
├── docs/
│   └── PARTS_LIST.md               # Hardware budget for prototype
├── src/embereye/
│   ├── adapters/                   # Drone-platform-specific input layers
│   │   ├── base.py                 # Adapter interface — all others inherit this
│   │   ├── recorded.py             # Read pre-recorded flights (dev + reproducibility)
│   │   ├── simulated.py            # PX4 SITL / Gazebo interface (later)
│   │   ├── rpi_custom.py           # Our physical prototype (later)
│   │   └── dji.py                  # Stub for DJI SDK (later)
│   ├── pipeline/                   # The actual data science
│   │   ├── detector.py             # Object detection (RGB) + hotspot detection (thermal)
│   │   ├── fusion.py               # Combine RGB + thermal detections
│   │   ├── georef.py               # Pixel coordinates → real-world coordinates
│   │   ├── tracker.py              # Link detections across frames
│   │   └── fire_model.py           # Perimeter + spread prediction
│   ├── models/                     # Domain types shared by everything
│   │   └── types.py                # Frame, Detection, Track, etc.
│   ├── dashboard/                  # Web UI backend (FastAPI)
│   │   └── server.py
│   └── run.py                      # Entry point — wires an adapter to the pipeline
├── data/
│   ├── recordings/                 # Recorded flights for dev/testing
│   ├── models/                     # Trained ML model weights
│   └── annotations/                # Labeled datasets
├── tests/                          # Unit tests for each pipeline stage
└── notebooks/                      # Exploratory analysis, model training notebooks
```

---

## Quick start (once dependencies are installed)

```bash
# Install
pip install -r requirements.txt

# Run against a pre-recorded flight (no hardware needed)
python -m embereye.run --adapter recorded --input data/recordings/demo_flight.mp4

# Later: run against physical prototype
python -m embereye.run --adapter rpi_custom

# Later: run against simulated drone in Gazebo
python -m embereye.run --adapter simulated
```

---

## Roadmap

**Semester 1 (weeks 1–15) — Foundation:**
- Recorded-footage adapter (so we can develop with no hardware)
- Fire/smoke/person detector trained on FLAME dataset
- Basic dashboard scaffold
- End-to-end recorded-footage demo by week 15

**Semester 2 (weeks 16–30) — Real-time and field validation:**
- Custom prototype adapter (physical drone)
- Live pipeline with real-time inference
- Geo-referencing and temporal tracking
- Fire spread model
- Field validation with controlled tests
- Final demo and thesis

---

## Mission

Wildland firefighting is dangerous work with a genuine information gap
between the field and command. Improved situational awareness saves lives.
This project is built to serve that mission first — technical
sophistication is in service of it, not the reverse.
