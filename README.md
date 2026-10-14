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
Every drone-specific concern lives inside a **drone adapter**, one class
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
EmberEye/
├── docs/
│   ├── ARCHITECTURE.md             # Three-stage system design
│   └── PARTS_LIST.md               # Hardware budget for prototype
├── src/embereye/
│   ├── adapters/                   # Drone-platform-specific input layers
│   │   ├── base.py                 # Adapter interface — all others inherit this
│   │   ├── recorded.py             # Read pre-recorded flights (dev + reproducibility)
│   │   ├── sim.py                  # Simulated drone over a simulated fire (MuJoCo)
│   │   ├── rpi_custom.py           # Our physical prototype (planned)
│   │   └── dji.py                  # DJI SDK adapter (planned)
│   ├── pipeline/                   # Stage 1: detection
│   │   ├── detector.py             # YOLO11-seg fire masks (planned)
│   │   ├── georef.py               # Pixel → ground point: camera ray + terrain (planned)
│   │   ├── temporal_filter.py      # Confirm fire per ground cell (planned)
│   │   ├── fire_mapper.py          # Ground map of confirmed fire (planned)
│   │   └── alerter.py              # One alert per fire region (planned)
│   ├── prediction/                 # Stage 2
│   │   ├── terrain.py              # Elevation → slope and aspect
│   │   └── spread.py               # Arrival-time fire spread model
│   ├── sim/                        # MuJoCo fire simulator (ground truth for testing)
│   ├── geo.py                      # GPS <-> local metres, ground grid
│   ├── models/
│   │   └── types.py                # Frame, Detection, etc.
│   ├── dashboard/                  # Web UI backend, FastAPI (planned)
│   └── run.py                      # Entry point — wires an adapter to the pipeline
├── data/                           # Not in git
│   ├── flame2/                     # FLAME 2 dataset
│   └── sim/                        # Simulator output
├── tests/                          # Unit tests
└── notebooks/                      # Exploratory analysis, labelling, training
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

# Run against the built-in fire simulator (no hardware or footage needed)
python -m embereye.run --adapter sim --duration 120

# Save a simulated flight with a preview video
python -m embereye.sim.record --out data/sim/demo
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
This project is built to serve that mission first, technical
sophistication is in service of it, not the reverse.
