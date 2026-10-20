# EmberEye

Drone-based wildfire detection system using multi-sensor fusion (RGB + thermal).

## Project Overview

EmberEye detects wildfires from UAV footage by fusing RGB and thermal imagery. The system is designed to be drone-agnostic — any platform that can provide frames flows through the same pipeline.

**Owner:** Will Scott (wkscott715@gmail.com)  
**GitHub:** https://github.com/WKS-DS/EmberEye  
**Status:** Phase 1 — detector development (labelling under review) and simulation proof of concept (sim adapter done)

## Current status
_Last updated: 2026-10-19_

- **Branch:** review-fixes (not pushed). main = sim adapter, 3 commits not pushed. origin/main on GitHub = initial scaffold + 2 web edits from 2026-10-19 (not fetched here), so main and origin/main have diverged.
- **Done:** review fixes Phases 0–3; Phase 4.1–4.3 (FLAME 2 labelling module + audit); Phase 5a scoring definition (added in Phase 2). 68 tests pass.
- **Waiting on Will:** hand-check of thermal masks (48 frames). This blocks Phase 4.4 (block split + YOLO-seg export).
- **Next for Claude Code:**
  1. Phase 4 decisions (gate, hand-check export + scoring script, smoky-frame policy, scene-confound note). Will re-sends these.
  2. Phase 5b: harder scenario + lawnmower path.
  3. Workflow setup: pre-commit + gitleaks, CI, subagents, settings/hooks.
  4. Build step 2: GeoRef + Fire Mapper. Include p90 latency metrics, a timestamped event log and GeoJSON output.
- **Open decisions:** Pi 5 vs Jetson Orin Nano (TODO(Will) in `docs/PARTS_LIST.md` and `docs/ARCHITECTURE.md`); the repo URL mismatch (claude.ai project says `wkscott/EmberEye`, git remote is `WKS-DS/EmberEye`).

## Architecture

```
DroneAdapter (async frame source: recorded, sim, later a real drone)
    ↓
Stage 1 pipeline
    ├── Detector (YOLO11-seg fire masks)
    ├── GeoRef (pixel → ground point: camera ray + terrain)
    ├── Temporal Filter (confirm fire per ground cell, N of last M views)
    ├── Fire Mapper (ground map of confirmed fire)
    └── Alerter (one alert per fire region)
    ↓
Stage 2: Spread Predictor (prediction/)
    ↓
Dashboard (FastAPI + web UI)
```

The full three-stage design (detection, spread prediction, suppression support) is in `docs/ARCHITECTURE.md`.

## Key Directories

```
EmberEye/
├── src/embereye/
│   ├── adapters/       # Drone platform adapters
│   │   ├── base.py     # DroneAdapter ABC
│   │   ├── recorded.py # Playback from files
│   │   └── sim.py      # Simulated drone over a simulated fire
│   ├── models/
│   │   └── types.py    # Frame, Detection, Track, etc.
│   ├── prediction/     # Stage 2
│   │   ├── terrain.py  # Elevation grid -> slope and aspect
│   │   └── spread.py   # SpreadPredictor (arrival-time fire spread)
│   ├── sim/            # MuJoCo fire simulator (ground truth for testing)
│   │   ├── camera.py   # CameraModel, CameraPose, OrbitPath
│   │   ├── world.py    # MuJoCo scene: ground tiles + camera
│   │   ├── fire_sim.py # FireSim, Scenario, SimObservation
│   │   └── record.py   # Save a simulated flight to disk
│   ├── data/
│   │   └── flame2.py   # FLAME 2 labels, palette detection, hot masks, YOLO-seg lines
│   ├── geo.py          # LocalFrame (GPS <-> metres), GroundGrid
│   ├── masks.py        # mask -> outline polygons (shared by sim and labelling)
│   ├── pipeline/       # Detection pipeline (WIP)
│   └── run.py          # Entry point
├── notebooks/
│   ├── 01_flame_exploration.ipynb
│   └── 02_generate_annotations.ipynb
├── data/
│   ├── flame2/         # FLAME 2 dataset (not in git)
│   │   ├── frames/     # 53,451 RGB + thermal pairs
│   │   └── labels.txt  # Fire/smoke labels
│   │   └── audit/      # Output of scripts/audit_flame2_labels.py
│   └── sim/            # Simulator output (not in git)
├── scripts/
│   └── audit_flame2_labels.py  # Evidence for reviewing the labelling method
├── tests/
│   ├── test_flame2.py
│   ├── test_geo.py
│   ├── test_recorded.py
│   ├── test_sim.py
│   ├── test_spread.py
│   └── test_types.py
├── docs/
│   ├── ARCHITECTURE.md
│   └── PARTS_LIST.md
└── venv.nosync/        # Python environment (not in git)
```

## Code Conventions

- **Python 3.11+** with type hints everywhere
- **Async/await** for I/O-bound operations (frame streaming)
- **Dataclasses** for domain types (see `models/types.py`)
- **pytest** for testing
- Imports: stdlib → third-party → local (blank line between groups)

## Current Work

### Detector Development (Phase 1)

1. ✅ Dataset downloaded (FLAME 2 — 53,451 frame pairs)
2. ✅ Labels parsed (39,751 fire frames, 13,700 no-fire)
3. ⚠️ Thermal thresholding pipeline built (notebook 02), but it only works before the palette change at about frame 17,000; being replaced
4. 🔄 Generate YOLO11-seg polygon labels from palette-aware thermal masks (under review before any full-dataset export)
5. ⬜ Train YOLO11-seg on annotated data
6. ⬜ Integrate detector into pipeline

### Simulation Proof of Concept

Build order from `docs/ARCHITECTURE.md`. The sim tests everything except the detector.

1. ✅ Sim adapter — SpreadPredictor fire on a ground grid, scripted MuJoCo camera, exact pose
2. ⬜ GeoRef + Fire Mapper — score against sim ground truth
3. ⬜ Arrival-time predictor — test with wrong wind/fuel vs ground truth
4. 🔄 FLAME 2 detector — parallel track (the list above)

How the sim works: `FireSim.run()` yields a `SimObservation` per frame with the RGB image, a fire mask read from MuJoCo's segmentation render (no ML), fire outlines as pixel polygons (same layout as YOLO `masks.xy`), the exact camera pose, and the true state of every ground cell.

### Annotation Strategy

FLAME 2 has frame-level labels (Fire/NoFire), not masks, so masks are generated from the thermal frames. A fixed grayscale threshold does **not** work: the thermal frames are false-colour JPEGs, the palette switches from an inferno-style one to rainbow/magenta ones at about frame 17,000, and the colours appear to be rescaled every frame. After the switch, almost no pixel passes the old threshold of 200.

The labelling logic lives in `src/embereye/data/flame2.py` (palette detection, per-palette hot-region masks, YOLO-seg export), so notebooks call it instead of re-implementing it. See `docs/ARCHITECTURE.md` (Detector → Training data strategy).

## Git Workflow

Commit daily. Small commits are fine.

```bash
git add -A
git commit -m "Descriptive message"
git push origin main
```

## Running the Project

```bash
# One-time setup: create the environment and install everything
python3.11 -m venv venv.nosync
source venv.nosync/bin/activate
pip install -r requirements.txt
pip install -e .

# Every new terminal: switch the environment on
source venv.nosync/bin/activate

# Run with recorded footage
python -m embereye.run --adapter recorded --input data/recordings/demo/

# Run the fire simulator through the adapter interface
python -m embereye.run --adapter sim --duration 120

# Save a simulated flight (frames, masks, poses, preview.mp4) to data/sim/demo/
python -m embereye.sim.record --out data/sim/demo

# Audit the FLAME 2 labelling method (reads the dataset, writes data/flame2/audit/)
python scripts/audit_flame2_labels.py

# Run tests
pytest
```

The environment folder is called `venv.nosync` (not `.venv`) on purpose: the Desktop syncs to iCloud, which marks files in dot-folders as hidden, and Python then ignores the file that makes `import embereye` work. The `.nosync` ending also keeps iCloud from uploading the environment.

## Data Files (not in git)

Large files live in `data/` which is gitignored:
- `data/flame2/` — FLAME 2 dataset (~8GB)
- `data/recordings/` — Test video files
- `data/sim/` — Simulator output
- `data/weights/` — Trained model weights

## Dependencies

Core (now):
- numpy, opencv-python, pytest
- mujoco (fire simulator), pyproj (GPS <-> metres)

Notebooks:
- pandas, matplotlib, Pillow

Coming soon:
- ultralytics (YOLO11)
- torch
- fastapi, uvicorn (dashboard)

## Notes for Claude Code

- Prefer explicit over clever code
- Always update this file when project structure changes
- Commit work at logical stopping points
- Before ending a session, or after finishing a phase, update "Current status": change the date, move finished items to Done, and add new next steps. Keep it short. Detailed history belongs in git, not here.
- For quick checks, use frames from both sides of the thermal palette change at about frame 17,000 (e.g. 15000 and 30000) plus a no-fire frame from 1–13700. Labelling behaves completely differently on each side.
