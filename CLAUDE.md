# EmberEye

Drone-based wildfire detection system using multi-sensor fusion (RGB + thermal).

## Project Overview

EmberEye detects wildfires from UAV footage by fusing RGB and thermal imagery. The system is designed to be drone-agnostic — any platform that can provide frames flows through the same pipeline.

**Owner:** Will Scott (wkscott715@gmail.com)  
**GitHub:** https://github.com/WKS-DS/EmberEye

## Current status
_Last updated: 2026-10-19_

- **Branch:** review-fixes (not pushed). main = sim adapter, 3 commits not pushed. origin/main on GitHub = initial scaffold + 2 web edits from 2026-10-19 (not fetched here), so main and origin/main have diverged.
- **Done:** review fixes Phases 0–3; Phase 4.1–4.3 (labelling module + audit); Phase 4 decisions (in `docs/ARCHITECTURE.md`, Detector → Labelling decisions); hand-check export (48 frames) + `scripts/score_handcheck.py`; Phase 5a scoring definition. 75 tests pass.
- **Waiting on Will:** labelme masks for the 48 frames in `data/flame2/handcheck/`, then the scores. This blocks Phase 4.4 (block split + YOLO-seg export).
- **Next for Claude Code:**
  1. Phase 5b: harder scenario + lawnmower path.
  2. Workflow setup: pre-commit + gitleaks, CI, subagents, settings/hooks. First fetch origin, show Will the two web commits, and rebase main onto origin/main with him.
  3. Phase 4.4, only after Will sends the scores. Raise first: `mask_to_yolo_seg` loses ~0.14 IoU vs the raw mask (it drops regions by outline area under 20, and simplifies with epsilon 1.0).
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
│   ├── masks.py        # mask <-> outline polygons, IoU (shared by sim and labelling)
│   ├── pipeline/       # Detection pipeline (WIP)
│   └── run.py          # Entry point
├── notebooks/
│   ├── 01_flame_exploration.ipynb
│   └── 02_generate_annotations.ipynb
├── data/
│   ├── flame2/         # FLAME 2 dataset (not in git)
│   │   ├── frames/     # 53,451 RGB + thermal pairs
│   │   ├── labels.txt  # Fire/smoke labels
│   │   ├── audit/      # Output of scripts/audit_flame2_labels.py
│   │   └── handcheck/  # 48 thermal frames + manifest.csv + Will's labelme JSON
│   └── sim/            # Simulator output (not in git)
├── scripts/
│   ├── audit_flame2_labels.py  # Evidence for reviewing the labelling method
│   ├── export_handcheck.py     # Pick the hand-check frames (8 per stratum)
│   └── score_handcheck.py      # IoU of hot_mask vs hand masks, per stratum
├── tests/
│   ├── test_flame2.py
│   ├── test_geo.py
│   ├── test_masks.py
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

1. ✅ Dataset downloaded, labels parsed (53,451 frame pairs: 39,751 fire, 13,700 no-fire)
2. 🔄 YOLO11-seg polygon labels from palette-aware thermal masks (hand-check, then export). Notebook 02's fixed threshold only works before frame ~17,000 and is being replaced.
3. ⬜ Train YOLO11-seg on annotated data
4. ⬜ Integrate detector into pipeline

### Simulation Proof of Concept

Build order from `docs/ARCHITECTURE.md`. The sim tests everything except the detector.

1. ✅ Sim adapter — SpreadPredictor fire on a ground grid, scripted MuJoCo camera, exact pose
2. ⬜ GeoRef + Fire Mapper — score against sim ground truth
3. ⬜ Arrival-time predictor — test with wrong wind/fuel vs ground truth
4. 🔄 FLAME 2 detector — parallel track (the list above)

How the sim works: `FireSim.run()` yields a `SimObservation` per frame with the RGB image, a fire mask read from MuJoCo's segmentation render (no ML), fire outlines as pixel polygons (same layout as YOLO `masks.xy`), the exact camera pose, and the true state of every ground cell.

### Annotation Strategy

FLAME 2 has frame-level labels (Fire/NoFire), not masks, so masks are generated from the thermal frames. A fixed grayscale threshold does **not** work: the thermal frames are false-colour JPEGs, the palette switches from an inferno-style one to a rainbow one at about frame 17,000 (whose background is mostly dark red in some frames, magenta in others), and the colours appear to be rescaled every frame. The labelling logic lives in `src/embereye/data/flame2.py` (palette detection, per-palette hot-region masks, YOLO-seg export), so notebooks call it instead of re-implementing it. See `docs/ARCHITECTURE.md` (Detector → Training data strategy, Labelling decisions).

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
