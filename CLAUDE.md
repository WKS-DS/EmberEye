# EmberEye

Drone-based wildfire detection system using multi-sensor fusion (RGB + thermal).

## Project Overview

EmberEye detects wildfires from UAV footage by fusing RGB and thermal imagery. The system is designed to be drone-agnostic — any platform that can provide frames flows through the same pipeline.

**Owner:** Will Scott (wkscott715@gmail.com)  
**GitHub:** https://github.com/WKS-DS/EmberEye

## Current status
_Last updated: 2026-10-21_

- **Branch:** `feat/flame2-split-export` → PR #2 (Phase 4.4), open for review. `chore/workflow-setup` holds 3 local doc commits (status, README roadmap, labelme 6 steps), not pushed: rebase it onto main after PR #2 merges. Repo `WKS-DS/EmberEye`; commits use the WKS-DS noreply email (repo-local git config) and `gh` runs as WKS-DS.
- **Done:** M1 (sim, incl. Phase 5b); review fixes Phases 0–4.3; hand-check (48 frames, every stratum ≥ 0.53, mean 0.63: `docs/findings/flame2-label-validation.md`); YOLO-seg export now matches the mask (IoU 0.97); held-out rainbow tuning (no clear gain, defaults kept); Phase 4.4 split + export in `data/flame2/yolo_seg.nosync/` (train 38,715 / val 5,200 / test 5,302 frames, 34 excluded; loads in Ultralytics). 89 tests pass.
- **Waiting on Will:** review and merge PR #2.
- **Next for Claude Code:**
  1. Workflow setup, steps B–E of `docs/WORKFLOW.md`, on `chore/workflow-setup`.
  2. M4: train YOLO11-seg, RGB baseline first (`rgb/rgb_baseline.yaml`), then the YY ablation and thermal only. Fire-frame metrics are the headline: no-fire val/test frames have near-copies in train (the drone hovers before the burn), on top of the scene confound.
  3. M2: GeoRef + Fire Mapper. Include p90 latency metrics, a timestamped event log and GeoJSON output. Far cells can be smaller than a pixel (in frame but unseen), so the coverage mask must allow for that.
- **Open decisions:** Pi 5 vs Jetson Orin Nano (TODO(Will) in `docs/PARTS_LIST.md` and `docs/ARCHITECTURE.md`); license; whether to move the repo out of iCloud; forest no-fire negatives from another dataset (TODO(Will) in `docs/ARCHITECTURE.md`).

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
│   │   ├── camera.py   # CameraModel, CameraPose, OrbitPath, LawnmowerPath
│   │   ├── world.py    # MuJoCo scene: ground tiles + camera
│   │   ├── fire_sim.py # FireSim, SimObservation, demo/hard scenarios, make_path
│   │   └── record.py   # Save a simulated flight to disk
│   ├── data/
│   │   ├── flame2.py   # FLAME 2 labels, palette detection, hot masks, YOLO-seg lines
│   │   └── flame2_split.py  # Train/val/test blocks, gaps, split checks
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
│   │   ├── handcheck/  # 48 thermal frames + manifest.csv + Will's labelme JSON + scores
│   │   └── yolo_seg.nosync/  # Phase 4.4 export: split.json, report, rgb/ + thermal/ YOLO-seg sets
│   └── sim/            # Simulator output (not in git)
├── scripts/
│   ├── audit_flame2_labels.py  # Evidence for reviewing the labelling method
│   ├── export_handcheck.py     # Pick the hand-check frames (8 per stratum)
│   ├── score_handcheck.py      # IoU of hot_mask vs hand masks, per stratum
│   ├── tune_rainbow_mask.py    # Held-out tuning of hot_mask on rainbow frames
│   ├── export_flame2_yolo.py   # Block split + YOLO-seg labels for training
│   └── sim_visibility.py       # Per sim frame: share of burning cells in view
├── tests/              # One test_<module>.py per module (pytest); none need the dataset
├── docs/
│   ├── ARCHITECTURE.md
│   ├── PARTS_LIST.md
│   └── findings/       # One write-up per finding, e.g. flame2-label-validation.md
└── venv.nosync/        # Python environment (not in git)
```

## Code Conventions

- **Python 3.11+** with type hints everywhere
- **Async/await** for I/O-bound operations (frame streaming)
- **Dataclasses** for domain types (see `models/types.py`)
- **pytest** for testing
- Imports: stdlib → third-party → local (blank line between groups)

## Current Work

Detector (milestone M4), a parallel track to the sim:
1. ✅ Thermal-derived masks, hand-checked (mean IoU 0.63, `docs/findings/flame2-label-validation.md`)
2. ✅ Block split + YOLO-seg export (`scripts/export_flame2_yolo.py`)
3. ⬜ Train YOLO11-seg: RGB baseline (YN + no-fire), YY ablation, thermal only
4. ⬜ Integrate the detector into the pipeline

Simulation (the sim tests everything except the detector): ✅ M1 sim adapter → ⬜ M2 GeoRef + Fire Mapper → ⬜ M3 arrival-time predictor tested with wrong wind/fuel.

How the sim works: `FireSim.run()` yields a `SimObservation` per frame with the RGB image, a fire mask read from MuJoCo's segmentation render (no ML), fire outlines as pixel polygons (same layout as YOLO `masks.xy`), the exact camera pose, and the true state of every ground cell.

### Annotation Strategy

FLAME 2 has frame-level labels (Fire/NoFire), not masks, so masks are generated from the thermal frames. A fixed grayscale threshold does **not** work: the thermal frames are false-colour JPEGs, the palette switches from an inferno-style one to a rainbow one at about frame 17,000 (whose background is mostly dark red in some frames, magenta in others), and the colours appear to be rescaled every frame. The labelling logic lives in `src/embereye/data/flame2.py` (palette detection, per-palette hot-region masks, YOLO-seg export), so notebooks call it instead of re-implementing it; the train/val/test blocks are in `flame2_split.py`. See `docs/ARCHITECTURE.md` (Detector → Training data strategy, Labelling decisions).

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

# The harder test: big grid, two fires, a hill (writes to data/sim/hard-lawnmower/)
python -m embereye.sim.record --scenario hard --path lawnmower

# Per frame: share of burning cells in view, and render time (prints only)
python scripts/sim_visibility.py --scenario hard --path orbit

# Audit the FLAME 2 labelling method (reads the dataset, writes data/flame2/audit/)
python scripts/audit_flame2_labels.py

# Score hot_mask against Will's labelme masks in data/flame2/handcheck/ (labelme steps in the script)
python scripts/score_handcheck.py

# Block split + YOLO-seg labels for training (~10 min; rebuilds data/flame2/yolo_seg.nosync/)
python scripts/export_flame2_yolo.py

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

`requirements.txt` is the dependency list. Core: numpy, opencv-python, pytest, mujoco (fire simulator), pyproj (GPS <-> metres). Notebooks: pandas, matplotlib, Pillow. Coming soon: ultralytics (YOLO11), torch, fastapi + uvicorn (dashboard).

## Notes for Claude Code

- Prefer explicit over clever code
- Always update this file when project structure changes
- Commit work at logical stopping points
- Before ending a session, or after finishing a phase, update "Current status": change the date, move finished items to Done, and add new next steps. Keep it short. Detailed history belongs in git, not here.
- For quick checks, use frames from both sides of the thermal palette change at about frame 17,000 (e.g. 15000 and 30000) plus a no-fire frame from 1–13700. Labelling behaves completely differently on each side.
