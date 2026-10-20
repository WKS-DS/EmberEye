# EmberEye

Drone-based wildfire detection system using multi-sensor fusion (RGB + thermal).

## Project Overview

EmberEye detects wildfires from UAV footage by fusing RGB and thermal imagery. The system is designed to be drone-agnostic — any platform that can provide frames flows through the same pipeline.

**Owner:** Will Scott (wkscott715@gmail.com)  
**GitHub:** https://github.com/WKS-DS/EmberEye  
**Status:** Phase 1 — Detector Development

## Architecture

```
DroneAdapter (async frame source)
    ↓
Pipeline
    ├── Detector (YOLOv11 fire/smoke detection)
    ├── Tracker (multi-frame object tracking)
    ├── GeoRef (pixel coords → lat/lon)
    └── Alerter (fire notifications)
    ↓
Dashboard (FastAPI + web UI)
```

## Key Directories

```
EmberEye/
├── src/embereye/
│   ├── adapters/       # Drone platform adapters
│   │   ├── base.py     # DroneAdapter ABC
│   │   └── recorded.py # Playback from files
│   ├── models/
│   │   └── types.py    # Frame, Detection, Track, etc.
│   ├── pipeline/       # Detection pipeline (WIP)
│   └── run.py          # Entry point
├── notebooks/
│   ├── 01_flame_exploration.ipynb
│   └── 02_generate_annotations.ipynb
├── data/
│   └── flame2/         # FLAME 2 dataset (not in git)
│       ├── frames/     # 53,451 RGB + thermal pairs
│       └── labels.txt  # Fire/smoke labels
├── tests/
└── docs/
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
3. ✅ Thermal thresholding pipeline built
4. 🔄 Generate YOLO bounding box annotations
5. ⬜ Train YOLOv11 on annotated data
6. ⬜ Integrate detector into pipeline

### Annotation Strategy

FLAME 2 has frame-level labels (Fire/NoFire), not bounding boxes. We generate pseudo-labels by thresholding the thermal channel:

```python
# Threshold=180 works well
gray = cv2.cvtColor(thermal, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

## Git Workflow

Commit daily. Small commits are fine.

```bash
git add -A
git commit -m "Descriptive message"
git push origin main
```

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run with recorded footage
python -m embereye.run --adapter recorded --input data/recordings/demo/

# Run tests
pytest
```

## Data Files (not in git)

Large files live in `data/` which is gitignored:
- `data/flame2/` — FLAME 2 dataset (~8GB)
- `data/recordings/` — Test video files
- `data/weights/` — Trained model weights

## Dependencies

Core (now):
- numpy, opencv-python, pytest

Coming soon:
- ultralytics (YOLOv11)
- torch
- fastapi, uvicorn (dashboard)

## Notes for Claude Code

- Prefer explicit over clever code
- Always update this file when project structure changes
- Commit work at logical stopping points
- Test on fire frame 13701+ and no-fire frame 1-13700 for quick validation
