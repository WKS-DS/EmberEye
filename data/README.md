# Datasets

Everything in `data/` is gitignored except this file. Datasets are referenced, never redistributed: download each one yourself into the folder given. Licences were checked at the source on 2026-10-24.

| Dataset | Used for | Folder | Source | Licence |
|---|---|---|---|---|
| FLAME 2 | Detector training and evaluation (M4) | `data/flame2/` (`frames/`, `labels.txt`) | [IEEE DataPort](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset), DOI [10.21227/swyw-6j78](https://doi.org/10.21227/swyw-6j78); needs a free IEEE account | **Not stated** on the dataset page. Without one, default copyright applies: use it for this research, don't redistribute it, and cite the dataset and its paper (IEEE Access, DOI [10.1109/ACCESS.2022.3222805](https://doi.org/10.1109/ACCESS.2022.3222805)). |
| FLAME (FLAME 1) | Candidate forest no-fire negatives (not downloaded yet) | `data/flame1/` | [IEEE DataPort](https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs), DOI [10.21227/qad6-r683](https://doi.org/10.21227/qad6-r683) | **Not stated**; same terms as FLAME 2. Paper: arXiv [2012.14036](https://arxiv.org/abs/2012.14036). |
| Next Day Wildfire Spread | Spread prediction benchmark (M3; not downloaded yet, ~4 GB) | `data/ndws/` | [Kaggle: fantineh/next-day-wildfire-spread](https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread); paper arXiv [2112.02447](https://arxiv.org/abs/2112.02447) | **CC BY 4.0**: reuse allowed with attribution. |
| LANDFIRE | Fuel models for spread prediction (Stage 2; not downloaded yet) | `data/landfire/` | [landfire.gov](https://landfire.gov/) | **Public domain** (USGS metadata: CC0 1.0, no use constraints). LANDFIRE asks that modified data be marked as modified. |
| USGS 3DEP | Elevation for terrain and GeoRef (not downloaded yet) | `data/3dep/` | [USGS National Map](https://apps.nationalmap.gov/downloader/) | **Public domain** (US government data). |

## Made by scripts, not downloaded

- `data/flame2/audit/`: `scripts/audit_flame2_labels.py`
- `data/flame2/handcheck/`: `scripts/export_handcheck.py`, plus the hand labels (keep these: they can't be regenerated)
- `data/flame2/yolo_seg.nosync/`: `scripts/export_flame2_yolo.py`
- `data/sim/`: `python -m embereye.sim.record`

## Never here

Partner data (anything shared by agencies or companies) lives outside the repo, e.g. `~/EmberEye-private/`. Model weights go to GitHub Releases, not git. See `docs/WORKFLOW.md`.
