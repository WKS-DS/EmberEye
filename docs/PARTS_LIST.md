# EmberEye Prototype — Parts List and Justification

**Project:** EmberEye: Aerial Thermal Situational Awareness for Wildland Fire Response
**Author:** William [Last Name]
**Date:** [insert]
**Estimated total:** $950 (base) / $1,220 (recommended)

---

## Design principle

The EmberEye software stack is designed to be **drone-agnostic** — it accepts
thermal + RGB + telemetry from any compatible source through an adapter
interface. This parts list funds a **low-cost custom prototype** for
development and validation only. The resulting software could be deployed
on any commercial drone (DJI M3T, Autel EVO Dual, Skydio X10) via a
platform-specific adapter, at zero additional software cost to end users.

This means: **the prototype's job is to prove the software, not to become
the deployment platform.** Departments already own drones; we let them
keep them.

---

## Base build (~$950)

| # | Component | Purpose | Est. cost | Notes |
|---|-----------|---------|-----------|-------|
| 1 | **Holybro X500 V2 ARF Kit (PX4)** | Airframe, motors, ESCs, Pixhawk 6C flight controller, GPS, telemetry radio, power module | $400 | Educational-grade open-source drone platform |
| 2 | **Raspberry Pi 5 (8GB) + case + 128GB microSD** | Onboard AI compute for real-time inference | $120 | Runs YOLO models at ~1–3 fps. TODO(Will): reconcile Pi 5 vs Jetson Orin Nano (`docs/ARCHITECTURE.md` plans a Jetson) |
| 3 | **FLIR Lepton 2.5 + PureThermal Mini Pro breakout** | Thermal imaging sensor (80×60 radiometric) | $200 | Actual temperature readings, not just relative heat |
| 4 | **Raspberry Pi Camera Module 3** | RGB visible-light imaging | $35 | 12MP, autofocus, wide-angle available |
| 5 | **3S 5200mAh LiPo batteries (×3)** | Flight power | $75 | Two for rotation, one spare |
| 6 | **RadioMaster Pocket ELRS transmitter + receiver** | Manual flight control (safety fallback) | $75 | Lets a pilot take over manually during test flights |
| 7 | **XT60 connectors, mounting hardware, misc** | Assembly | $30 | Screws, standoffs, zip ties, cables |
| 8 | **Propellers (spare set)** | Consumables | $15 | Break-and-replace |
|   | **BASE SUBTOTAL** | | **$950** | |

---

## Recommended upgrades ($270 additional)

| # | Component | Purpose | Est. cost | Notes |
|---|-----------|---------|-----------|-------|
| 9 | **FLIR Lepton 3.5 upgrade** (in place of Lepton 2.5) | Higher resolution thermal (160×120) | +$150 | Significantly better detection quality; adjust line 3 |
| 10 | **5.8GHz analog video transmitter + goggles** | Live FPV monitoring during flight | +$120 | Safety and situational awareness during test flights |
|   | **RECOMMENDED TOTAL** | | **~$1,220** | |

---

## Zero-cost software stack

All software components are open-source or free-tier for students:

- **PyTorch** (model training and inference)
- **Ultralytics YOLO11** (pretrained detection and segmentation models)
- **OpenCV** (thermal image processing)
- **FastAPI + PostgreSQL/PostGIS** (backend, spatial data)
- **React + Leaflet** (dashboard, mapping)
- **PX4 + MAVSDK-Python** (drone control and telemetry)
- **MuJoCo** (simulated drone camera over a simulated fire, for Semester 1 development; PX4 SITL is a possible later integration)
- **Roboflow (free tier)** (dataset management)
- **Weights & Biases (student tier)** (experiment tracking)

**Public datasets used for training (no cost):**
- FLAME dataset (Northern Arizona University wildland fire aerial imagery)
- FLAME 2 (thermal + RGB paired dataset)
- BoWFire dataset (general fire imagery)
- Corsican Fire Database

---

## Existing resources leveraged (no additional cost)

- Personal laptop for development and ground station
- Gonzaga HPC cluster access with NVIDIA H100 GPUs (for model training)
- FAA Part 107 certification (in progress by student)

---

## Test site plan

Semester 1 development uses a MuJoCo fire simulator (PX4 SITL possibly
later) and public datasets. Semester 2 field validation is planned through partnership with
either Spokane County Fire District, Washington DNR, or a local wildland
fire training facility — no additional cost anticipated. Backup: any
outdoor location with a legal small controlled heat source (propane
firepit) and volunteer "personnel" for detection validation.

---

## Deliverables produced with this funding

1. Working drone prototype (physical hardware)
2. Open-source drone-agnostic software stack (published to GitHub)
3. Trained ML models with documented performance metrics
4. Web-based incident commander dashboard
5. Field-validated demonstrations (recorded)
6. Written thesis documenting methodology, results, and limitations
7. Public dataset contribution (annotated thermal wildland imagery)

---

## Mission alignment

This project directly serves Gonzaga's Jesuit mission of *cura personalis*
and service to community through the development of technology that
supports first responders in life-threatening emergencies. Wildland
firefighting has significant loss-of-life risk (approximately 15–25
fatalities annually in the US), and improved situational awareness is one
of the most cited needs by field responders. The software's drone-agnostic
architecture ensures it can be adopted by resource-limited departments
using existing equipment, rather than requiring purchase of proprietary
systems.

---

## Recommendation to committee

I request funding at the **base subtotal of $950**, with the option to
increase to **$1,220** if funding permits the recommended thermal
resolution upgrade — which meaningfully improves the quality of the
capstone deliverable. If partial funding is available, priority order is:
frame (1), Pi (2), Lepton 2.5 (3), Pi Camera (4), batteries (5), radio
(6), then upgrades.
