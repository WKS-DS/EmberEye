"""
record.py — Run the fire simulator and save what it produced to disk.

Usage:
    python -m embereye.sim.record --out data/sim/demo
    python -m embereye.sim.record --scenario hard --path lawnmower    # writes to data/sim/hard-lawnmower

What gets written into the output folder:

    rgb/000000.png ...        the camera frames
    fire_mask/000000.png ...  white where the frame shows burning ground
    cell_ids/000000.png ...   which ground cell each pixel shows
                              (16-bit image: value = cell index + 1, 0 = none)
    truth.npz                 the true fire: arrival time, fuel and elevation per cell
    sim.json                  camera settings, grid, weather, and for every
                              frame the exact camera pose and fire outlines
    preview.mp4               a side-by-side video for a quick look:
                              camera view (left) and true map (right)

Code that wants the frames directly (tests, notebooks, the Fire Mapper)
does not need any of this — loop over FireSim.run() instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from embereye.sim.fire_sim import (
    BURNING,
    PATHS,
    SCENARIOS,
    FireSim,
    SimObservation,
    demo_scenario,
    make_path,
    visible_burning_fraction,
)

PREVIEW_FPS = 10
MAX_CELLS_IN_16_BIT_PNG = 65535     # largest value a 16-bit image can hold


def pose_to_dict(obs: SimObservation) -> dict:
    """The camera pose as plain numbers that JSON can store."""
    pose = obs.pose
    return {
        "east_m":      pose.east_m,
        "north_m":     pose.north_m,
        "up_m":        pose.up_m,
        "heading_deg": pose.heading_deg,
        "pitch_deg":   pose.pitch_deg,
        "lat":         pose.lat,
        "lon":         pose.lon,
        "alt_msl_m":   pose.alt_msl_m,
        "alt_agl_m":   pose.alt_agl_m,
        "rotation":    pose.rotation.tolist(),
    }


def draw_preview(sim: FireSim, obs: SimObservation) -> np.ndarray:
    """One preview video frame: camera view on the left, true map on the right."""
    height = sim.camera.height
    grid = sim.scenario.grid

    # --- Left: the camera frame with the fire outlines drawn on it.
    # OpenCV works in blue-green-red order, so convert before drawing.
    view = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
    outlines = [polygon.astype(np.int32) for polygon in obs.fire_polygons]
    cv2.polylines(view, outlines, isClosed=True, color=(255, 255, 255), thickness=1)
    label = (
        f"flight {obs.time_s:5.0f}s  fire {obs.fire_time_min:5.1f}min  "
        f"alt {obs.pose.alt_agl_m:4.0f}m  pitch {obs.pose.pitch_deg:5.1f}"
    )
    cv2.putText(view, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # --- Right: the true state of the ground seen from directly above,
    # north at the top. The map is scaled to fit a square panel as tall as
    # the camera frame, keeping the grid's shape, and centred with dark
    # padding on the short side.
    px_per_m = height / max(grid.width_m, grid.height_m)
    map_w = max(1, int(round(grid.width_m * px_per_m)))
    map_h = max(1, int(round(grid.height_m * px_per_m)))
    pad_x = (height - map_w) // 2
    pad_y = (height - map_h) // 2

    tile_rgb = (sim.tile_colors(obs.truth_state, obs.time_s) * 255).astype(np.uint8)
    ground = cv2.resize(cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR), (map_w, map_h), interpolation=cv2.INTER_NEAREST)
    top_down = np.full((height, height, 3), 40, dtype=np.uint8)
    top_down[pad_y:pad_y + map_h, pad_x:pad_x + map_w] = ground

    # Mark the drone and the direction it is looking.
    def to_map_px(east_m: float, north_m: float) -> tuple[int, int]:
        x = pad_x + (east_m + grid.width_m / 2) * px_per_m
        y = pad_y + (grid.height_m / 2 - north_m) * px_per_m
        return int(round(x)), int(round(y))

    heading = np.radians(obs.pose.heading_deg)
    drone = to_map_px(obs.pose.east_m, obs.pose.north_m)
    ahead = to_map_px(obs.pose.east_m + 80 * np.sin(heading), obs.pose.north_m + 80 * np.cos(heading))
    cv2.arrowedLine(top_down, drone, ahead, (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
    cv2.circle(top_down, drone, 5, (255, 200, 0), -1, cv2.LINE_AA)
    cv2.putText(top_down, "truth (N up)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return np.hstack([view, top_down])


def save_run(
    sim: FireSim,
    out_dir: Path,
    duration_s: float,
    frame_interval_s: float,
    preview: bool = True,
) -> int:
    """Fly the path and write every frame to `out_dir`. Returns the frame count."""
    out_dir = Path(out_dir)
    for name in ("rgb", "fire_mask", "cell_ids"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    scenario, camera, grid = sim.scenario, sim.camera, sim.scenario.grid
    save_cell_ids = grid.rows * grid.cols < MAX_CELLS_IN_16_BIT_PNG
    if not save_cell_ids:
        print("Grid has too many cells for a 16-bit image; skipping cell_ids/.")

    video = None
    frames = []
    for obs in sim.run(duration_s, frame_interval_s):
        name = f"{obs.index:06d}.png"
        cv2.imwrite(str(out_dir / "rgb" / name), cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "fire_mask" / name), obs.fire_mask.astype(np.uint8) * 255)
        if save_cell_ids:
            cv2.imwrite(str(out_dir / "cell_ids" / name), (obs.cell_ids + 1).astype(np.uint16))

        if preview:
            preview_frame = draw_preview(sim, obs)
            if video is None:
                h, w = preview_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video = cv2.VideoWriter(str(out_dir / "preview.mp4"), fourcc, PREVIEW_FPS, (w, h))
            video.write(preview_frame)

        frames.append({
            "index":           obs.index,
            "time_s":          obs.time_s,
            "fire_time_min":   obs.fire_time_min,
            "ts":              obs.ts.isoformat(),
            "pose":            pose_to_dict(obs),
            "fire_polygons":   [polygon.tolist() for polygon in obs.fire_polygons],
            "n_fire_pixels":   int(obs.fire_mask.sum()),
            "n_burning_cells": int((obs.truth_state == BURNING).sum()),
            "visible_burning_fraction": visible_burning_fraction(obs),
        })

    if video is not None:
        video.release()

    np.savez_compressed(
        out_dir / "truth.npz",
        arrival_time_min=sim.truth.arrival_time,
        fuel_codes=scenario.fuel_codes,
        elevation_m=scenario.terrain.elevation,
    )

    summary = {
        "camera": {
            "width":     camera.width,
            "height":    camera.height,
            "fov_v_deg": camera.fov_v_deg,
            "fov_h_deg": camera.fov_h_deg,
            "focal_px":  camera.focal_px,
        },
        "grid": {"rows": grid.rows, "cols": grid.cols, "cell_size_m": grid.cell_size_m},
        "origin": {
            "lat":      scenario.local_frame.origin_lat,
            "lon":      scenario.local_frame.origin_lon,
            "alt_m":    scenario.local_frame.origin_alt_m,
            "utm_epsg": scenario.local_frame.epsg,
        },
        "weather": {
            "wind_speed_mph":    scenario.weather.wind_speed_mph,
            "wind_to_deg":       scenario.weather.wind_to_deg,
            "fuel_moisture_pct": scenario.weather.fuel_moisture_pct,
        },
        "fire_age_at_start_min": scenario.fire_age_at_start_min,
        "burn_duration_min":     scenario.burn_duration_min,
        "frame_interval_s":      frame_interval_s,
        "frames":                frames,
    }
    (out_dir / "sim.json").write_text(json.dumps(summary, indent=1))
    return len(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EmberEye fire simulator and save the frames")
    parser.add_argument("--scenario",   choices=list(SCENARIOS), default="demo", help="Which simulated world")
    parser.add_argument("--path",       choices=PATHS, default="orbit", help="Which flight path")
    parser.add_argument("--out",        type=Path,  default=None,
                        help="Output folder (default: data/sim/demo for the demo orbit, else data/sim/<scenario>-<path>)")
    parser.add_argument("--duration",   type=float, default=1200.0, help="Length of the simulated flight, in seconds")
    parser.add_argument("--interval",   type=float, default=5.0,    help="Simulated seconds between frames")
    parser.add_argument("--terrain",    choices=["flat", "hill"], default=None, help="Shape of the ground (demo only; default flat)")
    parser.add_argument("--no-preview", action="store_true", help="Skip writing preview.mp4")
    args = parser.parse_args()

    if args.scenario == "demo":
        scenario = demo_scenario(args.terrain or "flat")
    elif args.terrain is not None:
        parser.error("--terrain only applies to the demo scenario")
    else:
        scenario = SCENARIOS[args.scenario]()

    # A separate default folder per scenario and path, so trying the new
    # ones never overwrites the saved demo.
    out = args.out
    if out is None:
        out = Path("data/sim/demo") if (args.scenario, args.path) == ("demo", "orbit") else Path(f"data/sim/{args.scenario}-{args.path}")

    with FireSim(scenario, make_path(args.path, scenario.grid)) as sim:
        n = save_run(sim, out, args.duration, args.interval, preview=not args.no_preview)
    print(f"Done. {n} frames written to {out}/")


if __name__ == "__main__":
    main()
