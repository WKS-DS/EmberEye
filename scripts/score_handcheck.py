"""
score_handcheck.py — Compare hot_mask's thermal masks with hand-drawn ones.

Usage:
    python scripts/score_handcheck.py [--top-fraction 0.10] [--min-contrast 0.35] [--gate 0.25]

Reads data/flame2/handcheck/manifest.csv (from scripts/export_handcheck.py)
and the labelme JSON saved next to each frame. For every labelled frame it
runs hot_mask with the frame gate on and reports, per stratum, the IoU of
your mask with:
- iou:       the hot_mask itself. Go/no-go for Phase 4.4 is a mean of at
             least 0.5 in every stratum.
- iou_label: the label YOLO would be trained on, i.e. the mask after
             mask_to_yolo_seg (holes filled, outlines simplified, small
             regions dropped). A gap between the two is lost in the export.
The hot_mask parameters are options so a stratum can be re-scored after tuning.

Writes scores.csv (one row per frame) and scores.md (the tables) to the
hand-check folder, and prints scores.md.

Labelling with labelme
----------------------
labelme is a desktop app; keep it out of venv.nosync (it brings Qt):
    brew install pipx && pipx ensurepath          # once, then open a new terminal
    pipx install --python /opt/homebrew/bin/python3.11 labelme
    labelme data/flame2/handcheck --labels fire --validatelabel exact \\
        --flags no_fire,unsure --nodata --autosave

- Draw a polygon (Ctrl+N, click round the region, click the first point to
  close it) around every region you judge to be fire, and label it "fire".
  Rectangles and circles also work; other shape types are rejected.
- No fire visible: draw nothing and tick the no_fire flag.
- Can't tell: tick unsure. The frame is left out of the scores.
- D / A move to the next / previous frame; --autosave writes
  frame_NNNNN.json next to the image as you go.

Rules the scores follow
-----------------------
- IoU of two empty masks is 1 (both say "no fire").
- A frame whose gated hot_mask is empty would be excluded from training, so
  it is listed but not counted in the mean IoU.
- A frame where your mask covers more than top_fraction of the image is
  flagged: hot_mask can never mark more than that, so its IoU is capped.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from embereye.data import flame2 as f2
from embereye.masks import iou, polygons_to_mask

HANDCHECK = f2.FLAME2_DIR / "handcheck"
PASS_IOU = 0.5
STRATA = [f"{variant} {flag}" for variant in (f2.INFERNO, f2.RAINBOW_RED, f2.RAINBOW_MAGENTA) for flag in ("YY", "YN")]


def shape_polygon(shape: dict) -> np.ndarray:
    """A labelme shape as an (N, 2) pixel polygon."""
    points = np.array(shape["points"], dtype=np.float64)
    kind = shape.get("shape_type") or "polygon"
    if kind == "polygon":
        return points
    if kind == "rectangle":
        (x1, y1), (x2, y2) = points
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if kind == "circle":
        (cx, cy), (ex, ey) = points
        radius = np.hypot(ex - cx, ey - cy)
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        return np.stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1)
    raise ValueError(f"shape type {kind!r} is not supported; redraw it as a polygon")


def hand_mask(annotation: dict, shape: tuple[int, int]) -> tuple[str, np.ndarray | None]:
    """(status, mask) for one labelme file. status is "ok", "unsure" or a
    problem description; mask is None unless status is "ok"."""
    flags = annotation.get("flags") or {}
    shapes = annotation.get("shapes") or []
    if (annotation.get("imageHeight"), annotation.get("imageWidth")) != shape:
        return "problem: JSON image size differs from the frame", None
    if flags.get("unsure"):
        return "unsure", None
    other = sorted({s["label"] for s in shapes if s["label"] != "fire"})
    if other:
        return f"problem: unexpected labels {other}", None
    if flags.get("no_fire") and shapes:
        return "problem: no_fire ticked but fire was drawn", None
    if not shapes and not flags.get("no_fire"):
        return "problem: nothing drawn and no_fire not ticked", None
    try:
        polygons = [shape_polygon(s) for s in shapes]
    except ValueError as err:
        return f"problem: {err}", None
    return "ok", polygons_to_mask(polygons, shape)


def training_label_mask(mask: np.ndarray) -> np.ndarray:
    """The mask YOLO will actually be trained on: the mask written out as
    YOLO-seg label lines, then filled back in."""
    height, width = mask.shape
    polygons = []
    for line in f2.mask_to_yolo_seg(mask):
        coords = np.array(line.split()[1:], dtype=np.float64).reshape(-1, 2)
        polygons.append(coords * [width, height])
    return polygons_to_mask(polygons, mask.shape)


def score_frame(row: dict, folder: Path, params: dict) -> dict:
    thermal = cv2.imread(str(folder / row["file"]))
    shape = thermal.shape[:2]
    out = {"frame": row["frame"], "stratum": row["stratum"], "iou": "", "iou_label": "",
           "hand_pct": "", "mask_pct": "", "hand_over_cap": ""}

    json_path = (folder / row["file"]).with_suffix(".json")
    if not json_path.exists():
        return {**out, "status": "not labelled"}
    status, hand = hand_mask(json.loads(json_path.read_text()), shape)
    if hand is None:
        return {**out, "status": status}

    palette = f2.detect_palette(thermal)
    if palette == f2.UNKNOWN:
        predicted = np.zeros(shape, dtype=bool)
    else:
        predicted = f2.hot_mask(thermal, palette, top_fraction=params["top_fraction"],
                                min_contrast=params["min_contrast"], max_frame_median=params["gate"])
    out.update({
        "iou": round(iou(predicted, hand), 3),
        "iou_label": round(iou(training_label_mask(predicted), hand), 3),
        "hand_pct": round(100 * hand.mean(), 1),
        "mask_pct": round(100 * predicted.mean(), 1),
        "hand_over_cap": "yes" if hand.mean() > params["top_fraction"] else "",
    })
    if not predicted.any():
        return {**out, "status": "excluded: empty gated mask"}
    return {**out, "status": "scored"}


def summary_row(name: str, rows: list[dict]) -> dict:
    scored = [r["iou"] for r in rows if r["status"] == "scored"]
    labelled = [r for r in rows if r["status"] != "not labelled"]
    problems = sum(r["status"].startswith("problem") for r in rows)
    mean = float(np.mean(scored)) if scored else None
    verdict = "-" if mean is None else ("pass" if mean >= PASS_IOU else f"below {PASS_IOU}")
    if mean is not None and (problems or len(labelled) < len(rows)):
        verdict += " (incomplete)"
    return {
        "stratum": name,
        "frames": len(rows),
        "labelled": len(labelled),
        "scored": len(scored),
        "mean_iou": round(mean, 3) if scored else "-",
        "median_iou": round(float(np.median(scored)), 3) if scored else "-",
        "min_iou": round(min(scored), 3) if scored else "-",
        "mean_iou_label": round(float(np.mean([r["iou_label"] for r in rows if r["status"] == "scored"])), 3) if scored else "-",
        "hand_over_cap": sum(r["hand_over_cap"] == "yes" for r in rows),
        "excluded": sum(r["status"].startswith("excluded") for r in rows),
        "unsure": sum(r["status"] == "unsure" for r in rows),
        "problems": problems,
        "verdict": verdict,
    }


def markdown_table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", type=Path, default=HANDCHECK, help="hand-check folder")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--min-contrast", type=float, default=0.35)
    parser.add_argument("--gate", type=float, default=0.25, help="max_frame_median for the frame gate")
    args = parser.parse_args()
    params = {"top_fraction": args.top_fraction, "min_contrast": args.min_contrast, "gate": args.gate}

    manifest = args.dir / "manifest.csv"
    if not manifest.exists():
        sys.exit(f"No manifest at {manifest}. Run scripts/export_handcheck.py first.")
    rows = [score_frame(r, args.dir, params) for r in csv.DictReader(open(manifest))]

    with open(args.dir / "scores.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = [summary_row(s, [r for r in rows if r["stratum"] == s]) for s in STRATA]
    summary.append(summary_row("all", rows))
    per_frame = [{k: r[k] for k in ("frame", "stratum", "status", "iou", "iou_label", "hand_pct", "mask_pct", "hand_over_cap")}
                 for r in rows if r["status"] != "not labelled"]
    report = [
        "# Hand-check scores", "",
        f"hot_mask with top_fraction={args.top_fraction}, min_contrast={args.min_contrast}, gate={args.gate}. "
        f"Go/no-go for 4.4: mean IoU >= {PASS_IOU} in every stratum.", "",
        markdown_table(summary), "",
        "`iou` = hot_mask vs your mask (the verdict uses this); `iou_label` = the exported YOLO-seg label vs your mask. "
        "`scored` frames count towards the IoU. `excluded` = gated mask empty (would be left out of training). "
        f"`hand_over_cap` = your mask covers more than {100 * args.top_fraction:.0f}% of the image, "
        "more than hot_mask can ever mark.", "",
        "## Per frame", "",
        markdown_table(per_frame) if per_frame else "No frames labelled yet.", "",
    ]
    (args.dir / "scores.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"Written to {args.dir}/scores.csv and scores.md")


if __name__ == "__main__":
    main()
