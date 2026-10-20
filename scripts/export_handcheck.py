"""
export_handcheck.py — Pick FLAME 2 thermal frames for hand-checking hot_mask.

Usage:
    python scripts/export_handcheck.py [--per-stratum 8]

Copies the chosen thermal frames to data/flame2/handcheck/ (not in git) as
frame_NNNNN.jpg, byte for byte, plus manifest.csv saying which stratum each
frame belongs to. Hand-draw the fire in each frame with labelme, then score
hot_mask against your masks with scripts/score_handcheck.py (its docstring
has the labelme steps).

Strata: palette variant (inferno, rainbow-red, rainbow-magenta; see
flame2.palette_variant) x the labels.txt smoke flag (YY = smoke fills at
least half the RGB frame, YN = it doesn't). Six strata, all fire frames.

Frames are picked without randomness, from every 10th fire frame:
1. one frame from every fire block in labels.txt, so every block is covered
   (blocks with the fewest strata to choose from go first)
2. more frames from 45000 onwards until there are at least MIN_LATE
3. each stratum filled up to --per-stratum, every time taking the frame
   farthest (in frame number) from all frames already picked, so they
   spread through the timeline
At every step, frames showing the same view as one already picked are
skipped. The drone hovers for long stretches, so frames thousands apart
can be near-copies; labelling those twice would waste effort.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from embereye.data import flame2 as f2

OUT = f2.FLAME2_DIR / "handcheck"
CANDIDATE_EVERY = 10
LATE_FROM = 45000
MIN_LATE = 8
# Two frames show the same view when their 32x32 grayscale thumbnails
# correlate above this. Measured: hovering near-copies 0.63-0.99,
# different views 0.18 or less.
SAME_VIEW_CORR = 0.5
FLAGS = ("YY", "YN")
VARIANTS = (f2.INFERNO, f2.RAINBOW_RED, f2.RAINBOW_MAGENTA)
STRATA = [f"{variant} {flag}" for variant in VARIANTS for flag in FLAGS]


def label_blocks(labels_path: Path) -> list[tuple[int, int, str]]:
    """The (first, last, flags) ranges exactly as listed in labels.txt."""
    blocks = []
    for line in labels_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            blocks.append((int(parts[0]), int(parts[1]), parts[2]))
    return blocks


def thumbnail(thermal_bgr: np.ndarray) -> np.ndarray:
    """32x32 grayscale thumbnail, flattened and scaled so that the dot
    product of two thumbnails is their correlation."""
    small = cv2.resize(cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY), (32, 32), interpolation=cv2.INTER_AREA)
    v = small.astype(np.float64).ravel()
    v -= v.mean()
    return v / max(np.linalg.norm(v), 1e-9)


def candidates(blocks) -> list[dict]:
    """Every 10th fire frame with its block, flag, palette variant and
    thumbnail. Frames listed in two blocks (41642) are skipped."""
    in_blocks: dict[int, list[tuple[int, int, str]]] = {}
    for block in blocks:
        for n in range(block[0], block[1] + 1):
            in_blocks.setdefault(n, []).append(block)

    rows = []
    fire_frames = sorted(n for n, bs in in_blocks.items() if len(bs) == 1 and bs[0][2][0] == "Y")
    for n in fire_frames[::CANDIDATE_EVERY]:
        first, last, flag = in_blocks[n][0]
        _, thermal = f2.load_frame(n)
        variant = f2.palette_variant(thermal)
        if variant == f2.UNKNOWN:
            continue
        rows.append({"frame": n, "block": f"{first}-{last}", "flag": flag, "variant": variant,
                     "stratum": f"{variant} {flag}", "thumb": thumbnail(thermal)})
    return rows


def new_views(options: list[dict], chosen: list[dict]) -> list[dict]:
    """The options that don't show the same view as any chosen frame
    (all options, if every one of them does)."""
    fresh = [c for c in options if all(float(c["thumb"] @ p["thumb"]) <= SAME_VIEW_CORR for p in chosen)]
    return fresh or options


def middle(options: list[dict], chosen: list[dict]) -> dict:
    """The middle (by frame number) of the options showing a new view."""
    options = new_views(options, chosen)
    return options[len(options) // 2]


def farthest(options: list[dict], chosen: list[dict]) -> dict:
    """Of the options showing a new view, the one whose frame number is
    farthest from every chosen frame."""
    options = new_views(options, chosen)
    if not chosen:
        return options[len(options) // 2]
    return max(options, key=lambda c: (min(abs(c["frame"] - p["frame"]) for p in chosen), -c["frame"]))


def pick(pool: list[dict], per_stratum: int) -> list[dict]:
    chosen: list[dict] = []

    def room(stratum: str) -> int:
        return per_stratum - sum(c["stratum"] == stratum for c in chosen)

    def unpicked(**match) -> list[dict]:
        return [c for c in pool if c not in chosen and all(c[k] == v for k, v in match.items())]

    # 1. Every block covered.
    blocks = sorted({c["block"] for c in pool}, key=lambda b: (len({c["stratum"] for c in pool if c["block"] == b}),
                                                               int(b.split("-")[0])))
    for block in blocks:
        strata = sorted({c["stratum"] for c in unpicked(block=block) if room(c["stratum"]) > 0},
                        key=lambda s: (per_stratum - room(s), STRATA.index(s)))
        if not strata:
            warnings.warn(f"Block {block} not covered: its strata are already full")
            continue
        chosen.append({**middle(unpicked(block=block, stratum=strata[0]), chosen), "picked_for": "block"})

    # 2. Enough late frames.
    while sum(c["frame"] >= LATE_FROM for c in chosen) < MIN_LATE:
        strata = [s for s in STRATA if room(s) > 0 and any(c["frame"] >= LATE_FROM for c in unpicked(stratum=s))]
        if not strata:
            warnings.warn(f"Could not reach {MIN_LATE} frames from {LATE_FROM} on")
            break
        stratum = max(strata, key=lambda s: (room(s), -STRATA.index(s)))
        options = [c for c in unpicked(stratum=stratum) if c["frame"] >= LATE_FROM]
        chosen.append({**farthest(options, chosen), "picked_for": "late"})

    # 3. Fill every stratum, spread out.
    for stratum in STRATA:
        while room(stratum) > 0:
            options = unpicked(stratum=stratum)
            if not options:
                warnings.warn(f"Stratum {stratum} has only {per_stratum - room(stratum)} frames")
                break
            chosen.append({**farthest(options, chosen), "picked_for": "spread"})

    return sorted(chosen, key=lambda c: c["frame"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-stratum", type=int, default=8)
    args = parser.parse_args()

    labels_path = f2.FLAME2_DIR / "labels.txt"
    if not labels_path.exists():
        sys.exit(f"FLAME 2 not found at {f2.FLAME2_DIR}")
    if OUT.exists() and any(OUT.glob("*.json")):
        sys.exit(f"{OUT} already has labelme JSON files in it. Move them away first; this would replace the frames.")

    print(f"Classifying every {CANDIDATE_EVERY}th fire frame...")
    pool = candidates(label_blocks(labels_path))
    chosen = pick(pool, args.per_stratum)

    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("frame_*.jpg"):
        old.unlink()
    for c in chosen:
        c["file"] = f"frame_{c['frame']:05d}.jpg"
        shutil.copyfile(f2.frame_paths(c["frame"])[1], OUT / c["file"])

    fields = ["file", "frame", "stratum", "variant", "flag", "block", "picked_for"]
    with open(OUT / "manifest.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chosen)

    print(f"\n{len(chosen)} frames written to {OUT}/\n")
    for stratum in STRATA:
        frames = [c["frame"] for c in chosen if c["stratum"] == stratum]
        print(f"  {stratum:<20} {len(frames):>2}  {frames}")
    all_blocks = {c["block"] for c in pool}
    covered = {c["block"] for c in chosen}
    print(f"\n  blocks covered: {len(covered)} of {len(all_blocks)}"
          f"   frames from {LATE_FROM} on: {sum(c['frame'] >= LATE_FROM for c in chosen)}")


if __name__ == "__main__":
    main()
