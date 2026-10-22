"""
tune_rainbow_mask.py — Can hot_mask stop marking "confetti" on rainbow frames?

Usage:
    python scripts/tune_rainbow_mask.py

On rainbow-palette frames hot_mask marks more than the hand masks do: it
takes scattered specks and cyan/blue pixels, while the labelling rule is
orange/yellow plus touching green. This script tries a few fixes without
fooling itself, using the 48 hand-labelled frames in data/flame2/handcheck/.

Protocol (fixed before any result was seen):
- Within each stratum, frames sorted by number alternate between a TUNING
  half and a HELD-OUT half (4 + 4).
- Only rainbow frames are changed; inferno settings stay as they are.
- Candidates: a higher min_contrast, a seed rule (keep a region only if
  some pixel in it reaches `seed` on the palette, i.e. yellow/orange), a
  local-density filter (keep a hot pixel only if at least `density` of its
  7x7 neighbourhood is hot), and their combinations.
- The candidate with the best mean IoU on the 16 rainbow TUNING frames is
  chosen. Only that one candidate is then scored on the held-out half.
- Adopt it only if, on the held-out rainbow frames, mean IoU improves by at
  least 0.03, no rainbow stratum drops by more than 0.02, and the share of
  rainbow fire frames left with an empty mask (every 50th frame) rises by
  no more than 1 percentage point. Otherwise keep the current settings.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))     # to reuse the labelme reader
from score_handcheck import hand_mask  # noqa: E402

from embereye.data import flame2 as f2  # noqa: E402
from embereye.masks import iou  # noqa: E402

HANDCHECK = f2.FLAME2_DIR / "handcheck"
GATE = 0.25
MIN_CONTRASTS = (0.35, 0.45, 0.55, 0.65)
SEEDS = (None, 0.75, 0.85)
DENSITIES = (None, 0.3, 0.5)
DENSITY_WINDOW_PX = 7
CURRENT = (0.35, None, None)
MIN_GAIN, MAX_STRATUM_DROP, MAX_EMPTY_RISE_PCT = 0.03, 0.02, 1.0


def rainbow_mask(thermal: np.ndarray, min_contrast: float, seed: Optional[float], density: Optional[float],
                 top_fraction: float = 0.10, min_area_px: int = 20) -> np.ndarray:
    """hot_mask for a rainbow frame (gate on), plus the optional density and seed rules."""
    position = f2.heat_position(thermal, f2.RAINBOW)
    if np.median(position) > GATE:
        return np.zeros(position.shape, dtype=bool)
    position = cv2.medianBlur((position * 255).astype(np.uint8), 3).astype(np.float32) / 255
    threshold = max(np.quantile(position, 1.0 - top_fraction), np.median(position) + min_contrast)
    hot = position >= threshold

    if density is not None:
        local = cv2.blur(hot.astype(np.float32), (DENSITY_WINDOW_PX, DENSITY_WINDOW_PX))
        hot &= local >= density

    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px
    if seed is not None:
        peak = np.zeros(n, dtype=np.float32)
        np.maximum.at(peak, labels.ravel(), position.ravel())
        keep &= peak >= seed
    return keep[labels]


def load_frames() -> list[dict]:
    rows = list(csv.DictReader(open(HANDCHECK / "manifest.csv")))
    frames = []
    for stratum in sorted({r["stratum"] for r in rows}):
        members = sorted((r for r in rows if r["stratum"] == stratum), key=lambda r: int(r["frame"]))
        for i, r in enumerate(members):
            thermal = cv2.imread(str(HANDCHECK / r["file"]))
            status, hand = hand_mask(json.loads((HANDCHECK / r["file"]).with_suffix(".json").read_text()),
                                     thermal.shape[:2])
            if hand is None:
                continue
            frames.append({"frame": int(r["frame"]), "stratum": stratum, "half": "tuning" if i % 2 == 0 else "held-out",
                           "thermal": thermal, "hand": hand, "palette": f2.detect_palette(thermal)})
    return frames


def mean_iou(frames: list[dict], config: tuple) -> float:
    return float(np.mean([iou(rainbow_mask(f["thermal"], *config), f["hand"]) for f in frames]))


def empty_share(config: tuple, sample: list[np.ndarray]) -> float:
    return 100.0 * float(np.mean([not rainbow_mask(t, *config).any() for t in sample]))


def main() -> None:
    frames = load_frames()
    rainbow = [f for f in frames if f["palette"] == f2.RAINBOW]

    # The current settings must reproduce hot_mask exactly, or the comparison means nothing.
    for f in rainbow:
        assert np.array_equal(rainbow_mask(f["thermal"], *CURRENT),
                              f2.hot_mask(f["thermal"], f2.RAINBOW, max_frame_median=GATE)), f["frame"]

    tuning = [f for f in rainbow if f["half"] == "tuning"]
    held_out = [f for f in rainbow if f["half"] == "held-out"]
    configs = list(itertools.product(MIN_CONTRASTS, SEEDS, DENSITIES))    # simplest first
    tuning_scores = {c: mean_iou(tuning, c) for c in configs}
    chosen = max(configs, key=lambda c: tuning_scores[c])      # ties go to the simpler (earlier) one
    best_tuning = tuning_scores[chosen]
    print(f"{len(configs)} candidates scored on {len(tuning)} rainbow tuning frames.")
    print(f"current {CURRENT}: {mean_iou(tuning, CURRENT):.3f}; chosen {chosen}: {best_tuning:.3f}\n")

    # Held-out: scored once, for the current settings and the chosen candidate only.
    print("HELD-OUT half (mean IoU vs hand masks):")
    print("| stratum | frames | current | chosen | change |\n|---|---|---|---|---|")
    drops = []
    for stratum in sorted({f["stratum"] for f in held_out}):
        members = [f for f in held_out if f["stratum"] == stratum]
        before, after = mean_iou(members, CURRENT), mean_iou(members, chosen)
        drops.append(before - after)
        print(f"| {stratum} | {len(members)} | {before:.3f} | {after:.3f} | {after - before:+.3f} |")
    before, after = mean_iou(held_out, CURRENT), mean_iou(held_out, chosen)
    print(f"| all rainbow | {len(held_out)} | {before:.3f} | {after:.3f} | {after - before:+.3f} |\n")

    labels = f2.parse_labels(f2.FLAME2_DIR / "labels.txt")
    sample = []
    for n in range(17501, max(labels) + 1, 50):
        if labels[n][0]:
            thermal = f2.load_frame(n)[1]
            if f2.detect_palette(thermal) == f2.RAINBOW:
                sample.append(thermal)
    empty_before, empty_after = empty_share(CURRENT, sample), empty_share(chosen, sample)
    print(f"rainbow fire frames with an empty mask (every 50th, n={len(sample)}): "
          f"current {empty_before:.1f}%, chosen {empty_after:.1f}%")

    adopt = (after - before >= MIN_GAIN and max(drops) <= MAX_STRATUM_DROP
             and empty_after - empty_before <= MAX_EMPTY_RISE_PCT)
    print(f"\nDecision: {'ADOPT ' + str(chosen) if adopt else 'KEEP the current settings'} "
          f"(needs held-out gain >= {MIN_GAIN}, no stratum down > {MAX_STRATUM_DROP}, "
          f"empty-mask rise <= {MAX_EMPTY_RISE_PCT} pt)")


if __name__ == "__main__":
    main()
