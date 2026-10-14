"""
audit_flame2_labels.py — Evidence for reviewing the FLAME 2 labelling method.

Usage:
    python scripts/audit_flame2_labels.py

Writes everything to data/flame2/audit/ (not in git) and prints a summary:

1. palette_scan.csv   palette features for every 200th fire frame, so the
                      palette switch points are visible
2. coverage.csv       per labels.txt block: palette mix, and for the old
                      (grayscale > 200) and new (palette-aware) methods the
                      share of frames with a mask and the median mask size
3. montage_*.png      RGB | thermal | new mask on thermal | new mask on RGB,
                      for smoky and clear frames of each palette, plus the
                      no-fire frames where the new method wrongly finds fire
4. alignment.csv / alignment.png
                      do thermal hot regions line up with flame visible in RGB?
5. summary.md         all of the above as tables

It samples every 50th frame, so it takes a few minutes. It only reads the
dataset; it writes no labels and no splits.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from embereye.data import flame2 as f2

SAMPLE_EVERY = 50
GATE = 0.25                 # max_frame_median value tried as the optional frame-level gate
FLAME_MIN_PX = 15           # a frame "clearly shows flame" with at least this many flame pixels
OUT = f2.FLAME2_DIR / "audit"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def label_blocks(labels_path: Path) -> list[tuple[int, int, str]]:
    """The (first, last, flags) ranges exactly as listed in labels.txt."""
    blocks = []
    for line in labels_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
            blocks.append((int(parts[0]), int(parts[1]), parts[2]))
    return blocks


def outline(image: np.ndarray, mask: np.ndarray, colour=(255, 255, 0)) -> np.ndarray:
    """Copy of `image` with the mask's outlines drawn on it."""
    out = image.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, colour, 1)
    return out


def montage_row(frame_num: int, note: str = "") -> np.ndarray:
    rgb, thermal = f2.load_frame(frame_num)
    palette = f2.detect_palette(thermal)
    mask = f2.hot_mask(thermal, palette) if palette != f2.UNKNOWN else np.zeros(thermal.shape[:2], bool)
    row = np.hstack([rgb, thermal, outline(thermal, mask), outline(rgb, mask)])
    text = f"{frame_num} {palette} mask {100 * mask.mean():.1f}% {note}"
    cv2.putText(row, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(row, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return row


def flame_pixels(rgb_bgr: np.ndarray) -> np.ndarray:
    """Pixels that look like visible flame in the RGB frame: bright, strongly
    saturated red-to-orange. Smoke (grey), trees (green) and soil (dull
    brown) fail the saturation or brightness tests."""
    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    hue_deg = hsv[..., 0].astype(int) * 2
    return ((hue_deg <= 40) | (hue_deg >= 350)) & (hsv[..., 1] > 150) & (hsv[..., 2] > 180)


def masks_for(thermal: np.ndarray) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, float]:
    """(palette, new mask, new mask with the frame gate, old mask, median heat) for one thermal frame."""
    palette = f2.detect_palette(thermal)
    empty = np.zeros(thermal.shape[:2], dtype=bool)
    if palette == f2.UNKNOWN:
        return palette, empty, empty, f2.old_hot_mask(thermal), float("nan")
    new = f2.hot_mask(thermal, palette)
    median_heat = float(np.median(f2.heat_position(thermal, palette)))
    gated = new if median_heat <= GATE else empty
    return palette, new, gated, f2.old_hot_mask(thermal), median_heat


# ---------------------------------------------------------------------------
# 1. Palette scan
# ---------------------------------------------------------------------------
def palette_scan(labels) -> list[dict]:
    rows = []
    for n in range(13701, max(labels) + 1, 200):
        _, thermal = f2.load_frame(n)
        rows.append({"frame": n, "palette": f2.detect_palette(thermal), **f2.palette_features(thermal)})
    with open(OUT / "palette_scan.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# 2. Coverage per block
# ---------------------------------------------------------------------------
def coverage(blocks) -> tuple[list[dict], dict]:
    rows, per_frame = [], {}
    for first, last, flags in blocks:
        palettes = Counter()
        old_any, new_any, gated_any, old_area, new_area = 0, 0, 0, [], []
        frames = list(range(first, last + 1, SAMPLE_EVERY))
        for n in frames:
            _, thermal = f2.load_frame(n)
            palette, new, gated, old, median_heat = masks_for(thermal)
            palettes[palette] += 1
            per_frame[n] = (palette, new.mean(), old.mean(), gated.mean(), median_heat)
            if old.any():
                old_any += 1
                old_area.append(100 * old.mean())
            if new.any():
                new_any += 1
                new_area.append(100 * new.mean())
            if gated.any():
                gated_any += 1
        rows.append({
            "block": f"{first}-{last}",
            "flags": flags,
            "sampled": len(frames),
            "palettes": " ".join(f"{k}:{v}" for k, v in sorted(palettes.items())),
            "old_frames_with_mask_pct": round(100 * old_any / len(frames), 1),
            "new_frames_with_mask_pct": round(100 * new_any / len(frames), 1),
            "gated_frames_with_mask_pct": round(100 * gated_any / len(frames), 1),
            "old_median_area_pct": round(float(np.median(old_area)), 2) if old_area else 0.0,
            "new_median_area_pct": round(float(np.median(new_area)), 2) if new_area else 0.0,
        })
    with open(OUT / "coverage.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, per_frame


# ---------------------------------------------------------------------------
# 3. Montages
# ---------------------------------------------------------------------------
def pick_spread(candidates: list[int], k: int) -> list[int]:
    """k frames spread evenly through a sorted list."""
    if len(candidates) <= k:
        return candidates
    return [candidates[int(i)] for i in np.linspace(0, len(candidates) - 1, k)]


def montages(labels, per_frame) -> list[str]:
    written = []
    groups = {
        "inferno_fire": lambda n: labels[n][0] and per_frame[n][0] == f2.INFERNO,
        "rainbow_fire": lambda n: labels[n][0] and per_frame[n][0] == f2.RAINBOW,
    }
    for name, belongs in groups.items():
        smoky = pick_spread(sorted(n for n in per_frame if belongs(n) and labels[n][1]), 3)
        clear = pick_spread(sorted(n for n in per_frame if belongs(n) and not labels[n][1]), 3)
        rows = [montage_row(n, "smoky") for n in smoky] + [montage_row(n, "clear") for n in clear]
        if rows:
            path = OUT / f"montage_{name}.png"
            cv2.imwrite(str(path), np.vstack(rows))
            written.append(path.name)

    # No-fire frames where the new method finds something (false positives).
    false_pos = sorted(n for n in per_frame if not labels[n][0] and per_frame[n][1] > 0)
    rows = [montage_row(n, "NO-FIRE") for n in pick_spread(false_pos, 6)]
    if rows:
        path = OUT / "montage_nofire_false_positives.png"
        cv2.imwrite(str(path), np.vstack(rows))
        written.append(path.name)

    # The unknown-palette frames, if any, so they can be looked at.
    unknown = sorted(n for n in per_frame if per_frame[n][0] == f2.UNKNOWN)
    rows = [montage_row(n, "unknown palette") for n in pick_spread(unknown, 6)]
    if rows:
        path = OUT / "montage_unknown_palette.png"
        cv2.imwrite(str(path), np.vstack(rows))
        written.append(path.name)
    return written


# ---------------------------------------------------------------------------
# 4. RGB / thermal alignment
# ---------------------------------------------------------------------------
def heat_at(heat: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Heat position at pixel coordinates; 0 where they fall off the image."""
    h, w = heat.shape
    xi, yi = np.rint(xs).astype(int), np.rint(ys).astype(int)
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    out = np.zeros(len(xs))
    out[inside] = heat[yi[inside], xi[inside]]
    return out


def best_transform(heat: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float, int, int]:
    """Scale (about the image centre) and shift that put the RGB flame pixels
    on the hottest thermal pixels. Returns (mean heat, scale, dx, dy)."""
    centre = (heat.shape[1] - 1) / 2
    best = (-1.0, 1.0, 0, 0)
    for scale in np.arange(0.80, 1.251, 0.05):
        for dx in range(-40, 41, 2):
            for dy in range(-40, 41, 2):
                score = heat_at(heat, centre + scale * (xs - centre) + dx, centre + scale * (ys - centre) + dy).mean()
                if score > best[0]:
                    best = (float(score), float(scale), dx, dy)
    return best


def alignment(labels, n_frames: int = 10, min_gap: int = 200) -> list[dict]:
    """Do thermal hot regions sit where RGB shows flame?

    For frames with clearly visible flame, compare how often the flame
    pixels land on a thermal hot region as-is against how often they would
    if they were shifted to a random place (the chance level)."""
    candidates = []
    for n in range(13701, max(labels) + 1, SAMPLE_EVERY // 2):
        rgb, _ = f2.load_frame(n)
        candidates.append((int(flame_pixels(rgb).sum()), n))
    chosen = []
    for count, n in sorted(candidates, reverse=True):
        if count < FLAME_MIN_PX:
            break
        if all(abs(n - m) >= min_gap for m in chosen):
            chosen.append(n)
        if len(chosen) == n_frames:
            break

    rng = np.random.default_rng(0)
    rows, tiles = [], []
    for n in sorted(chosen):
        rgb, thermal = f2.load_frame(n)
        palette = f2.detect_palette(thermal)
        heat = f2.heat_position(thermal, palette)
        hot = f2.hot_mask(thermal, palette)
        near_hot = (cv2.dilate(hot.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0).astype(float)
        ys, xs = np.nonzero(flame_pixels(rgb))

        hit_as_is = heat_at(near_hot, xs, ys).mean()
        # Chance level: the same flame shape moved to 200 random places.
        shifts = rng.integers(-60, 61, size=(200, 2))
        hit_random = np.mean([heat_at(near_hot, xs + dx, ys + dy).mean() for dx, dy in shifts])
        heat_as_is = heat_at(heat, xs, ys).mean()
        heat_random = np.mean([heat_at(heat, xs + dx, ys + dy).mean() for dx, dy in shifts])
        best_heat, scale, dx, dy = best_transform(heat, xs, ys)

        rows.append({
            "frame": n, "palette": palette, "flame_px": len(xs),
            "flame_on_hot_pct": round(100 * hit_as_is, 1), "chance_pct": round(100 * hit_random, 1),
            "heat_under_flame": round(float(heat_as_is), 2), "heat_chance": round(float(heat_random), 2),
            "best_scale": round(scale, 2), "best_dx": dx, "best_dy": dy, "heat_best": round(best_heat, 2),
        })

        marked = rgb.copy()
        marked[ys, xs] = (255, 0, 255)
        tile = np.hstack([outline(marked, hot), outline(thermal, hot), thermal])
        cv2.putText(tile, f"{n}: RGB flame (magenta) vs thermal hot outline (cyan), as-is", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    if tiles:
        cv2.imwrite(str(OUT / "alignment.png"), np.vstack(tiles))
        with open(OUT / "alignment.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# 5. False positives on no-fire frames
# ---------------------------------------------------------------------------
def false_positive_rate(labels, per_frame) -> dict:
    no_fire = [n for n in per_frame if not labels[n][0]]
    by_palette = Counter(per_frame[n][0] for n in no_fire)
    new_fp = [n for n in no_fire if per_frame[n][1] > 0]
    old_fp = [n for n in no_fire if per_frame[n][2] > 0]
    gated_fp = [n for n in no_fire if per_frame[n][3] > 0]
    return {
        "no_fire_frames_sampled": len(no_fire),
        "palettes": dict(by_palette),
        "old_fp_pct": round(100 * len(old_fp) / len(no_fire), 1),
        "new_fp_pct": round(100 * len(new_fp) / len(no_fire), 1),
        "gated_fp_pct": round(100 * len(gated_fp) / len(no_fire), 1),
        "new_fp_median_area_pct": round(float(np.median([100 * per_frame[n][1] for n in new_fp])), 2) if new_fp else 0.0,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def markdown_table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    if not (f2.FLAME2_DIR / "labels.txt").exists():
        sys.exit(f"FLAME 2 not found at {f2.FLAME2_DIR}")
    OUT.mkdir(parents=True, exist_ok=True)
    labels = f2.parse_labels(f2.FLAME2_DIR / "labels.txt")
    blocks = label_blocks(f2.FLAME2_DIR / "labels.txt")

    print("1/5 palette scan (every 200th fire frame)")
    scan = palette_scan(labels)
    switches = [(a["frame"], b["frame"], a["palette"], b["palette"]) for a, b in zip(scan, scan[1:]) if a["palette"] != b["palette"]]

    print(f"2/5 coverage (every {SAMPLE_EVERY}th frame)")
    cov, per_frame = coverage(blocks)

    print("3/5 montages")
    pictures = montages(labels, per_frame)

    print("4/5 alignment check")
    align = alignment(labels)

    print("5/5 false positives")
    fp = false_positive_rate(labels, per_frame)

    fire = [n for n in per_frame if labels[n][0]]
    totals = {
        "fire_frames_sampled": len(fire),
        "old_with_mask_pct": round(100 * np.mean([per_frame[n][2] > 0 for n in fire]), 1),
        "new_with_mask_pct": round(100 * np.mean([per_frame[n][1] > 0 for n in fire]), 1),
        "gated_with_mask_pct": round(100 * np.mean([per_frame[n][3] > 0 for n in fire]), 1),
        "palettes": dict(Counter(per_frame[n][0] for n in fire)),
    }
    heat_rows = []
    for name, members in [
        ("no-fire (1-13700)", [n for n in per_frame if not labels[n][0]]),
        ("fire, inferno palette", [n for n in fire if per_frame[n][0] == f2.INFERNO]),
        ("fire, rainbow palette", [n for n in fire if per_frame[n][0] == f2.RAINBOW]),
    ]:
        medians = np.array([per_frame[n][4] for n in members])
        p10, p50, p90 = np.percentile(medians, [10, 50, 90])
        heat_rows.append({"frames": name, "count": len(members),
                          "median_heat_p10": round(p10, 2), "median_heat_p50": round(p50, 2), "median_heat_p90": round(p90, 2),
                          f"share_above_gate_{GATE}": f"{100 * (medians > GATE).mean():.1f}%"})

    report = [
        "# FLAME 2 label audit", "",
        f"Sampled every {SAMPLE_EVERY}th frame. New method: `hot_mask` with default parameters.", "",
        "## Palette switches (every 200th fire frame)", "",
        *(f"- between frames {a} and {b}: {pa} -> {pb}" for a, b, pa, pb in switches), "",
        "## All fire frames", "", markdown_table([totals]), "",
        f"`gated` = new method plus the optional frame gate (`max_frame_median={GATE}`).", "",
        "## Frame median heat position (why the gate works)", "", markdown_table(heat_rows), "",
        "## Coverage per labels.txt block", "", markdown_table(cov), "",
        "## False positives on no-fire frames (1-13700)", "", markdown_table([fp]), "",
        "## RGB / thermal alignment", "", markdown_table(align) if align else "No frames with clear flame found.", "",
        "## Pictures", "", *(f"- {p}" for p in pictures + (["alignment.png"] if align else [])), "",
    ]
    (OUT / "summary.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"\nWritten to {OUT}/")


if __name__ == "__main__":
    main()
