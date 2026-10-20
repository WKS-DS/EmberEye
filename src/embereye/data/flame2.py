"""
flame2.py — Reading and labelling the FLAME 2 dataset.

FLAME 2 is 53,451 pairs of 254x254 frames (RGB + thermal) from a drone
flying over a prescribed burn, with frame-level labels only. To train a
segmentation model we need fire MASKS, so they are generated from the
thermal frames. The notebooks call this module instead of carrying their
own copies of the logic, so it can be tested.

Why this is not a simple threshold
----------------------------------
The thermal frames are false-colour JPEGs, not temperatures:
- The colour palette changes partway through the dataset. Up to about
  frame 17,370 it is inferno-like (black -> purple -> red -> orange ->
  yellow). From about frame 17,380 it is a "rainbow" palette whose cold
  end is dark red and whose hue runs backwards round the colour wheel as
  heat rises: pink -> magenta -> violet -> blue -> cyan -> green ->
  yellow -> orange.
- The colours appear to be rescaled every frame (auto-ranging): the
  hottest thing in each frame is near the top of the palette whatever its
  real temperature.

So `hot_mask` first converts each pixel to a "heat position" between 0
(cold end of the palette) and 1 (hot end) — a different conversion per
palette — and then keeps pixels that are hot RELATIVE TO THEIR OWN FRAME.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from embereye.masks import mask_to_polygons

# src/embereye/data/flame2.py -> parents[3] is the project folder.
FLAME2_DIR = Path(__file__).resolve().parents[3] / "data" / "flame2"
IMAGE_SIZE = 254    # every FLAME 2 frame is 254 x 254 pixels

INFERNO = "inferno"
RAINBOW = "rainbow"
UNKNOWN = "unknown"

# Rainbow frames split by background colour (see palette_variant).
RAINBOW_RED = "rainbow-red"
RAINBOW_MAGENTA = "rainbow-magenta"


# ---------------------------------------------------------------------------
# Labels and files
# ---------------------------------------------------------------------------
def parse_labels(path: Path) -> dict[int, tuple[bool, bool]]:
    """Read labels.txt into {frame number: (fire, smoke)}.

    The file lists ranges like `13701  14699  YY`. What the flags mean,
    according to the file's own header:
    - fire:  fire is visible in the RGB and/or the thermal frame. So a fire
             frame may show fire in thermal only.
    - smoke: smoke fills AT LEAST 50% of the RGB frame (a visual estimate).
             It does not mean "smoke present", so it can't be used as a
             smoke label.

    Some frames appear in two ranges (frame 41642 does). That is reported
    as a warning; the later range wins, as in the original notebook.
    """
    labels: dict[int, tuple[bool, bool]] = {}
    seen_twice: list[int] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        # Data lines are exactly: first frame, last frame, two Y/N letters.
        if len(parts) != 3 or not (parts[0].isdigit() and parts[1].isdigit()):
            continue
        first, last, flags = int(parts[0]), int(parts[1]), parts[2]
        fire, smoke = flags[0] == "Y", flags[1] == "Y"
        for frame in range(first, last + 1):
            if frame in labels:
                seen_twice.append(frame)
            labels[frame] = (fire, smoke)

    if seen_twice:
        warnings.warn(f"{len(seen_twice)} frame(s) appear in more than one range of {path}: {seen_twice[:10]}")
    return labels


def frame_paths(frame_num: int, root: Path = FLAME2_DIR) -> tuple[Path, Path]:
    """(rgb_path, thermal_path) for a frame number, e.g. 17701."""
    frames = Path(root) / "frames"
    rgb = frames / "254p RGB Images" / f"254p RGB Frame ({frame_num}).jpg"
    thermal = frames / "254p Thermal Images" / f"254p Thermal Frame ({frame_num}).jpg"
    return rgb, thermal


def load_frame(frame_num: int, root: Path = FLAME2_DIR) -> tuple[np.ndarray, np.ndarray]:
    """(rgb_bgr, thermal_bgr) images for a frame, as OpenCV loads them (blue-green-red order)."""
    rgb_path, thermal_path = frame_paths(frame_num, root)
    rgb, thermal = cv2.imread(str(rgb_path)), cv2.imread(str(thermal_path))
    if rgb is None or thermal is None:
        raise FileNotFoundError(f"Could not read frame {frame_num} under {root}")
    return rgb, thermal


# ---------------------------------------------------------------------------
# Heat position: where on its palette each pixel's colour sits
# ---------------------------------------------------------------------------
_INFERNO_LOOKUP: Optional[tuple[np.ndarray, np.ndarray]] = None
_LOOKUP_STEP = 4    # colours are rounded to steps of 4 (64 levels per channel) for the lookup table


def _inferno_lookup() -> tuple[np.ndarray, np.ndarray]:
    """For every (rounded) colour: its position along matplotlib's inferno
    colour map, and how far it is from the nearest inferno colour.

    Comparing every pixel against 256 palette colours is slow, so the
    answer is worked out ONCE for every possible rounded colour (64^3 of
    them) and kept in a module-level variable. Later calls just look it up.
    """
    global _INFERNO_LOOKUP
    if _INFERNO_LOOKUP is None:
        import matplotlib     # imported here so loading this module stays quick

        palette_rgb = matplotlib.colormaps["inferno"](np.linspace(0, 1, 256))[:, :3] * 255
        palette_bgr = palette_rgb[:, ::-1].astype(np.float32)

        levels = np.arange(0, 256, _LOOKUP_STEP, dtype=np.float32) + _LOOKUP_STEP / 2
        b, g, r = np.meshgrid(levels, levels, levels, indexing="ij")
        colours = np.stack([b, g, r], axis=-1).reshape(-1, 3)

        position = np.empty(len(colours), dtype=np.float32)
        distance = np.empty(len(colours), dtype=np.float32)
        for start in range(0, len(colours), 16384):          # in chunks to keep memory small
            chunk = colours[start:start + 16384]
            d2 = ((chunk[:, None, :] - palette_bgr[None, :, :]) ** 2).sum(axis=-1)
            nearest = d2.argmin(axis=1)
            position[start:start + 16384] = nearest / 255.0
            distance[start:start + 16384] = np.sqrt(d2[np.arange(len(chunk)), nearest])
        n = len(levels)
        _INFERNO_LOOKUP = (position.reshape(n, n, n), distance.reshape(n, n, n))
    return _INFERNO_LOOKUP


def inferno_position(thermal_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Heat position (0..1) of each pixel along the inferno palette, and its
    colour distance from that palette (small = the pixel really is an inferno colour)."""
    position, distance = _inferno_lookup()
    idx = (thermal_bgr // _LOOKUP_STEP).astype(np.intp)
    b, g, r = idx[..., 0], idx[..., 1], idx[..., 2]
    return position[b, g, r], distance[b, g, r]


def rainbow_position(thermal_bgr: np.ndarray) -> np.ndarray:
    """Heat position (0..1) of each pixel along the rainbow palette.

    The palette's hue runs BACKWARDS round the colour wheel as heat rises,
    starting just below pure red: dark red (cold) -> pink -> magenta ->
    violet -> blue -> cyan -> green -> yellow -> orange (hot). So position
    is simply (360 - hue) / 360, with hue in degrees.

    The cold dark red and the hot red-orange both sit near hue 0 but on
    opposite sides of it: the cold reds lean slightly blue (hue just under
    360, position near 0), the hot ones lean green (hue just over 0,
    position near 1). That is why the wrap is placed exactly at pure red.
    """
    hsv = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2HSV)
    hue_deg = hsv[..., 0].astype(np.float32) * 2          # OpenCV stores hue as 0..179 (degrees / 2)
    position = ((360.0 - hue_deg) % 360.0) / 360.0
    # Grey pixels have no meaningful hue; treat them as cold.
    position[hsv[..., 1] < 40] = 0.0
    return position


# ---------------------------------------------------------------------------
# Which palette is this frame?
# ---------------------------------------------------------------------------
def palette_features(thermal_bgr: np.ndarray) -> dict[str, float]:
    """The colour statistics detect_palette() decides from.

    - inferno_fit: median distance of the frame's colours from the inferno
      palette. Inferno frames sit almost on it; rainbow frames don't.
    - dark: share of near-black pixels (V < 60) — the inferno cold end.
    - dark_red: share of dark, saturated, slightly-blue reds — the rainbow cold end.
    - median_bgr_*: the frame's median colour, for the audit table.
    """
    _, distance = inferno_position(thermal_bgr)
    hsv = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2HSV)
    hue_deg = hsv[..., 0].astype(np.float32) * 2
    s, v = hsv[..., 1], hsv[..., 2]
    median = np.median(thermal_bgr.reshape(-1, 3), axis=0)
    return {
        "inferno_fit": float(np.median(distance)),
        "dark": float((v < 60).mean()),
        "dark_red": float((((hue_deg >= 340) | (hue_deg <= 4)) & (v < 160) & (s > 150)).mean()),
        "median_b": float(median[0]),
        "median_g": float(median[1]),
        "median_r": float(median[2]),
    }


def detect_palette(thermal_bgr: np.ndarray, inferno_max_fit: float = 45.0) -> str:
    """Classify a thermal frame's palette: "inferno", "rainbow" or "unknown".

    A frame is inferno when its colours lie close to the inferno palette
    (median distance under `inferno_max_fit`, in 0..255 colour units).
    FLAME 2's inferno-style palette is a little more saturated than
    matplotlib's in the purples and magentas, so pre-burn frames score up
    to about 31; rainbow frames score 60 or more. 45 sits in the gap.
    Otherwise it is rainbow when its colours look like the rainbow
    palette's: hardly any green in the typical pixel (the cold half of the
    rainbow palette is reds, pinks and purples) and red or blue clearly
    present. Anything else is "unknown", so it can be inspected rather
    than labelled wrongly.
    """
    features = palette_features(thermal_bgr)
    if features["inferno_fit"] < inferno_max_fit:
        return INFERNO
    if features["median_g"] < 20 and max(features["median_r"], features["median_b"]) > 60:
        return RAINBOW
    return UNKNOWN


def palette_variant(thermal_bgr: np.ndarray, min_dark_red: float = 0.4) -> str:
    """detect_palette(), with rainbow frames split by background colour:
    "inferno", "rainbow-red", "rainbow-magenta" or "unknown".

    It is the same rainbow palette either way. What differs is how much of
    the frame sits at its dark-red cold end: in some frames most of the
    ground is dark red, in others it is pink, magenta and violet (common
    from about frame 41,000). hot_mask's per-frame threshold depends on
    the frame's median, so the two are kept apart when checking masks.

    A rainbow frame is "rainbow-red" when at least `min_dark_red` of its
    pixels are the dark-red cold end (the `dark_red` feature). On every
    25th fire frame, 0.4 put 99% of frames whose cold half is pure dark
    red into "rainbow-red" and all frames whose cold half is clearly
    magenta into "rainbow-magenta"; frames in between are a mix of both.
    """
    palette = detect_palette(thermal_bgr)
    if palette != RAINBOW:
        return palette
    return RAINBOW_RED if palette_features(thermal_bgr)["dark_red"] >= min_dark_red else RAINBOW_MAGENTA


# ---------------------------------------------------------------------------
# Hot-region mask
# ---------------------------------------------------------------------------
def heat_position(thermal_bgr: np.ndarray, palette: str) -> np.ndarray:
    """Per-pixel heat position (0 = palette's cold end, 1 = hot end)."""
    if palette == INFERNO:
        position, _ = inferno_position(thermal_bgr)
        return position
    if palette == RAINBOW:
        return rainbow_position(thermal_bgr)
    raise ValueError(f"No heat conversion for palette {palette!r}")


def hot_mask(
    thermal_bgr: np.ndarray,
    palette: str,
    top_fraction: float = 0.10,
    min_contrast: float = 0.35,
    min_area_px: int = 20,
    max_frame_median: Optional[float] = None,
) -> np.ndarray:
    """True where a thermal frame is hot relative to the rest of the frame.

    Because the palette is rescaled every frame, there is no fixed
    temperature threshold. A pixel counts as hot when BOTH:
    - it is in the hottest `top_fraction` of the frame's pixels, and
    - its heat position is at least `min_contrast` above the frame's median.

    The contrast rule stops frames with nothing hot in them (where the
    palette is stretched over ordinary ground) from producing masks; the
    top-fraction rule stops a mostly warm frame from being masked almost
    entirely. Connected regions smaller than `min_area_px` are dropped.

    Optional frame-level gate, `max_frame_median`: when fire is in view,
    auto-ranging squeezes everything else towards the cold end of the
    palette, so the frame's median heat position is low. If the median is
    above `max_frame_median`, nothing in the frame stands out much and the
    mask is left empty. None (the default) turns the gate off. See the
    label audit for the evidence behind a value of about 0.25.
    """
    position = heat_position(thermal_bgr, palette)
    if max_frame_median is not None and np.median(position) > max_frame_median:
        return np.zeros(position.shape, dtype=bool)
    # A 3x3 median filter removes single-pixel JPEG speckle before thresholding.
    position = cv2.medianBlur((position * 255).astype(np.uint8), 3).astype(np.float32) / 255

    threshold = max(np.quantile(position, 1.0 - top_fraction), np.median(position) + min_contrast)
    hot = position >= threshold

    n, labels, stats, _ = cv2.connectedComponentsWithStats(hot.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area_px     # label 0 is the background
    return keep[labels]


def old_hot_mask(thermal_bgr: np.ndarray, threshold: int = 200, min_area_px: int = 50) -> np.ndarray:
    """The original notebook's method, kept for comparison in the audit:
    grayscale > 200, then contours of at least 50 pixels area."""
    gray = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY)
    binary = (gray > threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    kept = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    cv2.drawContours(mask, kept, -1, 1, thickness=-1)
    return mask.astype(bool)


# ---------------------------------------------------------------------------
# YOLO segmentation labels
# ---------------------------------------------------------------------------
def mask_to_yolo_seg(
    mask: np.ndarray,
    class_id: int = 0,
    epsilon_px: float = 1.0,
    min_area_px: float = 20.0,
) -> list[str]:
    """One YOLO segmentation label line per region of a mask.

    Each line is `class x1 y1 x2 y2 ...`, with coordinates divided by the
    image width/height so they run from 0 to 1. Outlines are simplified
    with cv2.approxPolyDP (points within `epsilon_px` of a straight line
    are dropped) so label files stay small.
    """
    height, width = mask.shape
    lines = []
    for polygon in mask_to_polygons(mask, min_area_px=min_area_px):
        simplified = cv2.approxPolyDP(polygon.reshape(-1, 1, 2), epsilon_px, closed=True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        xs = np.clip(simplified[:, 0] / width, 0.0, 1.0)
        ys = np.clip(simplified[:, 1] / height, 0.0, 1.0)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs, ys))
        lines.append(f"{class_id} {coords}")
    return lines
