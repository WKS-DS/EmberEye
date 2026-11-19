"""
masks.py — Turning True/False pixel masks into outlines.

Shared by the simulator (fire masks from MuJoCo) and the FLAME 2 labelling
code (fire masks from thermal frames), so both produce outlines in exactly
the same way — and in the same layout YOLO's segmentation output uses.
"""

from __future__ import annotations

import cv2
import numpy as np


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 4.0) -> list[np.ndarray]:
    """Trace the outline of each connected True region in a mask.

    Returns one (N, 2) float array of pixel (x, y) points per region — the
    same layout YOLO's segmentation output uses (`result.masks.xy`), so
    code written against the sim also works with the real detector.

    Only the OUTER outline is traced. A ring of flame around a burned-out
    middle therefore comes back as one filled shape, which is also how the
    real detector's polygons behave. Specks smaller than `min_area_px` are
    dropped.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3 or cv2.contourArea(contour) < min_area_px:
            continue
        polygons.append(contour.reshape(-1, 2).astype(np.float32))
    return polygons


def polygons_to_mask(polygons: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """The reverse of mask_to_polygons: fill each (N, 2) pixel (x, y)
    polygon into a True/False mask of the given (height, width).
    Points are rounded to the nearest pixel."""
    canvas = np.zeros(shape, dtype=np.uint8)
    for polygon in polygons:
        points = np.rint(np.asarray(polygon, dtype=np.float64)).astype(np.int32)
        if len(points) >= 3:
            cv2.fillPoly(canvas, [points.reshape(-1, 1, 2)], 1)
    return canvas.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection over union of two True/False masks of the same shape.
    Two empty masks agree completely, so their IoU is 1."""
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)
