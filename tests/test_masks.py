"""Tests for the shared mask helpers (embereye/masks.py)."""
import numpy as np

from embereye.masks import iou, mask_to_polygons, polygons_to_mask


def test_polygons_to_mask_fills_a_square():
    square = np.array([[10, 10], [19, 10], [19, 19], [10, 19]], dtype=float)
    mask = polygons_to_mask([square], (32, 32))
    assert mask.sum() == 100                    # pixels 10..19 in both directions
    assert mask[10, 10] and mask[19, 19] and not mask[20, 20]


def test_polygons_to_mask_round_trips_mask_to_polygons():
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:15, 5:20] = True
    mask[25:35, 22:30] = True
    assert iou(polygons_to_mask(mask_to_polygons(mask), mask.shape), mask) == 1.0


def test_polygons_to_mask_skips_degenerate_shapes():
    line = np.array([[1, 1], [5, 5]], dtype=float)
    assert not polygons_to_mask([line], (10, 10)).any()


def test_iou():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    a[0:4, 0:5] = True      # 20 pixels
    b[2:6, 0:5] = True      # 20 pixels, 10 shared
    assert iou(a, b) == 10 / 30
    assert iou(a, a) == 1.0
    assert iou(a, ~a) == 0.0


def test_iou_of_two_empty_masks_is_one():
    empty = np.zeros((5, 5), dtype=bool)
    assert iou(empty, empty) == 1.0
