"""Tests for the FLAME 2 labelling code (embereye/data/flame2.py).

Everything here uses small synthetic images made in the test, so the
tests run without the real dataset in data/.
"""
import cv2
import matplotlib
import numpy as np
import pytest

from embereye.data import flame2 as f2

SIZE = 64


# ---------------------------------------------------------------------------
# Synthetic frames
# ---------------------------------------------------------------------------
def inferno_bgr(position: float) -> np.ndarray:
    """The inferno colour at `position` (0..1), as a BGR uint8 triple."""
    r, g, b, _ = matplotlib.colormaps["inferno"](position)
    return np.array([b, g, r]) * 255


def rainbow_bgr(hue_deg: float, value: int = 230) -> np.ndarray:
    """A fully saturated colour of the given hue, as BGR."""
    hsv = np.uint8([[[int(hue_deg / 2) % 180, 255, value]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(float)


def frame_with_square(background_bgr, square_bgr, top_left=(20, 20), side=16) -> np.ndarray:
    """A frame of one colour with a square of another colour on it."""
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    frame[:] = np.clip(background_bgr, 0, 255).astype(np.uint8)
    y, x = top_left
    frame[y:y + side, x:x + side] = np.clip(square_bgr, 0, 255).astype(np.uint8)
    return frame


def square_mask(top_left=(20, 20), side=16) -> np.ndarray:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    y, x = top_left
    mask[y:y + side, x:x + side] = True
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return (a & b).sum() / max(1, (a | b).sum())


RAINBOW_COLD = np.array([2, 0, 124], dtype=float)      # FLAME 2's dark-red rainbow background (BGR)


# ---------------------------------------------------------------------------
# Labels and file names
# ---------------------------------------------------------------------------
def test_parse_labels_reads_ranges_and_skips_the_header(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text(
        "FirstFrameNumber - LastFrameNumber Fire(Y/N)Smoke(Y/N)\n"
        "Fire (Y/N) indicates whether or not there is fire visible\n"
        "\n"
        "1\t3\tNN\n"
        "4\t5\tYN\n"
        "6\t6\tYY\n"
    )
    labels = f2.parse_labels(path)
    assert labels == {
        1: (False, False), 2: (False, False), 3: (False, False),
        4: (True, False), 5: (True, False),
        6: (True, True),
    }


def test_parse_labels_warns_about_frames_in_two_ranges(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("1\t5\tYN\n5\t8\tYY\n")     # frame 5 is in both, as 41642 is in the real file
    with pytest.warns(UserWarning, match="more than one range"):
        labels = f2.parse_labels(path)
    assert labels[5] == (True, True)            # the later range wins


def test_frame_paths_follow_the_dataset_naming(tmp_path):
    rgb, thermal = f2.frame_paths(17701, root=tmp_path)
    assert rgb == tmp_path / "frames" / "254p RGB Images" / "254p RGB Frame (17701).jpg"
    assert thermal == tmp_path / "frames" / "254p Thermal Images" / "254p Thermal Frame (17701).jpg"


# ---------------------------------------------------------------------------
# Heat position
# ---------------------------------------------------------------------------
def test_inferno_position_recovers_where_a_colour_sits_on_the_palette():
    positions = np.linspace(0.05, 0.95, 10)
    frame = np.stack([inferno_bgr(p) for p in positions])[None].astype(np.uint8)
    recovered, distance = f2.inferno_position(frame)
    assert np.allclose(recovered[0], positions, atol=0.03)
    assert distance.max() < 6       # colours rounded for the lookup table, so not exactly zero


def test_rainbow_position_runs_from_dark_red_to_orange():
    colours = [
        RAINBOW_COLD,           # cold background
        rainbow_bgr(330),       # pink
        rainbow_bgr(300),       # magenta
        rainbow_bgr(240),       # blue
        rainbow_bgr(180),       # cyan
        rainbow_bgr(120),       # green
        rainbow_bgr(60),        # yellow
        rainbow_bgr(25),        # orange (hottest)
    ]
    frame = np.stack(colours)[None].astype(np.uint8)
    position = f2.rainbow_position(frame)[0]
    assert position[0] < 0.02
    assert all(a < b for a, b in zip(position, position[1:])), position


# ---------------------------------------------------------------------------
# Palette detection
# ---------------------------------------------------------------------------
def test_detect_palette_inferno():
    frame = frame_with_square(inferno_bgr(0.05), inferno_bgr(0.95))
    assert f2.detect_palette(frame) == f2.INFERNO


def test_detect_palette_rainbow():
    frame = frame_with_square(RAINBOW_COLD, rainbow_bgr(25))
    assert f2.detect_palette(frame) == f2.RAINBOW


def test_detect_palette_unknown_for_other_colours():
    frame = np.full((SIZE, SIZE, 3), (40, 200, 40), dtype=np.uint8)   # plain green
    assert f2.detect_palette(frame) == f2.UNKNOWN


# ---------------------------------------------------------------------------
# Hot masks
# ---------------------------------------------------------------------------
def test_hot_mask_finds_a_hot_square_on_a_cold_inferno_frame():
    frame = frame_with_square(inferno_bgr(0.05), inferno_bgr(0.95))
    mask = f2.hot_mask(frame, f2.INFERNO)
    assert iou(mask, square_mask()) > 0.85


def test_hot_mask_finds_a_hot_square_on_a_cold_rainbow_frame():
    frame = frame_with_square(RAINBOW_COLD, rainbow_bgr(25))
    mask = f2.hot_mask(frame, f2.RAINBOW)
    assert iou(mask, square_mask()) > 0.85


def test_hot_mask_ignores_a_frame_with_no_real_contrast():
    """Auto-ranging stretches the palette over ordinary ground when nothing
    is burning. Small differences like that must not become masks."""
    rng = np.random.default_rng(0)
    positions = 0.55 + rng.uniform(-0.1, 0.1, size=(SIZE, SIZE))
    frame = np.stack([inferno_bgr(p) for p in positions.ravel()]).reshape(SIZE, SIZE, 3).astype(np.uint8)
    assert not f2.hot_mask(frame, f2.INFERNO).any()


def test_frame_median_gate_empties_frames_without_a_cold_majority():
    # Mostly warm ground (0.45), a cold strip (0.05) and a hot square (0.95),
    # so the frame's median heat is about 0.45.
    frame = frame_with_square(inferno_bgr(0.45), inferno_bgr(0.95), top_left=(20, 30))
    frame[:, :20] = inferno_bgr(0.05).astype(np.uint8)
    assert f2.hot_mask(frame, f2.INFERNO).any()                              # gate off by default
    assert f2.hot_mask(frame, f2.INFERNO, max_frame_median=0.7).any()        # median ~0.45: passes
    assert not f2.hot_mask(frame, f2.INFERNO, max_frame_median=0.25).any()   # median ~0.45: gated


def test_hot_mask_drops_specks():
    frame = frame_with_square(inferno_bgr(0.05), inferno_bgr(0.95), side=3)   # 9 px, below min_area_px
    assert not f2.hot_mask(frame, f2.INFERNO).any()


def test_hot_mask_rejects_unknown_palettes():
    frame = frame_with_square(inferno_bgr(0.05), inferno_bgr(0.95))
    with pytest.raises(ValueError):
        f2.hot_mask(frame, f2.UNKNOWN)


def test_old_method_misses_rainbow_fire():
    """The notebook's grayscale > 200 rule finds nothing on the rainbow
    palette, which is why most fire frames used to get empty labels."""
    frame = frame_with_square(RAINBOW_COLD, rainbow_bgr(25, value=200))
    assert not f2.old_hot_mask(frame).any()
    assert f2.hot_mask(frame, f2.RAINBOW).any()


# ---------------------------------------------------------------------------
# YOLO segmentation labels
# ---------------------------------------------------------------------------
def test_yolo_seg_line_for_a_square():
    lines = f2.mask_to_yolo_seg(square_mask())
    assert len(lines) == 1

    parts = lines[0].split()
    assert parts[0] == "0"
    coords = np.array(parts[1:], dtype=float).reshape(-1, 2)
    assert len(coords) == 4                          # simplified down to the corners
    assert ((coords >= 0) & (coords <= 1)).all()     # normalised to the image size
    # Corners at pixels 20 and 35 of a 64-pixel image.
    assert np.allclose(sorted(set(np.round(coords[:, 0] * SIZE))), [20, 35])
    assert np.allclose(sorted(set(np.round(coords[:, 1] * SIZE))), [20, 35])


def test_yolo_seg_one_line_per_region_and_class_id():
    mask = square_mask((5, 5), 10) | square_mask((40, 40), 10)
    lines = f2.mask_to_yolo_seg(mask, class_id=2)
    assert len(lines) == 2
    assert all(line.startswith("2 ") for line in lines)


def test_yolo_seg_empty_mask_gives_no_lines():
    assert f2.mask_to_yolo_seg(np.zeros((SIZE, SIZE), dtype=bool)) == []
