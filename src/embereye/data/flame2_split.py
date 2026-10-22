"""
flame2_split.py — Which FLAME 2 frames go to train, val and test.

FLAME 2 is video, so neighbouring frames are near-copies. A random split
would put near-copies of test frames into training. Instead the timeline is
cut into contiguous blocks:

- VAL_TEST_BLOCKS lists the val and test blocks. They are placed by hand so
  that val and test each contain every stratum: no-fire frames, inferno
  fire frames with heavy smoke (YY) and without (YN), and rainbow fire
  frames of both background variants (see flame2.palette_variant), YY and YN.
- Everything else is train, except GAP_FRAMES frames on the train side of
  every val/test block. Those frames go to no split, so frames from
  different splits are always more than GAP_FRAMES apart.

The awkward strata decide the placement. All no-fire frames are 1-13700.
Inferno YN exists only in 14700-15980, so its val and test blocks straddle
the YY/YN boundaries at 14699/14700 and 15980/15981. Rainbow-red YN is rare
and sits mainly around 27,000, 36,000-41,000 and 52,000.

Gaps stop neighbouring frames from leaking across splits, but not a view
the drone comes back to minutes later; the export report measures that.
"""

from __future__ import annotations

from typing import Optional

GAP_FRAMES = 300
SPLITS = ("train", "val", "test")

# (first frame, last frame, split), in timeline order.
VAL_TEST_BLOCKS: list[tuple[int, int, str]] = [
    (1, 1300, "test"),          # no-fire, start of the flight
    (6801, 8100, "val"),        # no-fire, middle of the 1-13700 run
    (14500, 14899, "test"),     # inferno: 200 YY (block 13701-14699) + 200 YN (block 14700-15980)
    (15800, 16199, "val"),      # inferno: 181 YN + 219 YY (block 15981-19802)
    (23300, 24999, "val"),      # early rainbow, YY, red and magenta backgrounds
    (27100, 28899, "test"),     # rainbow YN (block 27184-27514) and YY, both backgrounds
    (40500, 42299, "val"),      # rainbow YN (to 41642) then YY, both backgrounds
    (51650, 53451, "test"),     # end of the flight: rainbow YY and YN, both backgrounds
]


def assign_splits(
    n_frames: int,
    blocks: list[tuple[int, int, str]] = VAL_TEST_BLOCKS,
    gap: int = GAP_FRAMES,
) -> dict[int, Optional[str]]:
    """The split of every frame 1..n_frames: "train", "val", "test", or
    None for a gap frame that belongs to no split.

    Raises ValueError if blocks overlap, run off the timeline, or blocks of
    different splits are not more than `gap` frames apart.
    """
    ordered = sorted(blocks)
    for first, last, split in ordered:
        if split not in ("val", "test") or not 1 <= first <= last <= n_frames:
            raise ValueError(f"Bad block {(first, last, split)}")
    for (_, last_a, split_a), (first_b, _, split_b) in zip(ordered, ordered[1:]):
        if first_b <= last_a:
            raise ValueError(f"Blocks overlap at frame {first_b}")
        if split_a != split_b and first_b - last_a <= gap:
            raise ValueError(f"{split_a} block ending at {last_a} is within {gap} frames of a {split_b} block")

    assignment: dict[int, Optional[str]] = {n: "train" for n in range(1, n_frames + 1)}
    for first, last, split in ordered:
        for n in range(first, last + 1):
            assignment[n] = split
    for first, last, _ in ordered:
        for n in [*range(first - gap, first), *range(last + 1, last + gap + 1)]:
            if assignment.get(n) == "train":
                assignment[n] = None
    return assignment


def split_ranges(assignment: dict[int, Optional[str]]) -> dict[str, list[tuple[int, int]]]:
    """The contiguous runs of frames in each split, e.g. {"val": [(6801, 8100), ...]}."""
    ranges: dict[str, list[tuple[int, int]]] = {split: [] for split in SPLITS}
    run_start, run_split, previous = None, None, None
    for n in sorted(assignment) + [None]:
        split = assignment.get(n) if n is not None else None
        if n is None or split != run_split or n != previous + 1:
            if run_split is not None:
                ranges[run_split].append((run_start, previous))
            run_start, run_split = n, split
        previous = n
    return ranges


def check_split(assignment: dict[int, Optional[str]], strata: dict[int, str], gap: int = GAP_FRAMES) -> list[str]:
    """Problems with a split; an empty list means none.

    - Every split must contain every stratum found in `strata` (frame ->
      stratum name, for the frames that will be used).
    - Frames of different splits must be more than `gap` frames apart.
    A frame can't be in two splits: `assignment` gives each frame one.
    """
    problems = []
    wanted = set(strata.values())
    for split in SPLITS:
        present = {strata[n] for n, s in assignment.items() if s == split and n in strata}
        for missing in sorted(wanted - present):
            problems.append(f"{split} has no {missing} frames")

    used = sorted(n for n, s in assignment.items() if s is not None)
    for a, b in zip(used, used[1:]):
        if assignment[a] != assignment[b] and b - a <= gap:
            problems.append(f"frames {a} ({assignment[a]}) and {b} ({assignment[b]}) are only {b - a} apart")
    return problems
