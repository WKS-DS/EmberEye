"""Tests for the FLAME 2 block split (embereye/data/flame2_split.py). No dataset needed."""
import pytest

from embereye.data.flame2_split import VAL_TEST_BLOCKS, assign_splits, check_split, split_ranges

BLOCKS = [(11, 20, "val"), (41, 50, "test")]


def test_blocks_get_their_split_and_gaps_get_none():
    assignment = assign_splits(70, BLOCKS, gap=5)
    assert all(assignment[n] == "val" for n in range(11, 21))
    assert all(assignment[n] == "test" for n in range(41, 51))
    assert all(assignment[n] is None for n in [*range(6, 11), *range(21, 26), *range(36, 41), *range(51, 56)])
    assert assignment[5] == assignment[26] == assignment[35] == assignment[56] == "train"


def test_split_ranges_lists_contiguous_runs():
    ranges = split_ranges(assign_splits(70, BLOCKS, gap=5))
    assert ranges == {"train": [(1, 5), (26, 35), (56, 70)], "val": [(11, 20)], "test": [(41, 50)]}


def test_bad_blocks_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assign_splits(70, [(11, 20, "val"), (18, 30, "test")], gap=5)
    with pytest.raises(ValueError, match="within"):
        assign_splits(70, [(11, 20, "val"), (24, 30, "test")], gap=5)
    with pytest.raises(ValueError, match="Bad block"):
        assign_splits(70, [(60, 80, "val")], gap=5)


def test_check_split_finds_missing_strata_and_frames_too_close():
    assignment = assign_splits(70, BLOCKS, gap=5)
    strata = {n: "a" for n in assignment}
    strata[45] = "b"                                      # only in test
    problems = check_split(assignment, strata, gap=5)
    assert "train has no b frames" in problems and "val has no b frames" in problems

    assignment[8] = "train"                               # a gap frame put back into train
    assert any("only 3 apart" in p for p in check_split(assignment, {}, gap=5))


def test_the_flame2_split_keeps_its_rules():
    assignment = assign_splits(53451)
    assert check_split(assignment, {}) == []
    counts = {s: sum(v == s for v in assignment.values()) for s in ("train", "val", "test", None)}
    assert counts["val"] == sum(last - first + 1 for first, last, s in VAL_TEST_BLOCKS if s == "val")
    assert 0.09 < counts["val"] / 53451 < 0.11 and 0.09 < counts["test"] / 53451 < 0.11
    assert all(assignment[n] == "test" for n in range(1, 1301))       # test starts at the first frame
    assert assignment[13700] == assignment[13701] == "train"           # no gap at the no-fire/fire boundary
