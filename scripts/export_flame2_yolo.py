"""
export_flame2_yolo.py — Phase 4.4: split FLAME 2 into blocks and export YOLO-seg labels.

Usage:
    python scripts/export_flame2_yolo.py [--copy]

Rebuilds data/flame2/yolo_seg.nosync/ from scratch (not in git). The
".nosync" ending keeps iCloud from uploading the ~200k small files; the
folder can be regenerated from this script at any time.

    split.json            every frame: split, fire/smoke flags, palette variant, label status
    split_report.md       counts per split and stratum, checks, leakage, caveats
    excluded_frames.csv   fire frames left out because they have no mask
    rgb/                  images/{train,val,test}/, labels/{train,val,test}/, lists, yaml
    thermal/              the same, with the thermal images

Images are symlinks into data/flame2/frames/; --copy copies them instead.
Both modalities get the same labels (masks come from the thermal frames,
and RGB/thermal alignment is not corrected; see the label audit).

Labels, as decided in Phase 4 (docs/ARCHITECTURE.md, Labelling decisions):
- labels.txt is the frame-level truth: no-fire frames get an empty label
  file whatever hot_mask finds.
- Fire frames get hot_mask with the frame gate on, written with
  mask_to_yolo_seg. A fire frame with no mask is excluded from every split
  and listed, never written as an empty label.
- One class, 0: fire. Smoke needs hand labels: thermal masks only find heat.

Experiments (val and test are the same for all three):
- rgb/rgb_baseline.yaml       train on YN fire frames + no-fire frames
- rgb/rgb_yy_ablation.yaml    train on all fire frames + no-fire frames
- thermal/thermal.yaml        train on all fire frames + no-fire frames
{val,test}_{yn,yy,nofire}.txt list the groups so YY and YN can be reported separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))     # to reuse the hand-check helpers
from export_handcheck import label_blocks, thumbnail  # noqa: E402

from embereye.data import flame2 as f2  # noqa: E402
from embereye.data.flame2_split import GAP_FRAMES, SPLITS, VAL_TEST_BLOCKS, assign_splits, check_split, split_ranges  # noqa: E402

OUT = f2.FLAME2_DIR / "yolo_seg.nosync"
GATE = 0.25                     # hot_mask max_frame_median, as decided in Phase 4
THUMB_EVERY = 10                # frames sampled for the leakage check
SAME_VIEW = (0.8, 0.9)          # thumbnail correlations, in RGB AND thermal, counted as the same view
STRATA = ["no-fire", "inferno YY", "inferno YN", "rainbow-red YY", "rainbow-red YN",
          "rainbow-magenta YY", "rainbow-magenta YN"]


def stratum(rec: dict) -> str:
    return "no-fire" if not rec["fire"] else f"{rec['palette']} {rec['flag']}"


def label_frame(n: int, fire: bool) -> tuple[str, list[str], dict, np.ndarray]:
    """(palette variant, YOLO-seg lines, extra fields, thermal image) for one frame.
    The extra fields include `label`: "empty", "fire" or "excluded: ...". """
    thermal = cv2.imread(str(f2.frame_paths(n)[1]))
    variant = f2.palette_variant(thermal)
    if not fire:
        return variant, [], {"label": "empty"}, thermal
    if variant == f2.UNKNOWN:
        return variant, [], {"label": "excluded: unknown palette"}, thermal
    palette = f2.INFERNO if variant == f2.INFERNO else f2.RAINBOW
    mask = f2.hot_mask(thermal, palette, max_frame_median=GATE)
    if not mask.any():
        return variant, [], {"label": "excluded: empty mask"}, thermal
    lines = f2.mask_to_yolo_seg(mask)
    if not lines:
        return variant, [], {"label": "excluded: no outline"}, thermal
    return variant, lines, {"label": "fire", "mask_pct": round(100 * float(mask.mean()), 2), "regions": len(lines)}, thermal


def write_modality(name: str, image_index: int, used: list[dict], lines: dict[int, list[str]], copy: bool) -> dict[str, int]:
    """Images, labels, list files and yaml for one modality. Returns list sizes."""
    root = OUT / name
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)

    groups: dict[str, list[str]] = {}
    for rec in used:
        n, split = rec["frame"], rec["split"]
        stem = f"frame_{n:05d}"
        source = f2.frame_paths(n)[image_index]
        image = root / "images" / split / f"{stem}.jpg"
        if copy:
            shutil.copyfile(source, image)
        else:
            image.symlink_to(os.path.relpath(source, image.parent))
        text = "\n".join(lines[n])
        (root / "labels" / split / f"{stem}.txt").write_text(text + "\n" if text else "")

        entry = f"./images/{split}/{stem}.jpg"
        if split == "train":
            groups.setdefault("train_all", []).append(entry)
            if rec["flag"] in ("YN", "NN"):
                groups.setdefault("train_yn", []).append(entry)
        else:
            groups.setdefault(split, []).append(entry)
            group = {"YN": "yn", "YY": "yy", "NN": "nofire"}[rec["flag"]]
            groups.setdefault(f"{split}_{group}", []).append(entry)
    for group, entries in groups.items():
        (root / f"{group}.txt").write_text("\n".join(entries) + "\n")

    yamls = {"rgb": [("rgb_baseline", "train_yn", "RGB baseline: YN fire frames + no-fire frames"),
                     ("rgb_yy_ablation", "train_all", "RGB ablation: all fire frames (YY too) + no-fire frames")],
             "thermal": [("thermal", "train_all", "Thermal only: all fire frames + no-fire frames")]}[name]
    for stem, train, title in yamls:
        (root / f"{stem}.yaml").write_text(
            f"# FLAME 2 fire segmentation. {title}.\n"
            f"# Made by scripts/export_flame2_yolo.py. `path` is absolute: re-run the export if the repo moves.\n"
            f"path: {root.resolve()}\n"
            f"train: {train}.txt\nval: val.txt\ntest: test.txt\n"
            f"names:\n  0: fire\n"
        )
    return {group: len(entries) for group, entries in sorted(groups.items())}


def leakage(records: list[dict], thumbs: dict[int, tuple[np.ndarray, np.ndarray]]) -> list[dict]:
    """For sampled val/test frames: does train hold a near-copy of the same view?

    A pair counts only when the RGB and the thermal thumbnails both
    correlate above the threshold. RGB alone over-counts: frames full of
    smoke correlate at 0.7-0.8 without showing the same place."""
    train = [r["frame"] for r in records if r["split"] == "train" and r["frame"] in thumbs]
    train_thermal = np.array([thumbs[n][0] for n in train])
    train_rgb = np.array([thumbs[n][1] for n in train])
    rows = []
    for split in ("val", "test"):
        for group, is_fire in (("fire", True), ("no-fire", False)):
            sampled = [r["frame"] for r in records if r["split"] == split and r["fire"] == is_fire and r["frame"] in thumbs]
            both = np.minimum(np.array([thumbs[n][0] for n in sampled]) @ train_thermal.T,
                              np.array([thumbs[n][1] for n in sampled]) @ train_rgb.T)
            best = both.max(axis=1)
            rows.append({"split": split, "frames": group, "sampled": len(sampled),
                         **{f"same_view_above_{t}_pct": round(100 * float((best > t).mean()), 1) for t in SAME_VIEW}})
    return rows


def markdown_table(rows: list[dict]) -> str:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    args = parser.parse_args()

    labels_path = f2.FLAME2_DIR / "labels.txt"
    if not labels_path.exists():
        sys.exit(f"FLAME 2 not found at {f2.FLAME2_DIR}")
    labels = f2.parse_labels(labels_path)
    n_frames = max(labels)
    assignment = assign_splits(n_frames)

    print(f"Labelling {n_frames} frames...")
    records, lines, thumbs = [], {}, {}
    for n in range(1, n_frames + 1):
        fire, smoke = labels[n]
        variant, frame_lines, extra, thermal = label_frame(n, fire)
        rec = {"frame": n, "split": assignment[n], "fire": fire,
               "flag": ("YY" if smoke else "YN") if fire else "NN", "palette": variant, **extra}
        if rec["split"] is None:
            rec["label"] = "gap: " + rec["label"]
        records.append(rec)
        if rec["split"] is not None and not rec["label"].startswith("excluded"):
            lines[n] = frame_lines
            if n % THUMB_EVERY == 0:
                thumbs[n] = (thumbnail(thermal), thumbnail(cv2.imread(str(f2.frame_paths(n)[0]))))
        if n % 5000 == 0:
            print(f"  {n}/{n_frames}")

    used = [r for r in records if r["frame"] in lines]
    excluded = [r for r in records if r["label"].startswith("excluded")]
    problems = check_split({r["frame"]: r["split"] for r in used}, {r["frame"]: stratum(r) for r in used})
    if problems:
        sys.exit("Split check failed:\n  " + "\n  ".join(problems))

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    print("Writing rgb/ and thermal/...")
    sizes = {name: write_modality(name, index, used, lines, args.copy) for name, index in (("rgb", 0), ("thermal", 1))}

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    split_json = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "gap_frames": GAP_FRAMES,
        "val_test_blocks": VAL_TEST_BLOCKS,
        "ranges": split_ranges(assignment),
        "hot_mask": {"gate_max_frame_median": GATE, "other_parameters": "defaults in embereye.data.flame2"},
        "frames": records,
    }
    (OUT / "split.json").write_text(json.dumps(split_json, separators=(",", ":")))
    digest = hashlib.sha256((OUT / "split.json").read_bytes()).hexdigest()
    with open(OUT / "excluded_frames.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["frame", "flag", "palette", "label"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(excluded)

    # ---- report
    counts = Counter((stratum(r), r["split"]) for r in used)
    gap_counts = Counter(stratum(r) for r in records if r["split"] is None)
    excl_counts = Counter(stratum(r) for r in excluded)
    strata_rows = [{"stratum": s, "train": counts[(s, "train")], "val": counts[(s, "val")], "test": counts[(s, "test")],
                    "gap (unused)": gap_counts[s], "excluded": excl_counts[s]} for s in STRATA]
    strata_rows.append({"stratum": "all", **{k: sum(r[k] for r in strata_rows) for k in ("train", "val", "test", "gap (unused)", "excluded")}})

    block_rows = []
    for first, last, flags in label_blocks(labels_path):
        members = [r for r in records if first <= r["frame"] <= last]
        split_mix = Counter(r["split"] for r in members if r["frame"] in lines)
        block_rows.append({"block": f"{first}-{last}", "flags": flags, "frames": len(members),
                           "train": split_mix["train"], "val": split_mix["val"], "test": split_mix["test"],
                           "gap": sum(r["split"] is None for r in members),
                           "excluded": sum(r["label"].startswith("excluded") for r in members)})

    # Checks, read back from the files written (rgb and thermal labels must also agree).
    written: dict[int, str] = {}
    in_two_splits, label_differs = 0, 0
    for split in SPLITS:
        for path in (OUT / "rgb" / "labels" / split).iterdir():
            n = int(path.stem.split("_")[1])
            in_two_splits += n in written
            written[n] = split
            label_differs += path.read_text() != (OUT / "thermal" / "labels" / split / path.name).read_text()
    by_frame = {r["frame"]: r for r in records}
    written_sorted = sorted(written)
    too_close = sum(1 for a, b in zip(written_sorted, written_sorted[1:]) if written[a] != written[b] and b - a <= GAP_FRAMES)
    nofire_nonempty = sum(1 for n, s in written.items()
                          if not by_frame[n]["fire"] and (OUT / "rgb" / "labels" / s / f"frame_{n:05d}.txt").stat().st_size)
    fire_empty = sum(1 for n, s in written.items()
                     if by_frame[n]["fire"] and not (OUT / "rgb" / "labels" / s / f"frame_{n:05d}.txt").stat().st_size)
    checks = [
        ("every split has frames of every stratum", all(counts[(s, split)] > 0 for s in STRATA for split in SPLITS)),
        (f"frames of different splits are more than {GAP_FRAMES} apart ({too_close} pairs too close)", too_close == 0),
        (f"no frame is in two splits ({in_two_splits} found)", in_two_splits == 0),
        (f"every no-fire frame has an empty label file ({nofire_nonempty} don't)", nofire_nonempty == 0),
        (f"every fire frame in a split has at least one outline ({fire_empty} don't)", fire_empty == 0),
        (f"rgb and thermal labels are identical ({label_differs} differ)", label_differs == 0),
        (f"{len(written)} label files written = {len(used)} frames used", len(written) == len(used)),
    ]
    leaks = leakage(used, thumbs)
    ranges = split_ranges(assignment)

    report = [
        "# FLAME 2 split and YOLO-seg export", "",
        f"Made {split_json['created']} from commit `{commit}` by `scripts/export_flame2_yolo.py`. "
        f"split.json sha256: `{digest}`.", "",
        f"Images are {'copies' if args.copy else 'symlinks into data/flame2/frames/'}. "
        "Labels: no-fire frames empty; fire frames from hot_mask with the frame gate "
        f"(max_frame_median={GATE}), exported by mask_to_yolo_seg. One class, 0: fire.", "",
        "## Blocks", "",
        *(f"- **{split}:** " + ", ".join(f"{a}-{b}" for a, b in ranges[split]) for split in SPLITS),
        f"- **gap (in no split):** {GAP_FRAMES} frames on the train side of every val/test block", "",
        "## Frames per stratum", "", markdown_table(strata_rows), "",
        "## Experiment sets", "",
        "| list | rgb | thermal |", "|---|---|---|",
        *(f"| {g} | {sizes['rgb'].get(g, '-')} | {sizes['thermal'].get(g, '-')} |"
          for g in sorted(set(sizes["rgb"]) | set(sizes["thermal"]))), "",
        "`train_yn` (YN fire + no-fire) is the RGB baseline; `train_all` is the YY ablation and the thermal training set.", "",
        "## Checks", "", *(f"- {'PASS' if ok else 'FAIL'}: {text}" for text, ok in checks), "",
        "## Per labels.txt block", "", markdown_table(block_rows), "",
        "## Leakage: does train hold the same view as val/test?", "",
        f"Every {THUMB_EVERY}th frame of val and test, compared with every {THUMB_EVERY}th train frame by "
        "32x32 thumbnail correlation; a pair is the same view only if RGB and thermal both correlate above the "
        "threshold. The gaps stop neighbouring frames leaking; they can't stop the drone hovering over, or "
        "returning to, one view for minutes. Before the burn it hovers over the same grassland, so no-fire "
        "val/test frames have near-copies in train: treat no-fire results as optimistic (on top of the scene "
        "confound below), and fire-frame metrics as the headline.", "",
        markdown_table(leaks), "",
        "## Excluded fire frames", "",
        f"{len(excluded)} fire frames have no mask and are in no split: "
        + ", ".join(f"{k}: {v}" for k, v in Counter(r["label"] for r in excluded).items())
        + ". Listed in `excluded_frames.csv`.", "",
        "## Caveats", "",
        "- **Scene confound.** Every no-fire frame is open grassland and sky from before the burn; the fire "
        "frames are forest. Image-level false-positive rates on these no-fire frames are not evidence that a "
        "detector avoids false alarms in forest. TODO(Will): add forest no-fire negatives from another "
        "dataset (e.g. FLAME 1's No_Fire set).",
        "- **Smoke** is not labelled. The `smoke` flag only says smoke fills at least half the RGB frame.",
        "- **RGB alignment.** Masks come from the thermal frames and are applied to RGB as they are; the label "
        "audit found no consistent RGB/thermal transform.",
        "- **Label quality** was checked on 48 hand-labelled frames: mean IoU 0.63, every stratum at least 0.53 "
        "(docs/findings/flame2-label-validation.md).",
        "- The planned hand-labelled test set should be drawn from the test blocks.", "",
    ]
    (OUT / "split_report.md").write_text("\n".join(report))
    print("\n".join(report))
    print(f"Written to {OUT}/")


if __name__ == "__main__":
    main()
