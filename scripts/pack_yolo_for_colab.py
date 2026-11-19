"""
pack_yolo_for_colab.py — Bundle one YOLO-seg training set into a single zip for Google Colab.

Usage:
    python scripts/pack_yolo_for_colab.py                    # RGB baseline: train_yn + val (~1.75 GB)
    python scripts/pack_yolo_for_colab.py --limit 50 --out /tmp/tiny.zip   # a few frames, for a quick test

Reads the Phase 4.4 export (data/flame2/yolo_seg.nosync/, made by
scripts/export_flame2_yolo.py) and writes data/flame2/colab/<name>.zip.
Upload that zip to Google Drive and train with notebooks/03_train_colab.ipynb.

Why a zip: the export's images are symlinks into data/flame2/frames/, so
the zip stores the real files. And Colab reads one big file from Drive
much faster than tens of thousands of small ones.

The test split is left out on purpose: it stays on this machine until the
final evaluation, so no training choice can be tuned against it.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EXPORT = PROJECT / "data" / "flame2" / "yolo_seg.nosync"
# The validation lists: all of val, plus its fire-frame and no-fire subsets for evaluation.
VAL_LISTS = ["val.txt", "val_yn.txt", "val_yy.txt", "val_nofire.txt"]


def evenly_spaced(lines: list[str], limit: int | None) -> list[str]:
    """`limit` lines spread over the whole list (all of them if limit is None).

    The lists are in frame order, and fire and no-fire frames sit in long
    runs, so taking the first few would give a test set of one kind only.
    """
    if limit is None or limit >= len(lines):
        return lines
    step = len(lines) / limit
    return [lines[int(i * step)] for i in range(limit)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sensor", choices=["rgb", "thermal"], default="rgb")
    parser.add_argument("--train-list", default="train_yn.txt", help="train_yn.txt (baseline) or train_all.txt (YY)")
    parser.add_argument("--limit", type=int, help="keep only this many evenly spaced frames per list")
    parser.add_argument("--out", type=Path, help="default: data/flame2/colab/flame2_<sensor>_<train list>.zip")
    args = parser.parse_args()

    source = EXPORT / args.sensor
    out = args.out or PROJECT / "data" / "flame2" / "colab" / f"flame2_{args.sensor}_{Path(args.train_list).stem}.zip"
    out.parent.mkdir(parents=True, exist_ok=True)

    lists = {
        name: evenly_spaced((source / name).read_text().split(), args.limit) for name in [args.train_list, *VAL_LISTS]
    }
    if args.limit is not None:
        # The subsets must only name frames that are in the zip.
        kept = set(lists["val.txt"])
        for name in VAL_LISTS[1:]:
            lists[name] = [line for line in (source / name).read_text().split() if line in kept]

    images = sorted({line for lines in lists.values() for line in lines})
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as archive:  # JPEGs don't compress further
        for name, lines in lists.items():
            archive.writestr(name, "\n".join(lines) + "\n")
        for image in images:
            relative = Path(image)  # e.g. images/train/frame_01601.jpg
            label = Path("labels", *relative.parts[1:-1], relative.stem + ".txt")
            archive.write((source / relative).resolve(), relative.as_posix())  # resolve() follows the symlink
            archive.write(source / label, label.as_posix())

    print(f"{out}: {len(images)} images, {out.stat().st_size / 1e9:.2f} GB")
    for name, lines in lists.items():
        print(f"  {name}: {len(lines)} frames")


if __name__ == "__main__":
    main()
