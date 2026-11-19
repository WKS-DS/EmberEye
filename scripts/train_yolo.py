"""
train_yolo.py — Train YOLO11-seg on a dataset zip made by scripts/pack_yolo_for_colab.py.

Usage (on Colab, see notebooks/03_train_colab.ipynb; works anywhere with a GPU):
    python scripts/train_yolo.py --data-dir /content/flame2_rgb --out /content/drive/MyDrive/vigil/runs
    python scripts/train_yolo.py --data-dir /content/flame2_rgb --out /content/drive/MyDrive/vigil/runs --resume

--data-dir is the unzipped dataset. --out should be on Google Drive: the
weights are saved there after every epoch, so a dropped Colab session
loses at most one epoch, and --resume carries on from the last one.

Defaults are for the RGB baseline: the smallest model (YOLO11n-seg), at
256 px because the frames are 254 x 254, with a fixed seed. Training
stops early once the validation score hasn't improved for --patience
epochs. Ultralytics' own validation score includes the no-fire frames,
which have near-copies in training, so treat it as a progress bar; the
fire-frame metrics come from a separate evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def write_dataset_yaml(data_dir: Path, train_list: str, val_list: str) -> Path:
    """The dataset file Ultralytics reads, with this machine's absolute path in it."""
    yaml_path = data_dir / "dataset.yaml"
    yaml_path.write_text(
        f"# Written by scripts/train_yolo.py\n"
        f"path: {data_dir.resolve()}\n"
        f"train: {train_list}\n"
        f"val: {val_list}\n"
        f"names:\n  0: fire\n"
    )
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, required=True, help="the unzipped dataset")
    parser.add_argument("--out", type=Path, required=True, help="folder for runs; put it on Google Drive")
    parser.add_argument("--name", default="rgb_baseline", help="run name: results go to <out>/<name>")
    parser.add_argument("--train-list", default="train_yn.txt")
    parser.add_argument("--val-list", default="val.txt")
    parser.add_argument("--model", default="yolo11n-seg.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20, help="stop after this many epochs without improvement")
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default=None, help="e.g. 0 for the first GPU, mps, cpu; default: auto")
    parser.add_argument("--resume", action="store_true", help="carry on from <out>/<name>/weights/last.pt")
    args = parser.parse_args()

    from ultralytics import YOLO  # imported here so --help works without it

    if args.resume:
        last = args.out / args.name / "weights" / "last.pt"
        if not last.exists():
            raise SystemExit(f"Nothing to resume: {last} doesn't exist.")
        YOLO(str(last)).train(resume=True)
    else:
        if (args.out / args.name).exists():
            raise SystemExit(f"{args.out / args.name} already exists. Use --resume, or a new --name.")
        data = write_dataset_yaml(args.data_dir, args.train_list, args.val_list)
        YOLO(args.model).train(
            data=str(data),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(args.out),
            name=args.name,
            seed=0,
            deterministic=True,
        )
    print(f"Done. Best weights: {args.out / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
