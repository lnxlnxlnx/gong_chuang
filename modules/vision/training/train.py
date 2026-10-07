#!/usr/bin/env python
"""Train (and export) the obstacle detector from the split YOLO dataset.

Run :mod:`training.prepare_dataset` once first; it writes ``data.yaml``.
This script is a thin, reproducible wrapper around ultralytics so the whole
job is one command with the project's paths baked in::

    python training/train.py                      # yolov8n, 100 epochs, GPU 0
    python training/train.py --epochs 300 --model yolov8s.pt
    python training/train.py --export-onnx        # also write models/best_obstacle.onnx

Artefacts land in ``models/runs/obstacle/`` (weights, curves, confusion
matrix).

``--model`` names an ultralytics checkpoint: a pretrained ``*.pt`` (searched
locally first, downloaded from GitHub only when missing), a previous run's
``best.pt``, or a bare ``yolov8n.yaml`` to train from scratch with no
network access at all.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from project_paths import models_path

DEFAULT_DATA = models_path("dataset", "data.yaml")
DEFAULT_PROJECT = models_path("runs")
DEFAULT_ONNX_OUT = models_path("best_obstacle.onnx")

from ultralytics import YOLO  # noqa: E402  (kept below the path constants)

#: Class table currently compiled into ``vision/yolov8.py``.
DEPLOYED_CLASS_NAMES = [
    "obj-red", "obj-green", "obj-blue", "obj-yellow", "obj-light blue",
    "obj-black", "target-1", "target-2", "target-3", "zhangai",
]


def read_num_classes(data_yaml: Path) -> int | None:
    """Read ``nc:`` out of the dataset yaml without requiring PyYAML."""
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("nc:"):
            try:
                return int(stripped.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def warn_about_class_table(num_classes: int | None) -> None:
    """Explain what must change when the new model cannot replace the old one."""
    if num_classes is None or num_classes == len(DEPLOYED_CLASS_NAMES):
        return
    print("\n" + "=" * 72)
    print(f"NOTE: this dataset has {num_classes} class(es) but the running robot")
    print(f"      expects {len(DEPLOYED_CLASS_NAMES)}: {', '.join(DEPLOYED_CLASS_NAMES)}")
    print("      The exported single-class model already outputs correct boxes, but")
    print("      the label text will be wrong until the class table matches. Either")
    print(f"      set vision/yolov8.py::DEFAULT_CLASS_NAMES to the {num_classes} new")
    print("      name(s), or pass class_names=[...] to create_detector().")
    print("=" * 72 + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the obstacle YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="dataset yaml from prepare_dataset.py")
    parser.add_argument("--model", default=str(models_path("yolov8n.pt")),
                        help="base weights or a previous checkpoint (*.pt); "
                             "default resolves under models/ so it does not follow the cwd")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="training input size; must be a multiple of 32")
    parser.add_argument("--batch", type=int, default=-1,
                        help="-1 lets ultralytics auto-fit the batch to the GPU")
    parser.add_argument("--device", default="0",
                        help="'0' for the first GPU, 'cpu' to train on the CPU")
    parser.add_argument("--workers", type=int, default=8,
                        help="dataloader processes; use 0 if Windows hangs")
    parser.add_argument("--patience", type=int, default=50,
                        help="stop early after this many epochs without improvement")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="fraction of the training set to use (smoke tests)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT,
                        help="run output root")
    parser.add_argument("--name", default="obstacle", help="run directory name")
    parser.add_argument("--resume", action="store_true",
                        help="continue the last run in project/name")
    parser.add_argument("--export-onnx", action="store_true",
                        help="export the best checkpoint for the OpenCV detector")
    parser.add_argument("--opset", type=int, default=12,
                        help="ONNX opset used for the export")
    parser.add_argument("--onnx-out", type=Path, default=DEFAULT_ONNX_OUT,
                        help="where to copy the exported ONNX model")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        print(f"dataset yaml not found: {data_yaml}\n"
              f"run 'python training/prepare_dataset.py' first", file=sys.stderr)
        return 2
    if args.imgsz % 32:
        print(f"--imgsz must be a multiple of 32, got {args.imgsz}", file=sys.stderr)
        return 2

    num_classes = read_num_classes(data_yaml)
    print(f"dataset : {data_yaml}")
    print(f"classes : {num_classes}")
    print(f"model   : {args.model}")
    warn_about_class_table(num_classes)

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        seed=args.seed,
        project=str(args.project.resolve()),
        name=args.name,
        resume=args.resume,
        pretrained=True,
        plots=True,
    )

    save_dir = Path(getattr(model.trainer, "save_dir", args.project / args.name))
    best = save_dir / "weights" / "best.pt"
    print(f"\nrun directory : {save_dir}")
    print(f"best weights  : {best}")
    print("inspect results.png / confusion_matrix.png / val_batch*_pred.jpg there")

    if args.export_onnx:
        if not best.is_file():
            print(f"cannot export: {best} is missing", file=sys.stderr)
            return 3
        exported = YOLO(str(best)).export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=False,  # vision.YOLOv8ONNXDetector requires a fixed input
        )
        destination = args.onnx_out.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported, destination)
        print(f"exported ONNX : {destination}")
        print("windows reads models/best.onnx -- back up the old file before "
              "replacing it, or point create_detector() at the new path")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
