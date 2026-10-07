#!/usr/bin/env python
"""Turn the flat ``models/data`` dump into an ultralytics YOLO dataset.

The raw capture keeps every frame in one directory (``models/data/image``)
with the matching YOLO ``.txt`` in a sibling directory
(``models/data/labels``).  Ultralytics expects the classic split layout::

    models/dataset/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml

This script validates every label line, pairs images with labels by file
stem, copies a deterministic train/val split into that layout and writes
``data.yaml``.  It only reads the source dataset.

Usage::

    python training/prepare_dataset.py
    python training/prepare_dataset.py --val-ratio 0.2 --seed 42 --force
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from project_paths import models_path

DEFAULT_SOURCE = models_path("data")
DEFAULT_OUTPUT = models_path("dataset")

#: Directory names accepted for the image / label halves of a flat dataset.
IMAGE_DIR_NAMES = ("image", "images", "img", "JPEGImages")
LABEL_DIR_NAMES = ("labels", "label", "annotations")

#: Extensions treated as input frames.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _find_child(root: Path, names) -> Path | None:
    """Return the first existing child directory named in ``names``."""
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def locate_dirs(source: Path) -> tuple[Path, Path]:
    """Find the image and label directories inside ``source``.

    Falls back to ``source`` itself for both when the dataset is already
    flat (frames and ``.txt`` side by side in one directory).
    """
    image_dir = _find_child(source, IMAGE_DIR_NAMES)
    label_dir = _find_child(source, LABEL_DIR_NAMES)
    if image_dir is None and label_dir is None:
        return source, source
    if image_dir is None:
        image_dir = source
    if label_dir is None:
        label_dir = source
    return image_dir, label_dir


def read_class_names(source: Path) -> list[str]:
    """Read ``classes.txt`` (one class per line) when it is present."""
    for candidate in (source / "classes.txt", source / "labels" / "classes.txt"):
        if candidate.is_file():
            names = [line.strip() for line in
                     candidate.read_text(encoding="utf-8").splitlines()]
            names = [name for name in names if name]
            if names:
                return names
    return []


def read_label_file(path: Path, num_classes: int) -> tuple[int, list[str]]:
    """Validate one YOLO label file; return ``(box_count, problems)``."""
    problems: list[str] = []
    boxes = 0
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            problems.append(f"line {number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            class_id = int(float(fields[0]))
            values = [float(value) for value in fields[1:]]
        except ValueError:
            problems.append(f"line {number}: non-numeric field")
            continue
        if num_classes and not 0 <= class_id < num_classes:
            problems.append(f"line {number}: class {class_id} outside 0..{num_classes - 1}")
        if any(value < 0.0 or value > 1.0 for value in values):
            problems.append(f"line {number}: coordinate outside [0, 1]")
        if values[2] <= 0.0 or values[3] <= 0.0:
            problems.append(f"line {number}: non-positive box size")
        boxes += 1
    return boxes, problems


def collect_pairs(image_dir: Path, label_dir: Path):
    """Pair frames with labels by stem and report every mismatch."""
    images = {path.stem: path for path in sorted(image_dir.iterdir())
              if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES}
    labels = {path.stem: path for path in sorted(label_dir.iterdir())
              if path.is_file() and path.suffix.lower() == ".txt"
              and path.stem != "classes"}

    paired = sorted(set(images) & set(labels))
    images_without_labels = sorted(set(images) - set(labels))
    labels_without_images = sorted(set(labels) - set(images))
    return images, labels, paired, images_without_labels, labels_without_images


def split_stems(stems, val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    """Shuffle ``stems`` deterministically and cut a validation slice off."""
    ordered = sorted(stems)
    random.Random(seed).shuffle(ordered)
    if len(ordered) == 1:
        return ordered, ordered
    val_count = int(round(len(ordered) * val_ratio))
    val_count = max(1, min(val_count, len(ordered) - 1))
    return ordered[val_count:], ordered[:val_count]


def write_data_yaml(path: Path, dataset_root: Path, class_names) -> None:
    """Write the ultralytics dataset description next to the split folders."""
    lines = [
        "# Generated by training/prepare_dataset.py -- safe to regenerate.",
        f"path: {dataset_root}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(class_names)}",
        "names:",
    ]
    lines += [f"  {index}: {name}" for index, name in enumerate(class_names)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_split(images, labels, output: Path, stems, split: str) -> None:
    """Copy one side of the split into ``images/<split>`` and ``labels/<split>``."""
    for kind, source in (("images", images), ("labels", labels)):
        target = output / kind / split
        target.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            shutil.copy2(source[stem], target / source[stem].name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="flat dataset directory (default: models/data)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="split dataset directory (default: models/dataset)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="fraction of frames used for validation (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="shuffle seed for a reproducible split (default: 42)")
    parser.add_argument("--force", action="store_true",
                        help="delete an existing output directory first")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    source = args.source.resolve()
    output = args.output.resolve()

    if not source.is_dir():
        print(f"source dataset not found: {source}", file=sys.stderr)
        return 2

    class_names = read_class_names(source)
    if not class_names:
        print("warning: no classes.txt found, assuming a single class 'obstacle'",
              file=sys.stderr)
        class_names = ["obstacle"]
    num_classes = len(class_names)

    image_dir, label_dir = locate_dirs(source)
    images, labels, paired, missing_labels, missing_images = collect_pairs(
        image_dir, label_dir)

    box_total = 0
    empty, broken = [], []
    for stem in paired:
        boxes, problems = read_label_file(labels[stem], num_classes)
        box_total += boxes
        if boxes == 0:
            empty.append(stem)
        if problems:
            broken.append((stem, problems))

    print(f"source       : {source}")
    print(f"images       : {len(images)}  ({image_dir})")
    print(f"labels       : {len(labels)}  ({label_dir})")
    print(f"paired       : {len(paired)}  boxes: {box_total}")
    print(f"classes      : {num_classes} -> {class_names}")
    if missing_labels:
        print(f"images without labels ({len(missing_labels)}): "
              f"{', '.join(missing_labels[:5])}")
    if missing_images:
        print(f"labels without images ({len(missing_images)}): "
              f"{', '.join(missing_images[:5])}")
    if empty:
        print(f"empty label files ({len(empty)}): {', '.join(empty[:5])}")
    if broken:
        print(f"invalid label files ({len(broken)}):")
        for stem, problems in broken[:10]:
            print(f"  {stem}: {'; '.join(problems[:3])}")
    if not paired:
        print("nothing to split: no image/label pair found", file=sys.stderr)
        return 2

    if output.exists():
        if not args.force:
            print(f"output already exists: {output} (use --force to rebuild)",
                  file=sys.stderr)
            return 2
        shutil.rmtree(output)

    train_stems, val_stems = split_stems(paired, args.val_ratio, args.seed)
    copy_split(images, labels, output, train_stems, "train")
    copy_split(images, labels, output, val_stems, "val")
    write_data_yaml(output / "data.yaml", output, class_names)

    print(f"\nwrote {output}")
    print(f"  train: {len(train_stems)} frames")
    print(f"  val  : {len(val_stems)} frames")
    print(f"  yaml : {output / 'data.yaml'}")
    print(f"\ntrain with:\n  python training/train.py --data \"{output / 'data.yaml'}\"")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
