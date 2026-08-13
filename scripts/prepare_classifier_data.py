"""Builds the Stage 2 (Checker) dataset -- sorts photos into
train/ashwagandha, train/not_ashwagandha, val/ashwagandha, val/not_ashwagandha,
which is just the folder layout Ultralytics wants for a classifier.

Positive photos come from data/raw/ashwagandha/. Negative photos need to live
under data/raw/negatives/<source_name>/ -- one folder per species or source
(e.g. data/raw/negatives/plantdoc/), and every subfolder in there gets lumped
together into one "not_ashwagandha" class. We don't have any of these yet, so
this script will just refuse to run until some get added (see data/README.md).

Uses symlinks instead of actually copying the photos, so running this doesn't
eat up double the disk space.

    python3 scripts/prepare_classifier_data.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from common import REPO_ROOT, ensure_dir, list_images

SEED = 42
VAL_FRACTION = 0.15


def dedupe_by_audit_report(images: list[Path], report_path: Path) -> list[Path]:
    if not report_path.exists():
        return images
    report = json.loads(report_path.read_text())
    drop = set()
    for group in report.get("duplicate_groups", []):
        drop.update(group[1:])  # same rule as split_dataset.py -- keep one copy, drop the rest
    return [p for p in images if p.name not in drop]


def collect_negatives(negatives_root: Path) -> list[Path]:
    if not negatives_root.is_dir():
        return []
    images = []
    for sub in sorted(p for p in negatives_root.iterdir() if p.is_dir()):
        images.extend(list_images(sub))
    return images


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def split(images: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    shuffled = images[:]
    random.Random(SEED).shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction)) if shuffled else 0
    return shuffled[n_val:], shuffled[:n_val]


def populate(class_name: str, images: list[Path], classify_root: Path) -> None:
    train_imgs, val_imgs = split(images, VAL_FRACTION)
    for split_name, split_imgs in (("train", train_imgs), ("val", val_imgs)):
        out_dir = ensure_dir(classify_root / split_name / class_name)
        for img in split_imgs:
            link_or_copy(img, out_dir / img.name)
    print(f"  {class_name}: {len(train_imgs)} train / {len(val_imgs)} val "
          f"({len(images)} total)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives-dir", type=Path,
                         default=REPO_ROOT / "data" / "raw" / "ashwagandha")
    parser.add_argument("--negatives-dir", type=Path,
                         default=REPO_ROOT / "data" / "raw" / "negatives")
    parser.add_argument("--audit-report", type=Path,
                         default=REPO_ROOT / "data" / "audit_report.json")
    parser.add_argument("--out-dir", type=Path,
                         default=REPO_ROOT / "data" / "classify")
    args = parser.parse_args()

    positives = dedupe_by_audit_report(list_images(args.positives_dir), args.audit_report)
    negatives = collect_negatives(args.negatives_dir)

    if not positives:
        raise SystemExit(f"No positive images found in {args.positives_dir}")

    if not negatives:
        raise SystemExit(
            f"No negative-class images found under {args.negatives_dir}.\n"
            "Stage 2 needs a 'not_ashwagandha' class to train a real classifier -- "
            "add other-species images (e.g. PlantDoc) as data/raw/negatives/<species>/*.jpg "
            "before running this script. See data/README.md."
        )

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    print("Building classifier dataset:")
    populate("ashwagandha", positives, args.out_dir)
    populate("not_ashwagandha", negatives, args.out_dir)
    print(f"\nDataset ready at {args.out_dir}")


if __name__ == "__main__":
    main()
