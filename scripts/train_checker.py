"""Trains Stage 2 (the Checker) -- a plain ashwagandha vs. not_ashwagandha
classifier that runs on the crops Stage 1 hands it.

Randaugment is on by default since our dataset is small and augmentation is the
cheapest way to fight overfitting -- same reasoning the Azadnia et al. 2024 paper
uses (that's the paper the README cites for the two-stage idea in general). We're
NOT using their actual ResNeSt + attention model here though, just plain
yolo26-cls. The fancier architecture is a "maybe later, once we have more data"
idea, not something built.

Needs scripts/prepare_classifier_data.py to have already been run (which itself
needs negative/other-species photos to exist -- we don't have any yet).

    python3 scripts/train_checker.py
    python3 scripts/train_checker.py --model-size s --epochs 80
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "classify",
                         help="Classification dataset root (default: data/classify)")
    parser.add_argument("--weights", type=Path, default=None,
                         help="Starting checkpoint override; defaults to pretrained yolo26{size}-cls.pt")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="m",
                         help="Used only if --weights is not given (default: m)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=-1, help="-1 for Ultralytics auto-batch")
    parser.add_argument("--auto-augment", default="randaugment",
                         choices=["randaugment", "autoaugment", "augmix", "none"])
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "checker")
    parser.add_argument("--name", default="train")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.weights and not args.weights.exists():
        raise SystemExit(f"Weights not found at {args.weights}")

    for split in ("train", "val"):
        split_dir = args.data_dir / split
        if not split_dir.is_dir() or not any(split_dir.iterdir()):
            raise SystemExit(
                f"{split_dir} is missing or empty. Run "
                "scripts/prepare_classifier_data.py first to build the classifier dataset."
            )

    start_weights = str(args.weights) if args.weights else f"yolo26{args.model_size}-cls.pt"
    model = YOLO(start_weights)
    model.train(
        data=str(args.data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        auto_augment=None if args.auto_augment == "none" else args.auto_augment,
        project=str(args.project),
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
