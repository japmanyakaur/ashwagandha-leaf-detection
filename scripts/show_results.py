"""Prints a summary of both stages' current results to the terminal.

Always recomputes live from the actual checkpoints in weights/ rather than
printing hardcoded numbers -- a hardcoded copy is exactly how weights/checker.pt
ended up silently stale (pointing at a two-day-old result) until this project
caught it by hand. Re-running here means what you see is always what the
checkpoint you'd actually use right now really does.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rule(char: str = "-", width: int = 60) -> None:
    print(char * width)


def show_finder_results() -> None:
    weights = REPO_ROOT / "weights" / "finder.pt"
    print("STAGE 1 -- FINDER (detection)")
    _rule()
    if not weights.exists():
        print(f"  no checkpoint at {weights}, skipping")
        return

    model = YOLO(str(weights))
    data_yaml = REPO_ROOT / "data" / "detect" / "data.yaml"
    if not data_yaml.exists():
        print(f"  {data_yaml} not found -- run scripts/split_dataset.py first")
        return

    metrics = model.val(data=str(data_yaml), split="test", verbose=False)
    print(f"  test mAP50:     {metrics.box.map50:.3f}")
    print(f"  test mAP50-95:  {metrics.box.map:.3f}")
    print(f"  test precision: {metrics.box.mp:.3f}")
    print(f"  test recall:    {metrics.box.mr:.3f}")
    print()


def show_checker_results() -> None:
    weights = REPO_ROOT / "weights" / "checker.pt"
    print("STAGE 2 -- CHECKER (classification)")
    _rule()
    if not weights.exists():
        print(f"  no checkpoint at {weights}, skipping")
        return

    data_dir = REPO_ROOT / "data" / "classify"
    if not data_dir.exists():
        print(f"  {data_dir} not found -- run scripts/prepare_classifier_data.py first")
        return

    model = YOLO(str(weights))

    for split in ("val", "test"):
        metrics = model.val(data=str(data_dir), split=split, imgsz=224, verbose=False)
        names = model.names
        print(f"  {split} top-1 accuracy: {metrics.top1*100:.3f}%")
        if hasattr(metrics, "confusion_matrix"):
            matrix = metrics.confusion_matrix.matrix
            print(f"  {split} confusion matrix (rows/cols = {names[0]}, {names[1]}):")
            for row in matrix:
                print("    " + "  ".join(f"{int(v):4d}" for v in row))
        print()


def show_dataset_composition() -> None:
    print("DATASET")
    _rule()
    classify_dir = REPO_ROOT / "data" / "classify"
    if not classify_dir.exists():
        print(f"  {classify_dir} not found")
        return

    for split in ("train", "val", "test"):
        pos = len(list((classify_dir / split / "ashwagandha").glob("*"))) \
            if (classify_dir / split / "ashwagandha").is_dir() else 0
        neg = len(list((classify_dir / split / "not_ashwagandha").glob("*"))) \
            if (classify_dir / split / "not_ashwagandha").is_dir() else 0
        print(f"  {split:5s}: {pos:4d} ashwagandha / {neg:4d} not_ashwagandha")
    print()


def main() -> None:
    print()
    _rule("=")
    print("ASHWAGANDHA LEAF DETECTION -- RESULTS SUMMARY")
    _rule("=")
    print()
    show_finder_results()
    show_checker_results()
    show_dataset_composition()


if __name__ == "__main__":
    sys.exit(main())
