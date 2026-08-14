"""Turns the CVAT-reviewed boxes into an actual train/val/test dataset that
Ultralytics can train on.

Where the labels come from: auto_label.py guesses boxes -> you fix them up by
hand in CVAT -> export as "YOLO 1.1" -> that export is data/detect/cvat_import.zip.
This script reads straight out of that zip file (no need to unzip it first),
drops the 6 duplicate photos notebooks/dataset_audit.ipynb flagged, shuffles
what's left with a fixed seed (so re-running this gives the same split every
time), and copies everything into images/ and labels/ folders. Also writes
data.yaml so notebooks/train_finder.ipynb knows where to find things.

Run notebooks/dataset_audit.ipynb first if data/audit_report.json doesn't
exist yet, then:
    python3 scripts/split_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
import zipfile
from pathlib import Path

import yaml

from common import REPO_ROOT, ensure_dir

DEFAULT_CLASSES = ["leaf"]


def duplicates_to_drop(audit_report: Path) -> set[str]:
    if not audit_report.exists():
        print(f"No audit report at {audit_report} -- skipping dedup. "
              "Run notebooks/dataset_audit.ipynb first to dedupe.")
        return set()
    report = json.loads(audit_report.read_text())
    drop = set()
    for group in report.get("duplicate_groups", []):
        drop.update(group[1:])  # first copy in each group stays, the rest get dropped
    return drop


def split_indices(n: int, val_frac: float, test_frac: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    # shuffle with a fixed seed so this is reproducible -- same photos land in
    # the same split every time we run this, instead of a fresh random split
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    n_val = round(n * val_frac)
    n_test = round(n * test_frac)
    val_idx = idx[:n_val]
    test_idx = idx[n_val:n_val + n_test]
    train_idx = idx[n_val + n_test:]
    return train_idx, val_idx, test_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cvat-zip", type=Path,
                         default=REPO_ROOT / "data" / "detect" / "cvat_import.zip")
    parser.add_argument("--audit-report", type=Path,
                         default=REPO_ROOT / "data" / "audit_report.json")
    parser.add_argument("--detect-root", type=Path, default=REPO_ROOT / "data" / "detect")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.cvat_zip.exists():
        raise SystemExit(f"No CVAT export at {args.cvat_zip}")

    drop = duplicates_to_drop(args.audit_report)

    with zipfile.ZipFile(args.cvat_zip) as zf:
        # CVAT's "YOLO 1.1" export puts every image + its .txt label under
        # obj_train_data/ -- grab the filenames (without extension) of just the images
        names = zf.namelist()
        stems = sorted({
            Path(n).stem for n in names
            if n.startswith("obj_train_data/") and n.endswith(".jpg")
        })
        stems = [s for s in stems if f"{s}.jpg" not in drop]

        train_idx, val_idx, test_idx = split_indices(
            len(stems), args.val_frac, args.test_frac, args.seed
        )
        splits = {"train": train_idx, "val": val_idx, "test": test_idx}

        for split, indices in splits.items():
            img_dir = ensure_dir(args.detect_root / "images" / split)
            lbl_dir = ensure_dir(args.detect_root / "labels" / split)
            for i in indices:
                stem = stems[i]
                (img_dir / f"{stem}.jpg").write_bytes(
                    zf.read(f"obj_train_data/{stem}.jpg")
                )
                (lbl_dir / f"{stem}.txt").write_bytes(
                    zf.read(f"obj_train_data/{stem}.txt")
                )

    data_yaml = args.detect_root / "data.yaml"
    config = {
        "path": str(args.detect_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(DEFAULT_CLASSES)},
    }
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))

    print(f"Total labeled images: {len(stems)} (dropped {len(drop)} duplicates)")
    for split, indices in splits.items():
        print(f"  {split}: {len(indices)}")
    print(f"Wrote {data_yaml}")


if __name__ == "__main__":
    main()
