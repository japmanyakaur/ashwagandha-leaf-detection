"""Crops individual ashwagandha leaves out of the raw whole-plant photos, using
the trained Finder -- so Stage 2 trains on the same kind of small, single-leaf
image it'll actually be handed at inference time, not a whole cluttered photo.

Why this exists: a whole plant-in-a-pot photo and an isolated single-leaf
photo are trivially separable by composition alone (background clutter,
image size, number of leaves visible) -- a classifier can hit ~100% "accuracy"
without ever learning anything about what makes a leaf specifically
ashwagandha. Training Stage 2 straight off data/raw/ashwagandha/ was doing
exactly that. This crops the same way scripts/pipeline_infer.py does at
inference time (same minimum-size filter), plus a PAD_FRAC margin around
each box -- an edge-to-edge crop was leaving prepare_classifier_data.py's
GrabCut step no background pixels to segment against, which made that step
fail (and silently skip compositing) on ~35% of positives. If
pipeline_infer.py doesn't pad boxes the same way at inference time, this
crop geometry no longer matches production exactly -- worth reconciling
once the padding fix is confirmed to help.

    python3 scripts/crop_positives.py --weights runs/finder/train/weights/best.pt

Then point prepare_classifier_data.py at the output (this is already its default):
    python3 scripts/prepare_classifier_data.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from common import REPO_ROOT, ensure_dir, list_images

PAD_FRAC = 0.12  # margin added around each detected box so GrabCut segmentation
                  # (in prepare_classifier_data.py) has real background pixels to
                  # work with, instead of an edge-to-edge leaf crop


def duplicates_to_drop(audit_report: Path) -> set[str]:
    if not audit_report.exists():
        return set()
    report = json.loads(audit_report.read_text())
    drop = set()
    for group in report.get("duplicate_groups", []):
        drop.update(group[1:])  # same rule as split_dataset.py -- keep one copy, drop the rest
    return drop


def crop_leaves(weights: Path, source: Path, out_dir: Path, conf: float, imgsz: int,
                 audit_report: Path) -> None:
    drop = duplicates_to_drop(audit_report)
    images = [p for p in list_images(source) if p.name not in drop]
    if not images:
        raise SystemExit(f"No images found in {source}")

    ensure_dir(out_dir)
    model = YOLO(str(weights))

    n_crops = 0
    for image_path in images:
        image = Image.open(image_path).convert("RGB")
        boxes = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0].boxes
        if boxes is None:
            continue
        for i, xyxy in enumerate(boxes.xyxy.tolist()):
            x1, y1, x2, y2 = xyxy
            pad_x, pad_y = (x2 - x1) * PAD_FRAC, (y2 - y1) * PAD_FRAC
            x1 = int(max(0, x1 - pad_x))
            y1 = int(max(0, y1 - pad_y))
            x2 = int(min(image.width, x2 + pad_x))
            y2 = int(min(image.height, y2 + pad_y))
            crop = image.crop((x1, y1, x2, y2))
            if crop.width < 2 or crop.height < 2:
                continue  # sliver of a box, not worth keeping
            crop.save(out_dir / f"{image_path.stem}_{i}.jpg")
            n_crops += 1

    print(f"Processed {len(images)} source photos (duplicates already dropped).")
    print(f"Wrote {n_crops} cropped leaf images to {out_dir}")
    if images:
        print(f"Average {n_crops / len(images):.1f} crops per photo.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--weights", type=Path, required=True,
                         help="Trained Finder checkpoint, e.g. runs/finder/train/weights/best.pt")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "raw" / "ashwagandha")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "crops" / "ashwagandha")
    parser.add_argument("--audit-report", type=Path,
                         default=REPO_ROOT / "data" / "audit_report.json")
    parser.add_argument("--conf", type=float, default=0.5,
                         help="Higher than the Finder's normal recall-oriented threshold -- "
                              "this builds training data, so only confident boxes should count")
    parser.add_argument("--imgsz", type=int, default=1280)
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"Weights not found at {args.weights}")

    crop_leaves(args.weights, args.source, args.out_dir, args.conf, args.imgsz, args.audit_report)


if __name__ == "__main__":
    main()
