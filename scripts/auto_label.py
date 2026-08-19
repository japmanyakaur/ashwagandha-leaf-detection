"""Runs the prior model over the raw ashwagandha photos to get a
first-draft set of bounding boxes. Used for drawing boxes in CVAT, which are then exported 
to YOLO 1.1 format for training the final model.

"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common import REPO_ROOT, ensure_dir, list_images


def auto_label(weights: Path, source: Path, labels_dir: Path, preview_dir: Path,
                conf: float, imgsz: int) -> None:
    images = list_images(source)
    if not images:
        raise SystemExit(f"No images found in {source}")

    ensure_dir(labels_dir)
    ensure_dir(preview_dir)

    model = YOLO(str(weights))

    n_with_boxes = 0
    n_total_boxes = 0

    for image_path in images:
        results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
        result = results[0]

        label_path = labels_dir / f"{image_path.stem}.txt"
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            n_with_boxes += 1
            n_total_boxes += len(boxes)
            lines = []
            for cls_id, xywhn in zip(boxes.cls.tolist(), boxes.xywhn.tolist()):
                cx, cy, w, h = xywhn
                lines.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            label_path.write_text("\n".join(lines) + "\n")
        else:
            label_path.write_text("")

        preview_path = preview_dir / image_path.name
        result.save(filename=str(preview_path))

    print(f"Processed {len(images)} images.")
    print(f"Images with >=1 proposed box: {n_with_boxes}")
    print(f"Total proposed boxes: {n_total_boxes}")
    print(f"Labels written to:  {labels_dir}")
    print(f"Previews written to: {preview_dir}")
    print("\nNext: look through the previews, fix up the boxes in CVAT, export as")
    print("YOLO 1.1 format to data/detect/cvat_import.zip, then run split_dataset.py.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True,
                         help="Path to a detector checkpoint, e.g. weights/leaf_localisation.pt")
    parser.add_argument("--source", type=Path, default=REPO_ROOT / "data" / "raw" / "ashwagandha",
                         help="Directory of raw images to label (default: data/raw/ashwagandha)")
    parser.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data" / "detect" / "labels" / "auto",
                         help="Where to write YOLO-format .txt label files")
    parser.add_argument("--preview-dir", type=Path, default=REPO_ROOT / "data" / "detect" / "preview",
                         help="Where to write annotated preview images for review")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(
            f"Weights not found at {args.weights}. Drop a detector checkpoint there first "
            "(see weights/README.md)."
        )

    auto_label(args.weights, args.source, args.labels_dir, args.preview_dir, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
