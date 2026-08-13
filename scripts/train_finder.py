"""The actual fine-tuning step for Stage 1 (the Finder). Mostly just calling
Ultralytics' train() with our dataset -- not much custom logic here.

Pass --weights weights/best_m.pt to start from the old leaf-localisation model
(it already knows what a leaf looks like, so we're just adapting it rather than
teaching it from zero). Skip --weights and it'll fall back to downloading
Ultralytics' COCO-pretrained YOLO26 weights instead, which don't know anything
about leaves specifically -- only use that if best_m.pt isn't around.

Needs data/detect/data.yaml, which means split_dataset.py has to have already
been run. If data.yaml happens to be missing this writes an empty template, but
that's just a fallback -- it does NOT build the images/labels folders for you.

    python3 scripts/train_finder.py --weights weights/best_m.pt
    python3 scripts/train_finder.py --weights weights/best_m.pt --epochs 150
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from common import REPO_ROOT, ensure_dir

DEFAULT_CLASSES = ["leaf"]


def ensure_data_yaml(data_yaml: Path, detect_root: Path) -> Path:
    if data_yaml.exists():
        return data_yaml
    ensure_dir(data_yaml.parent)
    config = {
        "path": str(detect_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(DEFAULT_CLASSES)},
    }
    data_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"No data.yaml found -- wrote a blank template to {data_yaml}.")
    print(f"You still need {detect_root}/images/{{train,val,test}} and "
          f"{detect_root}/labels/{{train,val,test}} to actually exist -- "
          "run scripts/split_dataset.py to build those.")
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "detect" / "data.yaml",
                         help="Path to YOLO detection data.yaml")
    parser.add_argument("--weights", type=Path, default=None,
                         help="Starting checkpoint -- normally weights/best_m.pt")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="m",
                         help="Only matters if --weights isn't given (default: m)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280,
                         help="Kept large so small leaves don't get blurred away in a cluttered photo")
    parser.add_argument("--batch", type=int, default=-1, help="-1 for Ultralytics auto-batch")
    parser.add_argument("--project", type=Path, default=REPO_ROOT / "runs" / "finder")
    parser.add_argument("--name", default="train")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.weights and not args.weights.exists():
        raise SystemExit(f"Weights not found at {args.weights}")

    detect_root = args.data.parent
    data_yaml = ensure_data_yaml(args.data, detect_root)

    start_weights = str(args.weights) if args.weights else f"yolo26{args.model_size}.pt"
    model = YOLO(start_weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        device=args.device,
    )


if __name__ == "__main__":
    main()
