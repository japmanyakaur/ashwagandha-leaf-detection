# Weights

This directory is gitignored (`weights/*.pt`) — checkpoints are large binaries and
belong outside version control, same convention as the prior `leaf-localisation` project.

Drop trained checkpoints here:

| File | Used by | Source |
|---|---|---|
| `leaf_localisation.pt` (or any name) | `scripts/auto_label.py`, `notebooks/train_finder.ipynb` | Prior `leaf-localisation` project's trained YOLO26 detector — gives Stage 1 a head start instead of training from COCO weights. |
| `checker.pt` (or any name) | `notebooks/pipeline_demo.ipynb` | Output of `scripts/train_checker.py` once Stage 2 has been trained. |

If no checkpoint is supplied, `notebooks/train_finder.ipynb` and `scripts/train_checker.py`
fall back to Ultralytics' pretrained `yolo26{size}.pt` / `yolo26{size}-cls.pt` COCO
weights (auto-downloaded), defaulting to size `m`.
