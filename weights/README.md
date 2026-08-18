# Weights

This directory is gitignored (`weights/*.pt`) — checkpoints are large binaries and
belong outside version control, same convention as the prior `leaf-localisation` project.

| File | Used by | Source |
|---|---|---|
| `best_m.pt` | `scripts/auto_label.py`, starting point for `notebooks/train_finder.ipynb` | Prior `leaf-localisation` project's trained YOLO26 detector — gives Stage 1 a head start instead of training from COCO weights. Not this project's own output. |
| `finder.pt` | `notebooks/pipeline_demo.ipynb` | This project's trained Stage 1 checkpoint — copy of `runs/finder/train/weights/best.pt`. Test mAP50 0.854 / mAP50-95 0.754. |
| `checker.pt` | `notebooks/pipeline_demo.ipynb` | This project's trained Stage 2 checkpoint — copy of `runs/checker/train-9/weights/best.pt`. Test accuracy 96.8% (125 images, one negative species). |

`finder.pt` and `checker.pt` are copied here by hand once a run looks solid, so
`pipeline_demo.ipynb` has a stable path that survives a `runs/` cleanup or a later
training run. The `runs/` folder each came from is kept alongside it (`runs/finder/train/`,
`runs/checker/train-9/`) for the full logs, plots, and confusion matrices.

If a checkpoint here goes missing, `notebooks/train_finder.ipynb` and
`notebooks/train_checker.ipynb` fall back to Ultralytics' pretrained
`yolo26{size}.pt` / `yolo26{size}-cls.pt` COCO weights (auto-downloaded),
defaulting to size `m`.
