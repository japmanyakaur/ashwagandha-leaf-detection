# ashwagandha-leaf-detection

A computer vision system that finds *Ashwagandha* (*Withania somnifera*) leaves
specifically — not just any leaf — in cluttered real-world photos (soil, pots, other
foliage). Extends a prior generic leaf detector (single `leaf` class, YOLO26-based)
with species-level discrimination.

## Architecture: two stages

A generic detector can only say "this looks leaf-shaped" — it has no species concept.
Species ID needs a dedicated second model, and it benefits from seeing an
already-isolated leaf rather than a cluttered scene. So the pipeline is split in two:

1. **Finder (Stage 1, detection)** — a YOLO26 object detector that draws boxes around
   anything leaf-shaped in a cluttered photo. Reuses/fine-tunes the prior
   `leaf-localisation` model. Tuned for **recall** (low confidence threshold) — false
   positives here get filtered out by Stage 2.
2. **Checker (Stage 2, classification)** — a YOLO26-cls binary classifier
   (`ashwagandha` vs `not_ashwagandha`) that looks inside each box the Finder proposed.
   By this point the leaf is already cropped out of the cluttered background, so this
   is close to a controlled, single-leaf classification problem. Tuned for
   **precision** (higher confidence threshold).

Splitting it this way also splits the data requirements: Stage 1 needs boxed
cluttered-scene photos (expensive to label), while Stage 2 just needs whole-image
species labels (cheap — no boxes required).

### Research reference

Stage 2's design is informed by Azadnia et al. 2024, ["Medicinal and poisonous plants
classification from visual characteristics of leaves using computer vision and deep
neural networks"](https://doi.org/10.1016/j.ecoinf.2024.102683) (*Ecological
Informatics*), which classifies medicinal/poisonous/weed leaves at ~99.6% accuracy
using a ResNeSt backbone with Channel Attention + Spatial Attention ("SCAM-Herb") and
Fast AutoAugment. Two things carry over directly: their setting is single-leaf,
plain-background classification (which is what Stage 2 sees, thanks to Stage 1's
cropping) and their reliance on strong automated augmentation to fight overfitting on
a small dataset — `scripts/train_checker.py` enables Ultralytics' built-in
`randaugment` by default for the same reason.

**Not yet implemented, but a natural upgrade path:** replacing `yolo26-cls` in Stage 2
with a purpose-built ResNeSt + Channel/Spatial Attention classifier matching the
paper's architecture. This isn't justified yet with the current dataset size (146
positive images, no negatives in hand at time of writing) — `yolo26-cls` is proven,
fast to train, and shares tooling with Stage 1. Worth revisiting once more data (and a
multi-class need, e.g. distinguishing several look-alike species) exists.

## Current Plan

STEP 1
146 Ashwagandha images
        ↓
STEP 2
Add existing best.pt
        ↓
STEP 3
Use best.pt to AUTO-LABEL leaves
        ↓
STEP 4
 manually check/correct those boxes
        ↓
STEP 5
Create proper train/val/test dataset
        ↓
STEP 6
Fine-tune YOLO26m
        ↓
STEP 7
Check Precision / Recall / mAP50 / mAP50-95
        ↓
STEP 8
Find what the model gets wrong
        ↓
STEP 9
Add hard negatives + improve dataset
        ↓
STEP 10
Train again
        ↓
STEP 11
Only THEN decide whether
Checker / SCAM-Herb is needed


## Setup

```bash
pip install -r requirements.txt
```

Requires a CUDA GPU for practical training times (auto-detected by
`torch`/`ultralytics`; falls back to CPU otherwise).

## Usage

```bash
# 1. Audit the raw dataset (duplicates, dimensions, corruption) -- open this in Jupyter
notebooks/dataset_audit.ipynb

# 2. Bootstrap Stage 1 box labels by running the prior leaf-localisation model
#    over the raw photos to get first-draft boxes
python3 scripts/auto_label.py --weights weights/best_m.pt
#    -> review data/detect/preview/*.jpg, fix the boxes up in CVAT, export as
#       "YOLO 1.1" format, save the export as data/detect/cvat_import.zip

# 3. Turn the CVAT export into an actual train/val/test dataset
python3 scripts/split_dataset.py

# 4. Fine-tune the Finder -- open this in Jupyter (checkpoints every epoch,
#    auto-resumes if interrupted, checks the test set and shows the plots inline)
notebooks/train_finder.ipynb

# 5. Build the Stage 2 classification dataset (needs negatives, see above)
python3 scripts/prepare_classifier_data.py

# 6. Train the Checker
python3 scripts/train_checker.py

# 7. Try the full two-stage pipeline on new images -- open this in Jupyter
notebooks/pipeline_demo.ipynb
```

## Repo layout

```
data/
  raw/ashwagandha/       source images (committed)
  raw/negatives/         other-species images for Stage 2 (not yet populated)
  detect/cvat_import.zip the CVAT-reviewed box labels (committed, not regeneratable)
  detect/images|labels/, classify/   generated training datasets (gitignored,
                                      rebuild with split_dataset.py / prepare_classifier_data.py)
weights/                 model checkpoints (gitignored, see weights/README.md)
notebooks/
  dataset_audit.ipynb    dataset dedupe/dimension/corruption audit, with duplicate photos shown inline
  train_finder.ipynb     fine-tunes the Stage 1 detector, auto-resumes if interrupted, checks the test set
  pipeline_demo.ipynb    runs both stages on sample photos and displays the results inline
scripts/
  auto_label.py          Stage 1 pseudo-labeling for bootstrapping box labels
  split_dataset.py       turns the CVAT export into a train/val/test dataset
  prepare_classifier_data.py   builds the Stage 2 train/val folder structure
  train_checker.py       fine-tunes the Stage 2 YOLO26-cls classifier
```

Why the split: `notebooks/` is for stuff you want to look at while it runs --
poking at duplicate photos, eyeballing pipeline output. `scripts/` is for stuff
that just needs to run start-to-finish unattended, especially the two training
scripts, which can take a long time and don't need a browser tab open the
whole time.
