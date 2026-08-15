# Ashwagandha-Leaf-Detection

A computer vision system that finds *Ashwagandha* (*Withania somnifera*) leaves
specifically  in cluttered real-world photos (soil, pots, other
foliage). Extends a prior generic leaf detector (single `leaf` class, YOLO26-based)
with species-level discrimination.

## Architecture: two stages

The pipeline is split in two:

1. **Finder (Stage 1, detection)** — a YOLO26 object detector that draws boxes around
    leaf in a cluttered photo. Fine-tunes the prior
   `leaf-localisation` model. 
2. **Checker (Stage 2, classification)** — a YOLO26-cls binary classifier
   (`ashwagandha` vs `not_ashwagandha`) that looks inside each box the Finder proposed.
   By this point the leaf is already cropped out of the cluttered background, so this
   is close to a controlled, single-leaf classification problem. Tuned for
   **precision** (higher confidence threshold).

### Research reference

Stage 2's design is informed by Azadnia et al. 2024, ["Medicinal and poisonous plants
classification from visual characteristics of leaves using computer vision and deep
neural networks"](https://doi.org/10.1016/j.ecoinf.2024.102683) (*Ecological
Informatics*), which classifies medicinal/poisonous/weed leaves at ~99.6% accuracy
using a ResNeSt backbone with Channel Attention + Spatial Attention ("SCAM-Herb") and
Fast AutoAugment. 

## Current Plan

STEP 1 <br>
146 Ashwagandha images <br>
        ↓<br>
STEP 2<br>
Add existing best.pt<br>
        ↓<br>
STEP <br>
Use best.pt to AUTO-LABEL leaves<br>
        ↓<br>
STEP 4<br>
 manually check/correct those boxes<br>
        ↓<br>
STEP 5<br>
Create proper train/val/test dataset<br>
        ↓<br>
STEP 6<br>
Fine-tune YOLO26m<br>
        ↓<br>
STEP 7<br>
Check Precision / Recall / mAP50 / mAP50-95<br>
        ↓<br>
STEP 8<br>
Find what the model gets wrong<br>
        ↓<br>
STEP 9<br>
Add hard negatives + improve dataset<br>
        ↓<br>
STEP 10<br>
Train again<br>
        ↓<br>
STEP 11<br>
Decide whether Checker / SCAM-Herb is needed<br>
<br>


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

