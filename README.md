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
a small dataset — `notebooks/train_checker.ipynb` enables Ultralytics' built-in
`randaugment` by default for the same reason.

**Not yet implemented, but a natural upgrade path:** replacing `yolo26-cls` in Stage 2
with a purpose-built ResNeSt + Channel/Spatial Attention classifier matching the
paper's architecture. This isn't justified yet with the current dataset size (146
positive images, no negatives in hand at time of writing) — `yolo26-cls` is proven,
fast to train, and shares tooling with Stage 1. Worth revisiting once more data (and a
multi-class need, e.g. distinguishing several look-alike species) exists.

## Current Status

**Stage 1 (Finder)**
- Trained on the CVAT-reviewed boxes, evaluated on a genuinely held-out test set (14 photos never touched during training)
- Test mAP50: **0.854**, test mAP50-95: **0.754**
- Known weakness: less confident on dense, overlapping leaf clusters than on isolated leaves — not yet addressed

**Stage 2 (Checker)**
Early training runs all reported a suspiciously perfect 100% accuracy — turned out to be the model learning shortcuts instead of species. Four separate shortcuts were found and fixed, one at a time, all in `scripts/prepare_classifier_data.py`:

1. **Photo composition** — positives were whole cluttered plant photos, negatives were isolated studio photos. Trivially separable without looking at species. Fixed by cropping positives down to single leaves before training.
2. **Resampling fingerprint** — negatives had been resized/compressed one extra time compared to positives, leaving a detectable artifact. Fixed by routing both classes through an identical final resize/save step.
3. **Background-blend seam** — only negatives were ever composited onto a new background, so the synthetic edge itself was a giveaway. Fixed by running both classes through the identical segment-and-composite step.
4. **Dense-cluster segmentation gaps** — some leaves (in dense, overlapping clusters) have no real background nearby to segment against. Added a confidence check plus a cruder fallback segmentation method, so both classes get consistent treatment instead of one class occasionally leaking its untouched original.

Also added a proper held-out **test split** (previously only train/val — val is exactly what checkpoint selection optimizes against, so it was never a neutral number).

**Final, trustworthy result: 96.8% accuracy on 125 held-out test images.**
- Every mistake was a false positive on the negative class — it never missed a real ashwagandha leaf
- Expected, given negatives are outnumbered roughly 5:1 by positives

**Checkpoints**
- Both stages now have trained checkpoints saved in `weights/` (`finder.pt`, `checker.pt`)
- The full two-stage pipeline is runnable end to end for the first time, via `notebooks/pipeline_demo.ipynb`

**What's next**
- Stage 2 has only ever seen **one** other species as "not ashwagandha" — the current number is trustworthy but represents an easy case
- Priority: source 1-2 more negative species, ideally ones that look similar to ashwagandha, before trusting this beyond the current dataset

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

# 5. Crop individual leaves out of the raw photos using the trained Finder --
#    Stage 2 needs to train on the same kind of small single-leaf crop it'll
#    actually be fed at inference time, not the whole cluttered photo
python3 scripts/crop_positives.py --weights runs/finder/train/weights/best.pt

# 6. Build the Stage 2 classification dataset (needs negatives, see above)
python3 scripts/prepare_classifier_data.py

# 7. Train the Checker -- open this in Jupyter (same checkpointing/resume as the Finder)
notebooks/train_checker.ipynb

# 8. Try the full two-stage pipeline on new images -- open this in Jupyter
notebooks/pipeline_demo.ipynb
```

## Repo layout

```
data/
  raw/ashwagandha/       source images (committed)
  raw/negatives/         other-species images for Stage 2 (populated)
  detect/cvat_import.zip the CVAT-reviewed box labels (committed, not regeneratable)
  detect/images|labels/, crops/, classify/   generated training datasets (gitignored,
                                      rebuild with split_dataset.py / crop_positives.py / prepare_classifier_data.py)
weights/                 model checkpoints (gitignored, see weights/README.md)
notebooks/
  dataset_audit.ipynb    dataset dedupe/dimension/corruption audit, with duplicate photos shown inline
  train_finder.ipynb     fine-tunes the Stage 1 detector, auto-resumes if interrupted, checks the test set
  train_checker.ipynb    fine-tunes the Stage 2 classifier, same auto-resume treatment
  pipeline_demo.ipynb    runs both stages on sample photos and displays the results inline
scripts/
  auto_label.py          Stage 1 pseudo-labeling for bootstrapping box labels
  split_dataset.py       turns the CVAT export into a train/val/test dataset
  crop_positives.py      crops individual leaves out of the raw photos using the trained Finder
  prepare_classifier_data.py   builds the Stage 2 train/val/test folder structure
```

Why the split: `notebooks/` is for stuff you want to look at while it runs --
poking at duplicate photos, eyeballing pipeline output. `scripts/` is for stuff
that just needs to run start-to-finish unattended, especially the two training
scripts, which can take a long time and don't need a browser tab open the
whole time.
