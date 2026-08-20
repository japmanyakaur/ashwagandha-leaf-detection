# Results

## Two-Stage Ashwagandha Leaf Detection Pipeline

The system consists of two stages:

1. **Finder** — localizes individual leaves in the full image.
2. **Checker** — classifies each localized leaf as `ashwagandha` or `not_ashwagandha`.

All reported final test results are from **held-out test sets that were not used for training or checkpoint selection**.

---

## Stage 1 — Finder

**Model:** YOLO26m  
**Task:** Leaf detection  
**Training:** 100 epochs  
**Initialization:** Fine-tuned from a prior leaf-localisation checkpoint

| Metric | Validation | Test |
|---|---:|---:|
| Precision | 0.927 | — |
| Recall | 0.860 | — |
| mAP50 | 0.940 | **0.854** |
| mAP50-95 | 0.816 | **0.754** |

The test set consisted of **14 held-out images**.

> **Known limitation:** The Finder performs better on isolated leaves than on dense, overlapping leaf clusters. This remains a potential area for future improvement.

Test precision and recall were not captured as numerical values; only their plots are available. Therefore, **test mAP50 and mAP50-95 are used as the primary quantitative test results**.

---

## Stage 2 — Checker

**Model:** YOLO26m-cls  
**Task:** Binary classification  
**Classes:** `ashwagandha` / `not_ashwagandha`  
**Training:** 23 epochs with early stopping

| Metric | Validation | Test |
|---|---:|---:|
| Top-1 Accuracy | 1.000 | **0.969** |

### Test Confusion Matrix

| Actual \ Predicted | Ashwagandha | Not Ashwagandha |
|---|---:|---:|
| **Ashwagandha** | **104** | 2 |
| **Not Ashwagandha** | 2 | **22** |

The test set contained **130 images**, with **126 correctly classified**, resulting in **96.9% test accuracy**.

### Look-alike species evaluation

A harder negative class, **_Physalis peruviana_**, was specifically included to test whether the Checker could distinguish Ashwagandha from a visually similar plant rather than relying only on obvious visual differences.

All **4 held-out Physalis test images were correctly classified**.

The remaining 2 false positives in the test set came from the easier negative species.

---

## Dataset

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Ashwagandha | 463 | 89 | 106 | **658** |
| Not Ashwagandha | 109 | 24 | 24 | **157** |
| **Total** | **572** | **113** | **130** | **815** |

### Negative-class composition

The `not_ashwagandha` class contains:

- **125 images** from an initially collected negative species with serrated margins and disease spotting.
- **32 images** of *_Physalis peruviana_*, a Solanaceae species selected as a visually harder negative and added specifically to test generalization against a look-alike species.



- 2 microscope close-ups
- 3 frost-damaged/dead plants

These were excluded because they were not representative of normal healthy leaf imagery.

---

## Final Checkpoints

| Checkpoint | Stage | Test Result |
|---|---|---:|
| `weights/finder.pt` | Stage 1 — Finder | **mAP50: 0.854** |**Precision: 0.927** 
| `weights/checker.pt` | Stage 2 — Checker | **Accuracy: 0.969** |

Full training logs, plots, and evaluation artifacts are available in:

```text
runs/finder/train/
runs/checker/train-3/