"""Build the Stage 2 (Checker) classification dataset.

Positives are cropped single-leaf images from data/crops/ashwagandha/.
Negatives are sourced from data/raw/negatives/<source>/.

To reduce dataset-specific visual cues, both classes undergo the same
segmentation, background compositing, resizing, and JPEG encoding pipeline.
Negative leaves are composited onto background patches sampled from the
ashwagandha images.

Segmentation uses GrabCut with a color-based fallback when confidence is
insufficient. The same procedure is applied to both classes to maintain
consistent preprocessing and minimize class-correlated artifacts.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from common import REPO_ROOT, ensure_dir, list_images

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15  
FINAL_SIZE = 224    
JPEG_QUALITY = 95
CLARITY_THRESHOLD = 0.70  


def dedupe_by_audit_report(images: list[Path], report_path: Path) -> list[Path]:
    if not report_path.exists():
        return images
    report = json.loads(report_path.read_text())
    drop = set()
    for group in report.get("duplicate_groups", []):
        drop.update(group[1:])  # keeps the first name in each group, drop the rest
    return [p for p in images if p.name not in drop]


def collect_negatives(negatives_root: Path) -> list[Path]:
    if not negatives_root.is_dir():
        return []
    images = []
    for sub in sorted(p for p in negatives_root.iterdir() if p.is_dir()):
        images.extend(list_images(sub))
    return images


def _is_background_like(patch_bgr: np.ndarray) -> bool:
    # soil/mulch in the ashwagandha photos is warm brown/tan; leaves are
    # strongly green. Reject any candidate patch whose hue falls in the green
    # band with real saturation, so the background pool doesn't get
    # contaminated with leaf material.
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 60)
    return green.mean() < 0.15


def build_background_pool(source_dir: Path, count: int, patch_size: int,
                           rng: random.Random) -> list[np.ndarray]:
    photos = list_images(source_dir)
    if not photos:
        raise SystemExit(f"No background source photos found in {source_dir}")

    pool: list[np.ndarray] = []
    attempts = 0
    max_attempts = count * 40
    while len(pool) < count and attempts < max_attempts:
        attempts += 1
        img = cv2.imread(str(rng.choice(photos)))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h <= patch_size or w <= patch_size:
            continue
        x = rng.randint(0, w - patch_size)
        y = rng.randint(0, h - patch_size)
        patch = img[y:y + patch_size, x:x + patch_size]
        if _is_background_like(patch):
            pool.append(patch)

    if not pool:
        raise SystemExit(
            f"Couldn't find any background-like patches in {source_dir} -- "
            "the green-hue filter in _is_background_like may need loosening."
        )
    return pool


def _color_threshold_mask(image_bgr: np.ndarray, strict: bool) -> np.ndarray | None:
# Fallback for cases where GrabCut's rectangle-margin assumption is unreliable,
# such as leaves from dense or overlapping clusters with limited visible background.
# This method classifies pixels independently using hue and saturation, avoiding
# reliance on a background border.
#
# The strict threshold follows _is_background_like's green-band logic and is
# preferred for its lower false-positive rate on real soil/background. A looser
# threshold is used only as a final fallback for cases where disease-related
# desaturation or hue variation prevents the strict threshold from separating
# foreground and background reliably.
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    if strict:
        green = (h >= 35) & (h <= 85) & (s >= 60)
    else:
        green = (h >= 30) & (h <= 100) & (s >= 25)
    fg = green.astype(np.float32)
    frac = fg.mean()
    if frac < 0.05 or frac > 0.97:
        return None  
    return np.clip(cv2.GaussianBlur(fg, (0, 0), sigmaX=2.5), 0.0, 1.0)


def _segment_with_color_threshold(image_bgr: np.ndarray) -> np.ndarray | None:
    mask = _color_threshold_mask(image_bgr, strict=True)
    if mask is None:
        mask = _color_threshold_mask(image_bgr, strict=False)
    return mask


def segment_leaf(image_bgr: np.ndarray) -> np.ndarray | None:
    """Returns a soft alpha mask (float32, 0..1, same HxW as image_bgr) isolating the
    leaf from its backdrop, or None if no method can find one -- which by then means
    the crop is genuinely all-background or all-leaf, so there's nothing to swap out
    either way (see module docstring, "fourth giveaway")."""
    h, w = image_bgr.shape[:2]
    margin = 0.06
    rect = (int(w * margin), int(h * margin), int(w * (1 - 2 * margin)), int(h * (1 - 2 * margin)))

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(image_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return _segment_with_color_threshold(image_bgr)

    fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    certain = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_BGD), 1.0, 0.0)
    frac = fg.mean()
    clarity = certain.mean()
    if frac < 0.05 or frac > 0.97 or clarity < CLARITY_THRESHOLD:
     
        return _segment_with_color_threshold(image_bgr)

    alpha = cv2.GaussianBlur(fg, (0, 0), sigmaX=2.5)
    return np.clip(alpha, 0.0, 1.0)


def composite_on_background(leaf_bgr: np.ndarray, alpha: np.ndarray,
                             background_pool: list[np.ndarray],
                             rng: random.Random) -> np.ndarray:
    h, w = leaf_bgr.shape[:2]
    bg = rng.choice(background_pool)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    alpha_3c = alpha[:, :, None]
    out = alpha_3c * leaf_bgr.astype(np.float32) + (1 - alpha_3c) * bg.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def segment_and_composite(background_pool: list[np.ndarray],
                           rng: random.Random) -> Callable[[Path, Path], None]:

    stats = {"composited": 0, "fallback": 0, "fallback_files": []}

    def _write(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()

        image_bgr = cv2.imread(str(src))
        alpha = segment_leaf(image_bgr) if image_bgr is not None else None
        if alpha is not None:
            composited = composite_on_background(image_bgr, alpha, background_pool, rng)
            rgb = cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(rgb)
            stats["composited"] += 1
        else:
            im = Image.open(src).convert("RGB")
            stats["fallback"] += 1
            stats["fallback_files"].append(src.name)
        im.resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS).save(dst, quality=JPEG_QUALITY)

    _write.stats = stats  # type: ignore[attr-defined]
    return _write


def crop_source_photo(path: Path) -> str:
    # crop_positives.py names crops "<source_stem>_<box_index>.jpg" -- group on the
    # source stem so several crops of the same leaf can't land on both sides of the
    # split (they're near-duplicates: same lighting, background, often overlapping
    # regions, so a val crop from a photo already seen in train isn't a real test)
    return path.stem.rsplit("_", 1)[0]


def one_group_per_image(path: Path) -> str:
    return path.stem


def split(images: list[Path], val_fraction: float, test_fraction: float,
          group_key: Callable[[Path], str]) -> tuple[list[Path], list[Path], list[Path]]:
    groups: dict[str, list[Path]] = {}
    for img in images:
        groups.setdefault(group_key(img), []).append(img)

    keys = list(groups.keys())
    random.Random(SEED).shuffle(keys)
    n_val = max(1, round(len(keys) * val_fraction)) if keys else 0
    n_test = max(1, round(len(keys) * test_fraction)) if keys else 0
    val_keys = set(keys[:n_val])
    test_keys = set(keys[n_val:n_val + n_test])

    train_imgs = [img for k, imgs in groups.items()
                  if k not in val_keys and k not in test_keys for img in imgs]
    val_imgs = [img for k, imgs in groups.items() if k in val_keys for img in imgs]
    test_imgs = [img for k, imgs in groups.items() if k in test_keys for img in imgs]
    return train_imgs, val_imgs, test_imgs


def populate(class_name: str, images: list[Path], classify_root: Path,
             group_key: Callable[[Path], str],
             write: Callable[[Path, Path], None]) -> None:
    train_imgs, val_imgs, test_imgs = split(images, VAL_FRACTION, TEST_FRACTION, group_key)
    for split_name, split_imgs in (("train", train_imgs), ("val", val_imgs), ("test", test_imgs)):
        out_dir = ensure_dir(classify_root / split_name / class_name)
        for img in split_imgs:
            write(img, out_dir / img.name)
    print(f"  {class_name}: {len(train_imgs)} train / {len(val_imgs)} val / "
          f"{len(test_imgs)} test ({len(images)} total)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives-dir", type=Path,
                         default=REPO_ROOT / "data" / "crops" / "ashwagandha")
    parser.add_argument("--negatives-dir", type=Path,
                         default=REPO_ROOT / "data" / "raw" / "negatives")
    parser.add_argument("--background-source", type=Path,
                         default=REPO_ROOT / "data" / "raw" / "ashwagandha",
                         help="Raw photos to sample soil/background patches from when "
                              "recompositing both classes (see segment_and_composite)")
    parser.add_argument("--audit-report", type=Path,
                         default=REPO_ROOT / "data" / "audit_report.json")
    parser.add_argument("--out-dir", type=Path,
                         default=REPO_ROOT / "data" / "classify")
    args = parser.parse_args()

    # positives come from cropped single-leaf images, not the raw whole-plant photos --
    # a classifier trained on whole photos vs. isolated leaves just learns to tell those
    # two photography styles apart, not ashwagandha vs. other species (see scripts/crop_positives.py)
    positives = dedupe_by_audit_report(list_images(args.positives_dir), args.audit_report)
    negatives = collect_negatives(args.negatives_dir)

    if not positives:
        raise SystemExit(
            f"No positive images found in {args.positives_dir}.\n"
            "Run scripts/crop_positives.py first to generate cropped ashwagandha leaf "
            "images from data/raw/ashwagandha/ using the trained Finder."
        )

    if not negatives:
        raise SystemExit(
            f"No negative-class images found under {args.negatives_dir}.\n"
            "Stage 2 needs a 'not_ashwagandha' class to train a real classifier -- "
            "add other-species images (e.g. PlantDoc) as data/raw/negatives/<species>/*.jpg "
            "before running this script. See data/README.md."
        )

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    print("Building classifier dataset:")
    print("  building background patch pool from", args.background_source)
    background_pool = build_background_pool(
        args.background_source, count=250, patch_size=200, rng=random.Random(SEED),
    )

    # both classes go through the identical segment+recomposite pipeline (separate
    # rng streams so they don't draw the exact same sequence of patches) so the
    # blend-seam artifact can't correlate with the label -- see module docstring
    positives_write = segment_and_composite(background_pool, random.Random(SEED))

    populate("ashwagandha", positives, args.out_dir, group_key=crop_source_photo,
             write=positives_write)
    print(f"  ({positives_write.stats['composited']} background-swapped, "
          f"{positives_write.stats['fallback']} fell back to plain resize -- "
          "segmentation looked degenerate on those)")
    if positives_write.stats["fallback_files"]:
        sample = positives_write.stats["fallback_files"][:10]
        print(f"  sample fallback files: {', '.join(sample)}")

    negatives_write = segment_and_composite(background_pool, random.Random(SEED + 1))

    populate("not_ashwagandha", negatives, args.out_dir, group_key=one_group_per_image,
             write=negatives_write)
    print(f"  ({negatives_write.stats['composited']} background-swapped, "
          f"{negatives_write.stats['fallback']} fell back to plain resize -- "
          "segmentation looked degenerate on those)")
    if negatives_write.stats["fallback_files"]:
        sample = negatives_write.stats["fallback_files"][:10]
        print(f"  sample fallback files: {', '.join(sample)}")
    print(f"\nDataset ready at {args.out_dir}")


if __name__ == "__main__":
    main()
