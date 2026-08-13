"""Small stuff that every other script in here needs, so it lives in one place
instead of getting copy-pasted five times: listing image files, hashing a file
to check for duplicates, making a folder if it doesn't exist yet."""

from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REPO_ROOT = Path(__file__).resolve().parent.parent


def list_images(directory: Path) -> list[Path]:
    """All image files directly inside `directory` (not recursive), sorted so
    the order is always the same between runs."""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def md5_of_file(path: Path, chunk_size: int = 1 << 16) -> str:
    # hash the actual bytes, not the filename -- this is how we catch photos
    # that got saved twice under two different names
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    # makes the folder (and any parents) if needed, no error if it's already there
    path.mkdir(parents=True, exist_ok=True)
    return path
