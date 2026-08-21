"""Small stuff that every other script in here needs, so it lives in one place
instead of getting copy-pasted five times: listing image files, hashing a file
to check for duplicates, making a folder if it doesn't exist yet."""

from __future__ import annotations

import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REPO_ROOT = Path(__file__).resolve().parent.parent


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def md5_of_file(path: Path, chunk_size: int = 1 << 16) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
