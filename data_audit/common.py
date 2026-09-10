from __future__ import annotations
import csv
import hashlib
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def image_records(dataset_dir: Path):
    for split in ("train", "test"):
        root = dataset_dir / split
        if not root.exists():
            continue
        for state_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for image in sorted(state_dir.rglob("*")):
                if image.suffix.lower() in IMAGE_EXTENSIONS:
                    yield {"path": image.resolve().as_posix(), "original_split": split, "state": state_dir.name}

def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
