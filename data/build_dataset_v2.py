"""Build a traceable, deduplicated four-class dataset V2."""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
CLASS_MAP = {
    "unripe": "verde",
    "ripe": "maduro",
    "overripe": "sobre_maduro",
    "rotten": "podrido",
}
SPLIT_MAP = {"train": "train", "valid": "validation", "test": "test"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def files(root: Path, source_dataset: str):
    for source_split, output_split in SPLIT_MAP.items():
        split_dir = root / source_split
        if not split_dir.is_dir():
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if class_dir.name not in CLASS_MAP or not class_dir.is_dir():
                continue
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in EXTENSIONS:
                    yield source_dataset, source_split, output_split, class_dir.name, path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external", type=Path, default=ROOT / "data/external/Banana Ripeness Classification Dataset")
    parser.add_argument("--current", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "data/dataset_v2")
    args = parser.parse_args()

    # Current data wins deterministically when an external image is byte-identical.
    candidates = list(files(args.current, "current")) + list(files(args.external, "external"))
    seen: set[str] = set()
    rows = []
    for source_dataset, source_split, output_split, original_class, source_path in candidates:
        file_hash = digest(source_path)
        if file_hash in seen:
            continue
        try:
            with Image.open(source_path) as image:
                image.verify()
            with Image.open(source_path) as image:
                width, height = image.size
        except (UnidentifiedImageError, OSError, ValueError):
            continue
        seen.add(file_hash)
        target_class = CLASS_MAP[original_class]
        target_dir = args.output / output_split / target_class
        target_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{file_hash[:16]}{source_path.suffix.lower()}"
        destination = target_dir / filename
        shutil.copy2(source_path, destination)
        rows.append({
            "split": output_split,
            "class_name": target_class,
            "original_class": original_class,
            "source_dataset": source_dataset,
            "source_split": source_split,
            "source_path": source_path.resolve().relative_to(ROOT).as_posix(),
            "image_path": destination.resolve().relative_to(ROOT).as_posix(),
            "sha256": file_hash,
            "width": width,
            "height": height,
        })

    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        fieldnames = list(rows[0]) if rows else ["split", "class_name", "original_class", "source_dataset", "source_split", "source_path", "image_path", "sha256", "width", "height"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} unique images to {args.output}")


if __name__ == "__main__":
    main()
